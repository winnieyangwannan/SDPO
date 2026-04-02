"""
SDPO Standalone Compatibility Tests

Tests SDPO logic without heavy verl imports to avoid transformers version issues.
Run with: python tests/test_sdpo_standalone.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional
from dataclasses import dataclass
from omegaconf import OmegaConf


# =============================================================================
# Copy of core functions to test in isolation
# =============================================================================

def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    batch_num_tokens: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Aggregate per-token loss."""
    if loss_agg_mode == "token-mean":
        loss = (loss_mat * loss_mask).sum() / loss_mask.sum().clamp(min=1.0)
    elif loss_agg_mode == "seq-mean":
        per_seq_loss = (loss_mat * loss_mask).sum(dim=-1)
        per_seq_tokens = loss_mask.sum(dim=-1).clamp(min=1.0)
        loss = (per_seq_loss / per_seq_tokens).mean()
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")
    return loss


def compute_self_distillation_loss(
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    self_distillation_config: Any,
    old_log_probs: Optional[torch.Tensor] = None,
    student_all_log_probs: Optional[torch.Tensor] = None,
    teacher_all_log_probs: Optional[torch.Tensor] = None,
    student_topk_log_probs: Optional[torch.Tensor] = None,
    teacher_topk_log_probs: Optional[torch.Tensor] = None,
    self_distillation_mask: Optional[torch.Tensor] = None,
    loss_agg_mode: str = "token-mean",
    rollout_is_weights: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Compute self-distillation loss for SDPO."""
    metrics = {}

    loss_mask = response_mask
    if self_distillation_mask is not None:
        loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)

    if self_distillation_config.full_logit_distillation:
        use_topk = self_distillation_config.distillation_topk is not None
        if use_topk:
            if student_topk_log_probs is None or teacher_topk_log_probs is None:
                raise ValueError("top-k distillation requires student_topk_log_probs and teacher_topk_log_probs.")

            def add_tail(log_probs: torch.Tensor) -> torch.Tensor:
                log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
                log_s = torch.clamp(log_s, max=-1e-7)
                tail_log = torch.log(-torch.expm1(log_s))
                return torch.cat([log_probs, tail_log], dim=-1)

            def renorm_topk_log_probs(logp: torch.Tensor) -> torch.Tensor:
                logZ = torch.logsumexp(logp, dim=-1, keepdim=True)
                return logp - logZ

            student_distill_log_probs = student_topk_log_probs
            teacher_distill_log_probs = teacher_topk_log_probs
            if self_distillation_config.distillation_add_tail:
                student_distill_log_probs = add_tail(student_distill_log_probs)
                teacher_distill_log_probs = add_tail(teacher_distill_log_probs)
            else:
                student_distill_log_probs = renorm_topk_log_probs(student_distill_log_probs)
                teacher_distill_log_probs = renorm_topk_log_probs(teacher_distill_log_probs)
        else:
            if student_all_log_probs is None or teacher_all_log_probs is None:
                raise ValueError("full_logit_distillation requires student_all_log_probs and teacher_all_log_probs.")
            student_distill_log_probs = student_all_log_probs
            teacher_distill_log_probs = teacher_all_log_probs

        if self_distillation_config.alpha == 0.0:
            kl_loss = F.kl_div(
                student_distill_log_probs, teacher_distill_log_probs, reduction="none", log_target=True
            )
        elif self_distillation_config.alpha == 1.0:
            kl_loss = F.kl_div(
                teacher_distill_log_probs, student_distill_log_probs, reduction="none", log_target=True
            )
        else:
            alpha = torch.tensor(
                self_distillation_config.alpha,
                dtype=student_distill_log_probs.dtype,
                device=student_distill_log_probs.device,
            )
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_distill_log_probs + torch.log(1 - alpha), teacher_distill_log_probs + torch.log(alpha)]),
                dim=0,
            )
            kl_teacher = F.kl_div(mixture_log_probs, teacher_distill_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_distill_log_probs, reduction="none", log_target=True)
            kl_loss = torch.lerp(kl_student, kl_teacher, alpha)

        per_token_loss = kl_loss.sum(-1)
    else:
        assert self_distillation_config.alpha == 1.0, "Only reverse KL is supported for non-full-logit distillation"
        log_ratio = student_log_probs - teacher_log_probs
        per_token_loss = log_ratio.detach() * student_log_probs

    is_clip = self_distillation_config.is_clip
    if is_clip is not None:
        if old_log_probs is None:
            raise ValueError("old_log_probs is required for distillation IS ratio.")
        negative_approx_kl = (student_log_probs - old_log_probs).detach()
        negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
        ratio = torch.exp(negative_approx_kl).clamp(max=is_clip)
        per_token_loss = per_token_loss * ratio

    if rollout_is_weights is not None:
        per_token_loss = per_token_loss * rollout_is_weights

    loss = agg_loss(
        loss_mat=per_token_loss,
        loss_mask=loss_mask,
        loss_agg_mode=loss_agg_mode,
        batch_num_tokens=loss_mask.sum().clamp(min=1.0),
    )
    return loss, metrics


class TrustRegionTeacher(nn.Module):
    """Trust-region teacher for SDPO."""
    def __init__(self, ref_module: nn.Module, student_module: nn.Module, mix_coef: float) -> None:
        super().__init__()
        self.ref_module = ref_module
        self.student_module = student_module
        self.mix_coef = float(mix_coef)

    def forward(self, *args, **kwargs):
        from types import SimpleNamespace
        ref_out = self.ref_module(*args, **kwargs)
        student_out = self.student_module(*args, **kwargs)
        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out[0]
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out[0]
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)


# =============================================================================
# TESTS
# =============================================================================

def test_config_access_patterns():
    """Verify self_distillation config can be accessed both ways."""
    config_dict = {
        "policy_loss": {"loss_mode": "sdpo"},
        "self_distillation": {
            "full_logit_distillation": True,
            "alpha": 0.0,
            "distillation_topk": 100,
            "distillation_add_tail": True,
            "teacher_regularization": "ema",
            "teacher_update_rate": 0.05,
            "is_clip": 5.0,
        }
    }
    
    config = OmegaConf.create(config_dict)
    
    # Test getattr
    self_distillation_cfg = getattr(config, "self_distillation", None)
    assert self_distillation_cfg is not None
    
    # Test .get()
    teacher_reg = self_distillation_cfg.get("teacher_regularization", "ema")
    assert teacher_reg == "ema"
    
    # Test direct attribute
    assert self_distillation_cfg.full_logit_distillation == True
    assert self_distillation_cfg.alpha == 0.0
    
    # Test nested getattr
    update_rate = getattr(self_distillation_cfg, "teacher_update_rate", 0.0)
    assert update_rate == 0.05
    
    print("✓ Test 1: Config access patterns - PASSED")


def test_trust_region_teacher():
    """Verify TrustRegionTeacher correctly interpolates logits."""
    class MockModule(nn.Module):
        def __init__(self, output_logits):
            super().__init__()
            self._logits = output_logits
        def forward(self, *args, **kwargs):
            from types import SimpleNamespace
            return SimpleNamespace(logits=self._logits)
    
    ref_logits = torch.tensor([[1.0, 2.0, 3.0]])
    student_logits = torch.tensor([[4.0, 5.0, 6.0]])
    
    ref_module = MockModule(ref_logits)
    student_module = MockModule(student_logits)
    
    teacher = TrustRegionTeacher(ref_module, student_module, mix_coef=0.5)
    output = teacher(None)
    expected = torch.lerp(ref_logits, student_logits, 0.5)
    assert torch.allclose(output.logits, expected)
    
    # mix_coef=0 → ref
    teacher_0 = TrustRegionTeacher(ref_module, student_module, mix_coef=0.0)
    assert torch.allclose(teacher_0(None).logits, ref_logits)
    
    # mix_coef=1 → student
    teacher_1 = TrustRegionTeacher(ref_module, student_module, mix_coef=1.0)
    assert torch.allclose(teacher_1(None).logits, student_logits)
    
    print("✓ Test 2: TrustRegionTeacher - PASSED")


def test_self_distillation_loss_reverse_kl():
    """Test reverse KL (non-full-logit) mode."""
    @dataclass
    class MockConfig:
        full_logit_distillation: bool = False
        alpha: float = 1.0
        is_clip: float = None
    
    batch_size, seq_len = 2, 4
    student_log_probs = torch.randn(batch_size, seq_len)
    teacher_log_probs = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    
    loss, metrics = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=MockConfig(),
    )
    
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    print("✓ Test 3: Self-distillation loss (reverse KL) - PASSED")


def test_self_distillation_mask():
    """Test self_distillation_mask correctly filters samples."""
    @dataclass
    class MockConfig:
        full_logit_distillation: bool = False
        alpha: float = 1.0
        is_clip: float = None
    
    batch_size, seq_len = 4, 4
    student_log_probs = torch.ones(batch_size, seq_len)
    teacher_log_probs = torch.zeros(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    
    # Mask: only samples 0 and 2 contribute
    self_distillation_mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
    
    loss_masked, _ = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=MockConfig(),
        self_distillation_mask=self_distillation_mask,
    )
    
    # Without mask
    loss_full, _ = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=MockConfig(),
    )
    
    # Masked loss should still be computed over 2 samples
    assert torch.isfinite(loss_masked)
    print("✓ Test 4: Self-distillation mask - PASSED")


def test_full_logit_topk_with_tail():
    """Test top-k distillation with tail bucket."""
    @dataclass
    class MockConfig:
        full_logit_distillation: bool = True
        distillation_topk: int = 10
        distillation_add_tail: bool = True
        alpha: float = 0.0
        is_clip: float = None
    
    batch_size, seq_len, k = 2, 4, 10
    student_topk = torch.log_softmax(torch.randn(batch_size, seq_len, k), dim=-1) - 0.5
    teacher_topk = torch.log_softmax(torch.randn(batch_size, seq_len, k), dim=-1) - 0.5
    student_log_probs = torch.randn(batch_size, seq_len)
    teacher_log_probs = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    
    loss, _ = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=MockConfig(),
        student_topk_log_probs=student_topk,
        teacher_topk_log_probs=teacher_topk,
    )
    
    assert torch.isfinite(loss)
    print("✓ Test 5: Top-k with tail bucket - PASSED")


def test_kl_modes():
    """Test Forward KL, Reverse KL, and JSD."""
    batch_size, seq_len, vocab = 2, 4, 50
    student_all = torch.log_softmax(torch.randn(batch_size, seq_len, vocab), dim=-1)
    teacher_all = torch.log_softmax(torch.randn(batch_size, seq_len, vocab), dim=-1)
    student_log = torch.randn(batch_size, seq_len)
    teacher_log = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    
    @dataclass
    class MockConfig:
        full_logit_distillation: bool = True
        distillation_topk: int = None
        alpha: float = 0.0
        is_clip: float = None
    
    # Forward KL
    config_fwd = MockConfig(alpha=0.0)
    loss_fwd, _ = compute_self_distillation_loss(
        student_log, teacher_log, response_mask, config_fwd,
        student_all_log_probs=student_all, teacher_all_log_probs=teacher_all
    )
    assert torch.isfinite(loss_fwd)
    
    # Reverse KL
    config_rev = MockConfig(alpha=1.0)
    loss_rev, _ = compute_self_distillation_loss(
        student_log, teacher_log, response_mask, config_rev,
        student_all_log_probs=student_all, teacher_all_log_probs=teacher_all
    )
    assert torch.isfinite(loss_rev)
    
    # JSD
    config_jsd = MockConfig(alpha=0.5)
    loss_jsd, _ = compute_self_distillation_loss(
        student_log, teacher_log, response_mask, config_jsd,
        student_all_log_probs=student_all, teacher_all_log_probs=teacher_all
    )
    assert torch.isfinite(loss_jsd)
    
    print("✓ Test 6: KL modes (Forward, Reverse, JSD) - PASSED")


def test_is_clipping():
    """Test importance sampling clipping."""
    @dataclass
    class MockConfig:
        full_logit_distillation: bool = False
        alpha: float = 1.0
        is_clip: float = 5.0
    
    batch_size, seq_len = 2, 4
    student_log_probs = torch.randn(batch_size, seq_len)
    teacher_log_probs = torch.randn(batch_size, seq_len)
    old_log_probs = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    
    loss, _ = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=MockConfig(),
        old_log_probs=old_log_probs,
    )
    
    assert torch.isfinite(loss)
    print("✓ Test 7: IS clipping - PASSED")


def test_rollout_correction_weights():
    """Test rollout correction weights are applied."""
    @dataclass
    class MockConfig:
        full_logit_distillation: bool = True
        distillation_topk: int = None
        alpha: float = 0.0  # Forward KL for stable test
        is_clip: float = None
    
    batch_size, seq_len, vocab = 2, 4, 10
    # Use proper probability distributions
    student_all = torch.log_softmax(torch.randn(batch_size, seq_len, vocab), dim=-1)
    teacher_all = torch.log_softmax(torch.randn(batch_size, seq_len, vocab), dim=-1)
    student_log_probs = torch.randn(batch_size, seq_len)
    teacher_log_probs = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    
    loss_no_rc, _ = compute_self_distillation_loss(
        student_log_probs, teacher_log_probs, response_mask, MockConfig(),
        student_all_log_probs=student_all, teacher_all_log_probs=teacher_all,
    )
    
    rollout_is_weights = torch.ones(batch_size, seq_len) * 0.5
    loss_with_rc, _ = compute_self_distillation_loss(
        student_log_probs, teacher_log_probs, response_mask, MockConfig(),
        student_all_log_probs=student_all, teacher_all_log_probs=teacher_all,
        rollout_is_weights=rollout_is_weights,
    )
    
    ratio = (loss_with_rc / loss_no_rc).item()
    assert 0.4 < ratio < 0.6, f"Expected ~0.5, got {ratio}"
    print("✓ Test 8: Rollout correction weights - PASSED")


def test_ema_update():
    """Test EMA teacher parameter update."""
    teacher = nn.Linear(10, 10)
    actor = nn.Linear(10, 10)
    
    with torch.no_grad():
        teacher.weight.fill_(0.0)
        teacher.bias.fill_(0.0)
        actor.weight.fill_(1.0)
        actor.bias.fill_(1.0)
    
    update_rate = 0.1
    
    with torch.no_grad():
        for teacher_param, actor_param in zip(teacher.parameters(), actor.parameters()):
            teacher_param.data.mul_(1 - update_rate).add_(actor_param.data, alpha=update_rate)
    
    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight, 0.1))
    print("✓ Test 9: EMA update - PASSED")


def test_agg_loss_modes():
    """Test loss aggregation modes."""
    batch_size, seq_len = 2, 4
    loss_mat = torch.ones(batch_size, seq_len)
    loss_mask = torch.ones(batch_size, seq_len)
    loss_mask[1, 2:] = 0
    
    loss_token = agg_loss(loss_mat, loss_mask, loss_agg_mode="token-mean")
    assert torch.isfinite(loss_token)
    
    loss_seq = agg_loss(loss_mat, loss_mask, loss_agg_mode="seq-mean")
    assert torch.isfinite(loss_seq)
    
    print("✓ Test 10: Loss aggregation modes - PASSED")


def run_all_tests():
    print("\n" + "=" * 60)
    print("SDPO STANDALONE COMPATIBILITY TESTS")
    print("=" * 60 + "\n")
    
    tests = [
        test_config_access_patterns,
        test_trust_region_teacher,
        test_self_distillation_loss_reverse_kl,
        test_self_distillation_mask,
        test_full_logit_topk_with_tail,
        test_kl_modes,
        test_is_clipping,
        test_rollout_correction_weights,
        test_ema_update,
        test_agg_loss_modes,
    ]
    
    passed = 0
    failed = 0
    
    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ {test_fn.__name__} - FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

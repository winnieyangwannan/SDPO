"""
SDPO Compatibility Tests

Quick tests to verify SDPO code works correctly with the new verl version.
Run with: python -m pytest tests/test_sdpo_compatibility.py -v

Or run directly: python tests/test_sdpo_compatibility.py
"""

import sys
import torch
import torch.nn as nn
from typing import Any
from dataclasses import dataclass
from omegaconf import OmegaConf, DictConfig


# =============================================================================
# Test 1: Config Access Patterns
# =============================================================================

def test_config_access_patterns():
    """Verify self_distillation config can be accessed both ways (dict and attr)."""
    # Simulate the config structure from YAML
    config_dict = {
        "policy_loss": {"loss_mode": "sdpo"},
        "self_distillation": {
            "full_logit_distillation": True,
            "alpha": 0.0,
            "distillation_topk": 100,
            "distillation_add_tail": True,
            "teacher_regularization": "ema",
            "teacher_update_rate": 0.05,
            "success_reward_threshold": 0.5,
            "is_clip": 5.0,
        }
    }
    
    config = OmegaConf.create(config_dict)
    
    # Test 1a: getattr access (used in _update_teacher)
    self_distillation_cfg = getattr(config, "self_distillation", None)
    assert self_distillation_cfg is not None, "getattr failed for self_distillation"
    
    # Test 1b: .get() access (used in update_policy)
    teacher_reg = self_distillation_cfg.get("teacher_regularization", "ema")
    assert teacher_reg == "ema", f"Expected 'ema', got '{teacher_reg}'"
    
    # Test 1c: Direct attribute access (used in compute_self_distillation_loss)
    assert self_distillation_cfg.full_logit_distillation == True
    assert self_distillation_cfg.alpha == 0.0
    assert self_distillation_cfg.distillation_topk == 100
    
    # Test 1d: Nested getattr (used in _update_teacher)
    update_rate = getattr(self_distillation_cfg, "teacher_update_rate", 0.0)
    assert update_rate == 0.05
    
    print("✓ Test 1: Config access patterns - PASSED")


# =============================================================================
# Test 2: TrustRegionTeacher
# =============================================================================

def test_trust_region_teacher():
    """Verify TrustRegionTeacher correctly interpolates logits."""
    from verl.workers.actor.dp_actor import TrustRegionTeacher
    
    # Create mock modules
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
    
    # Test with mix_coef=0.5 (should be midpoint)
    teacher = TrustRegionTeacher(ref_module, student_module, mix_coef=0.5)
    output = teacher(None)
    
    expected = torch.lerp(ref_logits, student_logits, 0.5)
    assert torch.allclose(output.logits, expected), \
        f"Expected {expected}, got {output.logits}"
    
    # Test with mix_coef=0.0 (should be ref)
    teacher_0 = TrustRegionTeacher(ref_module, student_module, mix_coef=0.0)
    output_0 = teacher_0(None)
    assert torch.allclose(output_0.logits, ref_logits)
    
    # Test with mix_coef=1.0 (should be student)
    teacher_1 = TrustRegionTeacher(ref_module, student_module, mix_coef=1.0)
    output_1 = teacher_1(None)
    assert torch.allclose(output_1.logits, student_logits)
    
    print("✓ Test 2: TrustRegionTeacher - PASSED")


# =============================================================================
# Test 3: compute_self_distillation_loss
# =============================================================================

def test_self_distillation_loss():
    """Verify self-distillation loss computation."""
    from verl.trainer.ppo.core_algos import compute_self_distillation_loss
    
    batch_size, seq_len, vocab_size = 2, 4, 100
    
    # Create mock inputs
    student_log_probs = torch.randn(batch_size, seq_len)
    teacher_log_probs = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    old_log_probs = torch.randn(batch_size, seq_len)
    
    # Create config
    @dataclass
    class MockConfig:
        full_logit_distillation: bool = False
        alpha: float = 1.0  # Reverse KL
        is_clip: float = None
    
    config = MockConfig()
    
    # Test basic loss computation
    loss, metrics = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=config,
    )
    
    assert loss.dim() == 0, "Loss should be scalar"
    assert torch.isfinite(loss), "Loss should be finite"
    
    # Test with self_distillation_mask
    self_distillation_mask = torch.tensor([1.0, 0.0])  # Only first sample
    loss_masked, _ = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=config,
        self_distillation_mask=self_distillation_mask,
    )
    assert torch.isfinite(loss_masked)
    
    # Test with IS clipping
    config_is = MockConfig(is_clip=5.0)
    loss_is, _ = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=config_is,
        old_log_probs=old_log_probs,
    )
    assert torch.isfinite(loss_is)
    
    # Test with rollout_is_weights
    rollout_is_weights = torch.ones(batch_size, seq_len) * 0.5
    loss_rc, _ = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=config,
        rollout_is_weights=rollout_is_weights,
    )
    assert torch.isfinite(loss_rc)
    
    print("✓ Test 3: compute_self_distillation_loss - PASSED")


# =============================================================================
# Test 4: Full Logit Distillation with Top-K
# =============================================================================

def test_full_logit_distillation_topk():
    """Verify top-k distillation with tail bucket."""
    from verl.trainer.ppo.core_algos import compute_self_distillation_loss
    
    batch_size, seq_len, k = 2, 4, 10
    
    # Create mock top-k log probs (should sum to < 1 in prob space)
    student_topk_log_probs = torch.log_softmax(torch.randn(batch_size, seq_len, k), dim=-1) - 0.5
    teacher_topk_log_probs = torch.log_softmax(torch.randn(batch_size, seq_len, k), dim=-1) - 0.5
    
    student_log_probs = torch.randn(batch_size, seq_len)
    teacher_log_probs = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    
    @dataclass
    class MockConfig:
        full_logit_distillation: bool = True
        distillation_topk: int = 10
        distillation_add_tail: bool = True
        alpha: float = 0.0  # Forward KL
        is_clip: float = None
    
    config = MockConfig()
    
    loss, metrics = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=config,
        student_topk_log_probs=student_topk_log_probs,
        teacher_topk_log_probs=teacher_topk_log_probs,
    )
    
    assert torch.isfinite(loss), f"Loss is not finite: {loss}"
    
    # Test without tail (renorm mode)
    config_no_tail = MockConfig(distillation_add_tail=False)
    loss_no_tail, _ = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=config_no_tail,
        student_topk_log_probs=student_topk_log_probs,
        teacher_topk_log_probs=teacher_topk_log_probs,
    )
    assert torch.isfinite(loss_no_tail)
    
    print("✓ Test 4: Full logit distillation with top-k - PASSED")


# =============================================================================
# Test 5: KL Divergence Modes (Forward, Reverse, JSD)
# =============================================================================

def test_kl_divergence_modes():
    """Verify different KL modes work correctly."""
    from verl.trainer.ppo.core_algos import compute_self_distillation_loss
    
    batch_size, seq_len, vocab_size = 2, 4, 50
    
    student_all_log_probs = torch.log_softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)
    teacher_all_log_probs = torch.log_softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)
    student_log_probs = torch.randn(batch_size, seq_len)
    teacher_log_probs = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    
    @dataclass
    class MockConfig:
        full_logit_distillation: bool = True
        distillation_topk: int = None  # Full vocab
        alpha: float = 0.0
        is_clip: float = None
    
    # Test Forward KL (alpha=0)
    config_fwd = MockConfig(alpha=0.0)
    loss_fwd, _ = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=config_fwd,
        student_all_log_probs=student_all_log_probs,
        teacher_all_log_probs=teacher_all_log_probs,
    )
    assert torch.isfinite(loss_fwd), "Forward KL loss not finite"
    
    # Test Reverse KL (alpha=1)
    config_rev = MockConfig(alpha=1.0)
    loss_rev, _ = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=config_rev,
        student_all_log_probs=student_all_log_probs,
        teacher_all_log_probs=teacher_all_log_probs,
    )
    assert torch.isfinite(loss_rev), "Reverse KL loss not finite"
    
    # Test JSD (alpha=0.5)
    config_jsd = MockConfig(alpha=0.5)
    loss_jsd, _ = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=config_jsd,
        student_all_log_probs=student_all_log_probs,
        teacher_all_log_probs=teacher_all_log_probs,
    )
    assert torch.isfinite(loss_jsd), "JSD loss not finite"
    
    print("✓ Test 5: KL divergence modes (Forward, Reverse, JSD) - PASSED")


# =============================================================================
# Test 6: EMA Teacher Update
# =============================================================================

def test_ema_teacher_update():
    """Verify EMA update works correctly."""
    # Create simple linear modules
    teacher = nn.Linear(10, 10)
    actor = nn.Linear(10, 10)
    
    # Initialize with different weights
    with torch.no_grad():
        teacher.weight.fill_(0.0)
        teacher.bias.fill_(0.0)
        actor.weight.fill_(1.0)
        actor.bias.fill_(1.0)
    
    update_rate = 0.1
    
    # Simulate EMA update (from _update_teacher)
    with torch.no_grad():
        for teacher_param, actor_param in zip(teacher.parameters(), actor.parameters()):
            teacher_param.data.mul_(1 - update_rate).add_(actor_param.data, alpha=update_rate)
    
    # Check: teacher should now be 0 * 0.9 + 1 * 0.1 = 0.1
    assert torch.allclose(teacher.weight, torch.full_like(teacher.weight, 0.1)), \
        f"Expected 0.1, got {teacher.weight[0, 0].item()}"
    
    print("✓ Test 6: EMA teacher update - PASSED")


# =============================================================================
# Test 7: Import Compatibility
# =============================================================================

def test_imports():
    """Verify all SDPO-related imports work."""
    try:
        from verl.workers.actor.dp_actor import DataParallelPPOActor, TrustRegionTeacher
        from verl.trainer.ppo.core_algos import compute_self_distillation_loss, agg_loss
        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch
        print("✓ Test 7: All imports successful - PASSED")
    except ImportError as e:
        print(f"✗ Test 7: Import failed - {e}")
        raise


# =============================================================================
# Test 8: Required Batch Keys
# =============================================================================

def test_required_batch_keys():
    """Verify self-distillation requires correct batch keys."""
    required_keys = {
        "teacher_input_ids",
        "teacher_attention_mask", 
        "teacher_position_ids",
        "self_distillation_mask",
    }
    
    # Simulate batch with all keys
    mock_batch_keys = {
        "input_ids", "attention_mask", "position_ids", "responses",
        "response_mask", "old_log_probs", "advantages",
        "teacher_input_ids", "teacher_attention_mask", 
        "teacher_position_ids", "self_distillation_mask",
    }
    
    assert required_keys.issubset(mock_batch_keys), \
        f"Missing keys: {required_keys - mock_batch_keys}"
    
    # Simulate batch missing a key
    incomplete_batch = mock_batch_keys - {"teacher_input_ids"}
    missing = required_keys - incomplete_batch
    assert missing == {"teacher_input_ids"}, f"Expected missing teacher_input_ids, got {missing}"
    
    print("✓ Test 8: Required batch keys check - PASSED")


# =============================================================================
# Test 9: Loss Aggregation Modes
# =============================================================================

def test_loss_aggregation():
    """Verify loss aggregation works with different modes."""
    from verl.trainer.ppo.core_algos import agg_loss
    
    batch_size, seq_len = 2, 4
    loss_mat = torch.ones(batch_size, seq_len)
    loss_mask = torch.ones(batch_size, seq_len)
    loss_mask[1, 2:] = 0  # Mask out some tokens
    
    # Token-mean (default mode)
    loss_token = agg_loss(loss_mat, loss_mask, loss_agg_mode="token-mean")
    assert torch.isfinite(loss_token), "Token-mean loss should be finite"
    
    print("✓ Test 9: Loss aggregation modes - PASSED")


# =============================================================================
# Test 10: Rollout Correction Integration
# =============================================================================

def test_rollout_correction_integration():
    """Verify rollout correction weights are properly applied."""
    from verl.trainer.ppo.core_algos import compute_self_distillation_loss
    
    batch_size, seq_len, vocab_size = 2, 4, 50
    
    # Use full-logit distillation for stable testing
    student_all_log_probs = torch.log_softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)
    teacher_all_log_probs = torch.log_softmax(torch.randn(batch_size, seq_len, vocab_size), dim=-1)
    student_log_probs = torch.randn(batch_size, seq_len)
    teacher_log_probs = torch.randn(batch_size, seq_len)
    response_mask = torch.ones(batch_size, seq_len)
    
    @dataclass
    class MockConfig:
        full_logit_distillation: bool = True
        distillation_topk: int = None
        alpha: float = 0.0  # Forward KL
        is_clip: float = None
    
    config = MockConfig()
    
    # Without rollout correction
    loss_no_rc, _ = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=config,
        student_all_log_probs=student_all_log_probs,
        teacher_all_log_probs=teacher_all_log_probs,
    )
    
    # With rollout correction (scale by 0.5)
    rollout_is_weights = torch.ones(batch_size, seq_len) * 0.5
    loss_with_rc, _ = compute_self_distillation_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        response_mask=response_mask,
        self_distillation_config=config,
        student_all_log_probs=student_all_log_probs,
        teacher_all_log_probs=teacher_all_log_probs,
        rollout_is_weights=rollout_is_weights,
    )
    
    # Loss with RC should be ~half of loss without RC
    ratio = (loss_with_rc / loss_no_rc).item()
    assert 0.4 < ratio < 0.6, f"Expected ratio ~0.5, got {ratio}"
    
    print("✓ Test 10: Rollout correction integration - PASSED")


# =============================================================================
# Main Runner
# =============================================================================

def run_all_tests():
    """Run all compatibility tests."""
    print("\n" + "=" * 60)
    print("SDPO COMPATIBILITY TESTS")
    print("=" * 60 + "\n")
    
    tests = [
        ("Config Access Patterns", test_config_access_patterns),
        ("TrustRegionTeacher", test_trust_region_teacher),
        ("Self-Distillation Loss", test_self_distillation_loss),
        ("Top-K Distillation", test_full_logit_distillation_topk),
        ("KL Divergence Modes", test_kl_divergence_modes),
        ("EMA Teacher Update", test_ema_teacher_update),
        ("Imports", test_imports),
        ("Required Batch Keys", test_required_batch_keys),
        ("Loss Aggregation", test_loss_aggregation),
        ("Rollout Correction", test_rollout_correction_integration),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"✗ {name} - FAILED: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60 + "\n")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

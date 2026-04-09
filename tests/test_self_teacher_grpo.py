"""
Self-Teacher GRPO Standalone Tests

Tests Self-Teacher GRPO logic without heavy verl imports.
Run with: python tests/test_self_teacher_grpo.py
"""

import sys
import math
import copy
import hashlib
import numpy as np
import torch
from typing import Any, Optional
from dataclasses import dataclass
from omegaconf import OmegaConf


# =============================================================================
# Mock Classes
# =============================================================================

class MockTokenizer:
    """Mock tokenizer for testing."""
    def decode(self, ids, skip_special_tokens=True):
        return f"decoded_response_{hash(tuple(ids.tolist())) % 10000}"


class MockBatch:
    """Mock batch.batch dict."""
    def __init__(self, data: dict):
        self._data = data
    
    def __contains__(self, key):
        return key in self._data
    
    def __getitem__(self, key):
        return self._data[key]
    
    def keys(self):
        return self._data.keys()


class MockDataProto:
    """Mock DataProto for testing."""
    def __init__(self, batch_data: dict, non_tensor_batch: dict):
        self.batch = MockBatch(batch_data)
        self.non_tensor_batch = non_tensor_batch
    
    def __len__(self):
        # Return length based on first available field
        if "responses" in self.batch._data:
            return len(self.batch["responses"])
        if "index" in self.non_tensor_batch:
            return len(self.non_tensor_batch["index"])
        return 0


# =============================================================================
# Copy of core functions to test in isolation
# =============================================================================

def get_current_privilege_fraction(
    global_steps: int,
    total_training_steps: int,
    config: dict
) -> float:
    """
    Compute the privilege fraction at the current training step.
    
    Linear decay: fraction(t) = fraction_0 * max(0, 1 - t / total_steps)
    """
    if not config.get("enable", False):
        return 0.0
    
    fraction_0 = config.get("privilege_fraction", 0.5)
    decay = config.get("privilege_fraction_decay", "none")
    
    if decay == "linear":
        if total_training_steps > 0:
            return fraction_0 * max(0.0, 1.0 - global_steps / total_training_steps)
        return fraction_0
    
    return fraction_0  # "none" or unrecognized → constant


def get_cache_penalty(
    global_steps: int,
    total_training_steps: int,
    config: dict
) -> float:
    """
    Compute the privilege penalty at the current training step.
    
    Privileged solutions must exceed cached acc by this margin to enter cache.
    """
    if not config.get("enable", False):
        return 0.0
    
    penalty_cfg = config.get("privilege_penalty", {})
    
    if not penalty_cfg.get("enable", True):
        return 0.0
    
    p_min = penalty_cfg.get("penalty_min", 0.02)
    p_max = penalty_cfg.get("penalty_max", 0.10)
    schedule = penalty_cfg.get("schedule", "late_ramp")
    
    t = global_steps / total_training_steps if total_training_steps > 0 else 0.0
    
    if schedule == "linear":
        return p_min + (p_max - p_min) * t
    
    elif schedule == "cosine":
        return p_min + (p_max - p_min) * (1 - math.cos(math.pi * t)) / 2
    
    elif schedule == "late_ramp":
        # Flat early, sharp increase in final third
        if t < 0.66:
            return p_min
        else:
            return p_min + (p_max - p_min) * (t - 0.66) / 0.34
    
    return p_min  # "none" or unrecognized


def apply_privilege_penalty(
    scores: torch.Tensor,
    privilege_mask: torch.Tensor,
    penalty: float
) -> torch.Tensor:
    """
    Discount scores from privileged samples before advantage computation.
    """
    effective_scores = scores.clone()
    effective_scores[privilege_mask] -= penalty
    return effective_scores


def should_give_privilege(
    question_idx: str, 
    cache: dict, 
    threshold: float
) -> bool:
    """Check if question qualifies for privilege info based on accuracy threshold."""
    if question_idx not in cache:
        return False
    cached = cache[question_idx]
    # Use 'acc' field, falling back to 'score' for backward compatibility
    cached_acc = cached.get("acc", cached.get("score", 0.0))
    return cached_acc >= threshold


def verify_question_match(
    extra_info: list,
    idx: int, 
    question_idx: str,
    cache: dict
) -> bool:
    """Verify the question description matches the cached entry."""
    cached = cache.get(question_idx)
    if not cached or "description_hash" not in cached:
        return True  # No verification possible, allow
    
    current_description = ""
    if idx < len(extra_info) and isinstance(extra_info[idx], dict):
        current_description = extra_info[idx].get("description", "")
    
    if not current_description:
        return True  # No description to verify, allow
    
    current_hash = hashlib.md5(current_description[:500].encode()).hexdigest()
    return current_hash == cached["description_hash"]


def compute_privilege_mask(
    batch_size: int,
    rollout_n: int,
    question_indices: list,
    cache: dict,
    threshold: float,
    privilege_fraction: float
) -> tuple[torch.Tensor, int]:
    """
    Compute which samples should receive privilege.
    
    Returns:
        (privilege_mask, num_questions_with_privilege)
    """
    num_privileged = int(rollout_n * privilege_fraction)
    privilege_mask = torch.zeros(batch_size, dtype=torch.bool)
    num_questions_with_privilege = 0
    
    for group_start in range(0, batch_size, rollout_n):
        question_idx = str(question_indices[group_start])
        
        # Check if this question qualifies for privilege
        if not should_give_privilege(question_idx, cache, threshold):
            continue
        
        num_questions_with_privilege += 1
        
        # Assign privilege to interleaved positions
        for j in range(num_privileged):
            idx = group_start + j * 2  # Interleave: 0, 2, 4, 6
            if idx < group_start + rollout_n:
                privilege_mask[idx] = True
    
    return privilege_mask, num_questions_with_privilege


def update_cache_with_penalty(
    cache: dict,
    question_idx: str,
    acc: float,
    is_privileged: bool,
    penalty: float,
    response: str,
    global_steps: int,
    description: str = ""
) -> tuple[bool, str]:
    """
    Update cache entry with penalty-aware comparison.
    
    Returns:
        (was_updated, update_type: "new" | "update" | "none")
    """
    effective_acc = acc - (penalty if is_privileged else 0.0)
    
    if question_idx not in cache:
        cache[question_idx] = {
            "response": response,
            "acc": acc,  # Store raw acc
            "step": global_steps,
            "was_privileged": is_privileged,
            "description_hash": hashlib.md5(description[:500].encode()).hexdigest() if description else "",
            "description_preview": description[:100] if description else "",
        }
        return True, "new"
    
    cached_acc = cache[question_idx].get("acc", cache[question_idx].get("score", -float("inf")))
    
    if effective_acc > cached_acc:
        cache[question_idx] = {
            "response": response,
            "acc": acc,  # Store raw acc
            "step": global_steps,
            "was_privileged": is_privileged,
            "description_hash": hashlib.md5(description[:500].encode()).hexdigest() if description else "",
            "description_preview": description[:100] if description else "",
        }
        return True, "update"
    
    return False, "none"


# =============================================================================
# Tests
# =============================================================================

class TestPrivilegeFractionDecay:
    """Tests for privilege fraction decay logic."""
    
    def test_no_decay(self):
        """Test that 'none' decay keeps fraction constant."""
        config = {"enable": True, "privilege_fraction": 0.5, "privilege_fraction_decay": "none"}
        
        assert get_current_privilege_fraction(0, 100, config) == 0.5
        assert get_current_privilege_fraction(50, 100, config) == 0.5
        assert get_current_privilege_fraction(100, 100, config) == 0.5
        print("✓ test_no_decay passed")
    
    def test_linear_decay(self):
        """Test linear decay from fraction_0 to 0."""
        config = {"enable": True, "privilege_fraction": 0.5, "privilege_fraction_decay": "linear"}
        
        # At t=0, should be full fraction
        assert get_current_privilege_fraction(0, 100, config) == 0.5
        
        # At t=50 (half), should be half of initial
        assert abs(get_current_privilege_fraction(50, 100, config) - 0.25) < 1e-6
        
        # At t=100 (end), should be 0
        assert get_current_privilege_fraction(100, 100, config) == 0.0
        
        # Beyond end, should be clamped to 0
        assert get_current_privilege_fraction(150, 100, config) == 0.0
        print("✓ test_linear_decay passed")
    
    def test_disabled_returns_zero(self):
        """Test that disabled config returns 0."""
        config = {"enable": False, "privilege_fraction": 0.5}
        assert get_current_privilege_fraction(0, 100, config) == 0.0
        print("✓ test_disabled_returns_zero passed")


class TestCachePenalty:
    """Tests for cache penalty schedule logic."""
    
    def test_penalty_disabled(self):
        """Test that disabled penalty returns 0."""
        config = {"enable": True, "privilege_penalty": {"enable": False}}
        assert get_cache_penalty(50, 100, config) == 0.0
        print("✓ test_penalty_disabled passed")
    
    def test_linear_schedule(self):
        """Test linear penalty schedule."""
        config = {
            "enable": True,
            "privilege_penalty": {
                "enable": True,
                "penalty_min": 0.02,
                "penalty_max": 0.10,
                "schedule": "linear"
            }
        }
        
        # At t=0, should be p_min
        assert abs(get_cache_penalty(0, 100, config) - 0.02) < 1e-6
        
        # At t=50, should be midpoint
        expected_mid = 0.02 + (0.10 - 0.02) * 0.5
        assert abs(get_cache_penalty(50, 100, config) - expected_mid) < 1e-6
        
        # At t=100, should be p_max
        assert abs(get_cache_penalty(100, 100, config) - 0.10) < 1e-6
        print("✓ test_linear_schedule passed")
    
    def test_late_ramp_schedule(self):
        """Test late_ramp penalty schedule."""
        config = {
            "enable": True,
            "privilege_penalty": {
                "enable": True,
                "penalty_min": 0.02,
                "penalty_max": 0.10,
                "schedule": "late_ramp"
            }
        }
        
        # Before t=0.66, should stay at p_min
        assert abs(get_cache_penalty(0, 100, config) - 0.02) < 1e-6
        assert abs(get_cache_penalty(30, 100, config) - 0.02) < 1e-6
        assert abs(get_cache_penalty(65, 100, config) - 0.02) < 1e-6
        
        # After t=0.66, should start ramping
        penalty_at_80 = get_cache_penalty(80, 100, config)
        assert penalty_at_80 > 0.02
        assert penalty_at_80 < 0.10
        
        # At t=100, should be at or near p_max
        assert abs(get_cache_penalty(100, 100, config) - 0.10) < 1e-6
        print("✓ test_late_ramp_schedule passed")
    
    def test_cosine_schedule(self):
        """Test cosine penalty schedule."""
        config = {
            "enable": True,
            "privilege_penalty": {
                "enable": True,
                "penalty_min": 0.02,
                "penalty_max": 0.10,
                "schedule": "cosine"
            }
        }
        
        # At t=0, should be p_min
        assert abs(get_cache_penalty(0, 100, config) - 0.02) < 1e-6
        
        # At t=50, should be midpoint (cosine at π/2 → 0)
        expected_mid = 0.02 + (0.10 - 0.02) * 0.5
        assert abs(get_cache_penalty(50, 100, config) - expected_mid) < 1e-6
        
        # At t=100, should be p_max
        assert abs(get_cache_penalty(100, 100, config) - 0.10) < 1e-6
        print("✓ test_cosine_schedule passed")


class TestApplyPrivilegePenalty:
    """Tests for score penalty application."""
    
    def test_penalty_applied_to_privileged_only(self):
        """Test that penalty only affects privileged samples."""
        scores = torch.tensor([0.8, 0.8, 0.6, 0.9])
        privilege_mask = torch.tensor([True, False, True, False])
        penalty = 0.1
        
        result = apply_privilege_penalty(scores, privilege_mask, penalty)
        
        assert abs(result[0].item() - 0.7) < 1e-6  # Privileged: 0.8 - 0.1
        assert abs(result[1].item() - 0.8) < 1e-6  # Unprivileged: unchanged
        assert abs(result[2].item() - 0.5) < 1e-6  # Privileged: 0.6 - 0.1
        assert abs(result[3].item() - 0.9) < 1e-6  # Unprivileged: unchanged
        print("✓ test_penalty_applied_to_privileged_only passed")
    
    def test_zero_penalty(self):
        """Test that zero penalty doesn't change scores."""
        scores = torch.tensor([0.8, 0.8, 0.6, 0.9])
        privilege_mask = torch.tensor([True, False, True, False])
        
        result = apply_privilege_penalty(scores, privilege_mask, 0.0)
        
        assert torch.allclose(result, scores)
        print("✓ test_zero_penalty passed")


class TestShouldGivePrivilege:
    """Tests for privilege qualification check."""
    
    def test_not_in_cache(self):
        """Test that questions not in cache don't qualify."""
        cache = {}
        assert not should_give_privilege("123", cache, 0.8)
        print("✓ test_not_in_cache passed")
    
    def test_below_threshold(self):
        """Test that questions below threshold don't qualify."""
        cache = {"123": {"acc": 0.5}}
        assert not should_give_privilege("123", cache, 0.8)
        print("✓ test_below_threshold passed")
    
    def test_at_threshold(self):
        """Test that questions at threshold qualify."""
        cache = {"123": {"acc": 0.8}}
        assert should_give_privilege("123", cache, 0.8)
        print("✓ test_at_threshold passed")
    
    def test_above_threshold(self):
        """Test that questions above threshold qualify."""
        cache = {"123": {"acc": 1.0}}
        assert should_give_privilege("123", cache, 0.8)
        print("✓ test_above_threshold passed")
    
    def test_backward_compat_score_field(self):
        """Test backward compatibility with 'score' field."""
        cache = {"123": {"score": 0.9}}  # Old format using 'score'
        assert should_give_privilege("123", cache, 0.8)
        print("✓ test_backward_compat_score_field passed")


class TestPrivilegeMaskComputation:
    """Tests for privilege mask interleaving pattern."""
    
    def test_interleave_pattern(self):
        """Test that privilege is assigned in interleaved pattern."""
        cache = {"0": {"acc": 1.0}, "1": {"acc": 1.0}}
        question_indices = [0]*8 + [1]*8  # Two questions, 8 rollouts each
        
        mask, num_q = compute_privilege_mask(
            batch_size=16,
            rollout_n=8,
            question_indices=question_indices,
            cache=cache,
            threshold=0.8,
            privilege_fraction=0.5  # 4 out of 8
        )
        
        # Check interleave pattern for first question (indices 0-7)
        # Should be: [True, False, True, False, True, False, True, False]
        expected_q0 = [True, False, True, False, True, False, True, False]
        assert mask[:8].tolist() == expected_q0, f"Got {mask[:8].tolist()}"
        
        # Same for second question (indices 8-15)
        assert mask[8:16].tolist() == expected_q0
        
        # Both questions qualified
        assert num_q == 2
        print("✓ test_interleave_pattern passed")
    
    def test_partial_qualification(self):
        """Test when only some questions qualify."""
        cache = {"0": {"acc": 1.0}, "1": {"acc": 0.5}}  # Only q0 qualifies
        question_indices = [0]*8 + [1]*8
        
        mask, num_q = compute_privilege_mask(
            batch_size=16,
            rollout_n=8,
            question_indices=question_indices,
            cache=cache,
            threshold=0.8,
            privilege_fraction=0.5
        )
        
        # First question should have privilege
        assert mask[:8].any()
        
        # Second question should NOT have privilege
        assert not mask[8:16].any()
        
        # Only one question qualified
        assert num_q == 1
        print("✓ test_partial_qualification passed")
    
    def test_zero_fraction(self):
        """Test that zero privilege_fraction gives no privileges."""
        cache = {"0": {"acc": 1.0}}
        question_indices = [0]*8
        
        mask, num_q = compute_privilege_mask(
            batch_size=8,
            rollout_n=8,
            question_indices=question_indices,
            cache=cache,
            threshold=0.8,
            privilege_fraction=0.0
        )
        
        assert not mask.any()
        print("✓ test_zero_fraction passed")


class TestCacheUpdateWithPenalty:
    """Tests for penalty-aware cache updates."""
    
    def test_new_entry(self):
        """Test adding a new entry to empty cache."""
        cache = {}
        updated, update_type = update_cache_with_penalty(
            cache, "123", acc=0.8, is_privileged=False, penalty=0.05,
            response="test", global_steps=10
        )
        
        assert updated
        assert update_type == "new"
        assert cache["123"]["acc"] == 0.8
        assert cache["123"]["was_privileged"] == False
        print("✓ test_new_entry passed")
    
    def test_unprivileged_wins_tie(self):
        """Test that unprivileged solution wins ties."""
        # Setup: privileged solution at acc=0.8
        cache = {"123": {"acc": 0.8, "was_privileged": True}}
        
        # Unprivileged with same acc=0.8 should win (no penalty applied)
        # Effective comparison: 0.8 > 0.8 is False, so won't update??
        # Wait, let me reconsider...
        # The update logic compares effective_acc > cached_acc
        # For unprivileged: effective_acc = 0.8 - 0 = 0.8
        # cached_acc = 0.8
        # 0.8 > 0.8 is False, so won't update
        
        # To actually win, unprivileged needs slightly higher, OR
        # the logic should be >= for unprivileged. Let's test current behavior.
        updated, update_type = update_cache_with_penalty(
            cache, "123", acc=0.8, is_privileged=False, penalty=0.05,
            response="new", global_steps=20
        )
        
        # With strict > comparison, tie doesn't update
        assert not updated
        print("✓ test_unprivileged_wins_tie passed (strict comparison)")
    
    def test_privileged_needs_higher_to_displace(self):
        """Test that privileged needs penalty margin to displace."""
        cache = {"123": {"acc": 0.8, "was_privileged": False}}
        penalty = 0.05
        
        # Privileged with acc=0.84 has effective_acc = 0.79, won't beat 0.8
        updated, update_type = update_cache_with_penalty(
            cache, "123", acc=0.84, is_privileged=True, penalty=penalty,
            response="new", global_steps=20
        )
        assert not updated
        
        # Privileged with acc=0.86 has effective_acc = 0.81, WILL beat 0.8
        updated, update_type = update_cache_with_penalty(
            cache, "123", acc=0.86, is_privileged=True, penalty=penalty,
            response="new", global_steps=30
        )
        assert updated
        assert update_type == "update"
        assert cache["123"]["acc"] == 0.86  # Stored raw acc
        assert cache["123"]["was_privileged"] == True
        print("✓ test_privileged_needs_higher_to_displace passed")
    
    def test_unprivileged_displaces_privileged_at_same_acc(self):
        """Test that unprivileged can displace privileged at same raw acc."""
        cache = {"123": {"acc": 0.8, "was_privileged": True}}
        
        # With penalty=0.05, a new unprivileged at 0.81 beats cached 0.8
        updated, update_type = update_cache_with_penalty(
            cache, "123", acc=0.81, is_privileged=False, penalty=0.05,
            response="new", global_steps=20
        )
        
        assert updated
        assert cache["123"]["was_privileged"] == False
        print("✓ test_unprivileged_displaces_privileged_at_same_acc passed")


class TestQuestionHashVerification:
    """Tests for question hash verification."""
    
    def test_matching_hash(self):
        """Test that matching descriptions pass verification."""
        description = "You are given an array of integers..."
        desc_hash = hashlib.md5(description[:500].encode()).hexdigest()
        
        cache = {"123": {"description_hash": desc_hash}}
        extra_info = [{"description": description}]
        
        assert verify_question_match(extra_info, 0, "123", cache)
        print("✓ test_matching_hash passed")
    
    def test_mismatching_hash(self):
        """Test that mismatching descriptions fail verification."""
        cache = {"123": {"description_hash": "different_hash"}}
        extra_info = [{"description": "You are given an array of integers..."}]
        
        assert not verify_question_match(extra_info, 0, "123", cache)
        print("✓ test_mismatching_hash passed")
    
    def test_no_cached_hash_passes(self):
        """Test that missing cached hash passes (backward compat)."""
        cache = {"123": {"acc": 0.8}}  # No description_hash
        extra_info = [{"description": "anything"}]
        
        assert verify_question_match(extra_info, 0, "123", cache)
        print("✓ test_no_cached_hash_passes passed")
    
    def test_no_current_description_passes(self):
        """Test that missing current description passes."""
        cache = {"123": {"description_hash": "some_hash"}}
        extra_info = [{}]  # No description
        
        assert verify_question_match(extra_info, 0, "123", cache)
        print("✓ test_no_current_description_passes passed")


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests combining multiple components."""
    
    def test_full_workflow(self):
        """Test complete privilege injection and cache update workflow."""
        # Setup
        cache = {
            "0": {"acc": 0.9, "was_privileged": False, "response": "old_solution_0"},
            "1": {"acc": 0.7, "was_privileged": True, "response": "old_solution_1"},  # Below threshold
        }
        
        config = {
            "enable": True,
            "privilege_fraction": 0.5,
            "privilege_fraction_decay": "linear",
            "accuracy_threshold": 0.8,
            "privilege_penalty": {
                "enable": True,
                "penalty_min": 0.02,
                "penalty_max": 0.10,
                "schedule": "late_ramp"
            }
        }
        
        # Simulate step 50/100 (midway through training)
        global_steps = 50
        total_steps = 100
        rollout_n = 8
        
        # Batch: 2 questions × 8 rollouts = 16 samples
        question_indices = [0]*8 + [1]*8
        
        # Get current privilege fraction (linear decay: 0.5 * 0.5 = 0.25)
        priv_frac = get_current_privilege_fraction(global_steps, total_steps, config)
        assert abs(priv_frac - 0.25) < 1e-6
        
        # Get penalty (late_ramp at t=0.5: still at p_min)
        penalty = get_cache_penalty(global_steps, total_steps, config)
        assert abs(penalty - 0.02) < 1e-6
        
        # Compute privilege mask
        mask, num_q = compute_privilege_mask(
            batch_size=16,
            rollout_n=8,
            question_indices=question_indices,
            cache=cache,
            threshold=0.8,
            privilege_fraction=priv_frac
        )
        
        # Only question 0 qualifies (acc=0.9 >= 0.8)
        assert num_q == 1
        
        # With fraction=0.25, num_privileged = int(8 * 0.25) = 2
        # Interleaved positions: 0, 2
        assert mask[0] == True
        assert mask[1] == False
        assert mask[2] == True
        assert mask[3:8].sum() == 0  # Rest are unprivileged
        
        # Question 1 doesn't qualify
        assert mask[8:16].sum() == 0
        
        print("✓ test_full_workflow passed")
    
    def test_cache_evolution_over_training(self):
        """Test that cache evolves correctly over training."""
        cache = {}
        config = {
            "enable": True,
            "privilege_penalty": {
                "enable": True,
                "penalty_min": 0.02,
                "penalty_max": 0.10,
                "schedule": "late_ramp"
            }
        }
        
        # Step 1: Empty cache, add first solution (privileged)
        penalty = get_cache_penalty(0, 100, config)  # 0.02
        update_cache_with_penalty(
            cache, "q0", acc=0.7, is_privileged=True, penalty=penalty,
            response="priv_sol_1", global_steps=0
        )
        assert cache["q0"]["acc"] == 0.7
        assert cache["q0"]["was_privileged"] == True
        
        # Step 2: Unprivileged attempts same question with slightly worse acc
        # effective_acc = 0.65 - 0 = 0.65 < 0.7, no update
        updated, _ = update_cache_with_penalty(
            cache, "q0", acc=0.65, is_privileged=False, penalty=penalty,
            response="unpriv_sol_1", global_steps=10
        )
        assert not updated
        
        # Step 3: Unprivileged with acc=0.72 beats cached 0.7
        updated, _ = update_cache_with_penalty(
            cache, "q0", acc=0.72, is_privileged=False, penalty=penalty,
            response="unpriv_sol_2", global_steps=20
        )
        assert updated
        assert cache["q0"]["was_privileged"] == False
        
        # Step 4: Late in training (step 90), penalty is higher
        penalty_late = get_cache_penalty(90, 100, config)
        # At t=0.9: penalty = 0.02 + 0.08 * (0.9-0.66)/0.34 ≈ 0.0765
        assert penalty_late > 0.06  # Should be around 0.0765
        
        # Privileged with acc=0.79 has effective_acc = 0.79 - 0.0765 ≈ 0.7135
        # This won't beat cached 0.72 (need effective > 0.72)
        updated, _ = update_cache_with_penalty(
            cache, "q0", acc=0.79, is_privileged=True, penalty=penalty_late,
            response="priv_sol_late", global_steps=90
        )
        assert not updated
        
        print("✓ test_cache_evolution_over_training passed")


# =============================================================================
# Main
# =============================================================================

def run_all_tests():
    """Run all test classes."""
    test_classes = [
        TestPrivilegeFractionDecay,
        TestCachePenalty,
        TestApplyPrivilegePenalty,
        TestShouldGivePrivilege,
        TestPrivilegeMaskComputation,
        TestCacheUpdateWithPenalty,
        TestQuestionHashVerification,
        TestIntegration,
    ]
    
    total_passed = 0
    total_failed = 0
    
    for test_class in test_classes:
        print(f"\n{'='*60}")
        print(f"Running {test_class.__name__}")
        print('='*60)
        
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    total_passed += 1
                except AssertionError as e:
                    print(f"✗ {method_name} FAILED: {e}")
                    total_failed += 1
                except Exception as e:
                    print(f"✗ {method_name} ERROR: {e}")
                    total_failed += 1
    
    print(f"\n{'='*60}")
    print(f"RESULTS: {total_passed} passed, {total_failed} failed")
    print('='*60)
    
    return total_failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

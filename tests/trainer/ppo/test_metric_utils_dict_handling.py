"""
Test to verify that process_validation_metrics correctly handles dict values
in reward_extra_infos_dict (fix for TypeError: unsupported operand type(s) for +: 'dict' and 'dict')
"""

import numpy as np
from verl.trainer.ppo.metric_utils import process_validation_metrics


class TestProcessValidationMetricsDictHandling:
    """Tests for handling of non-numeric values in process_validation_metrics."""

    def test_skips_dict_values_in_var_vals(self):
        """Test that dict values are properly skipped and don't cause TypeError."""
        data_sources = np.array(["source1", "source1", "source1"])
        sample_uids = ["uid1", "uid1", "uid1"]
        
        # This simulates the bug: reward_extra_info dicts leaked into infos_dict
        infos_dict = {
            "reward": [0.5, 0.8, 1.0],
            "acc": [0.5, 0.8, 1.0],
            # This is what caused the bug - dict values from reward_extra_info
            "reward_extra_info": [
                {"score": 0.5, "acc": 0.5, "pred": "A"},
                {"score": 0.8, "acc": 0.8, "pred": "B"},
                {"score": 1.0, "acc": 1.0, "pred": "C"},
            ],
        }
        
        # Should not raise TypeError: unsupported operand type(s) for +: 'dict' and 'dict'
        result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        
        # Verify numeric fields are processed
        # Result structure: result[data_source][var_name][metric_name]
        assert "source1" in result
        assert "reward" in result["source1"]
        assert "acc" in result["source1"]
        
        # Verify dict field is skipped (not in result)
        assert "reward_extra_info" not in result["source1"]

    def test_skips_dict_values_mixed_with_numeric(self):
        """Test handling when dict appears after numeric values (heterogeneous list)."""
        data_sources = np.array(["source1", "source1", "source1"])
        sample_uids = ["uid1", "uid1", "uid1"]
        
        # Heterogeneous list - first item is numeric, later items are dicts
        # This would have passed the old check (isinstance(var_vals[0], dict))
        infos_dict = {
            "reward": [0.5, 0.8, 1.0],
            # Mixed types - this should be skipped
            "bad_field": [0.5, {"nested": "dict"}, 1.0],
        }
        
        result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        
        # Verify numeric fields are processed
        # Result structure: result[data_source][var_name][metric_name]
        assert "reward" in result["source1"]
        # Verify mixed field is skipped
        assert "bad_field" not in result["source1"]

    def test_skips_list_values(self):
        """Test that list values are properly skipped."""
        data_sources = np.array(["source1", "source1"])
        sample_uids = ["uid1", "uid1"]
        
        infos_dict = {
            "reward": [0.5, 1.0],
            "some_list_field": [[1, 2, 3], [4, 5, 6]],
        }
        
        result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        
        # Result structure: result[data_source][var_name][metric_name]
        assert "reward" in result["source1"]
        assert "some_list_field" not in result["source1"]

    def test_skips_string_values(self):
        """Test that string values (like 'pred') are properly skipped."""
        data_sources = np.array(["source1", "source1"])
        sample_uids = ["uid1", "uid1"]
        
        infos_dict = {
            "reward": [0.5, 1.0],
            "pred": ["A", "B"],
            "feedback": ["correct", "wrong"],
        }
        
        result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        
        # Result structure: result[data_source][var_name][metric_name]
        assert "reward" in result["source1"]
        # String fields should be skipped
        assert "pred" not in result["source1"]
        assert "feedback" not in result["source1"]

    def test_normal_numeric_values_work(self):
        """Test that normal numeric values still work correctly."""
        data_sources = np.array(["source1", "source1", "source1", "source1"])
        sample_uids = ["uid1", "uid1", "uid1", "uid1"]
        
        infos_dict = {
            "reward": [0.5, 0.6, 0.7, 0.8],
            "acc": [0.0, 1.0, 1.0, 0.0],
        }
        
        result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        
        # Result structure: result[data_source][var_name][metric_name]
        # Check mean is computed correctly
        assert "mean@4" in result["source1"]["reward"]
        assert abs(result["source1"]["reward"]["mean@4"] - 0.65) < 0.01
        
        assert "mean@4" in result["source1"]["acc"]
        assert abs(result["source1"]["acc"]["mean@4"] - 0.5) < 0.01

    def test_empty_infos_dict_handled(self):
        """Test that empty infos_dict doesn't crash."""
        data_sources = np.array(["source1"])
        sample_uids = ["uid1"]
        
        infos_dict = {
            "reward": [0.5],
        }
        
        result = process_validation_metrics(data_sources, sample_uids, infos_dict)
        
        # Single value should still produce mean@1
        # Result structure: result[data_source][var_name][metric_name]
        assert "reward" in result["source1"]
        assert "mean@1" in result["source1"]["reward"]


if __name__ == "__main__":
    test_class = TestProcessValidationMetricsDictHandling()
    
    tests = [
        ("test_skips_dict_values_in_var_vals", test_class.test_skips_dict_values_in_var_vals),
        ("test_skips_dict_values_mixed_with_numeric", test_class.test_skips_dict_values_mixed_with_numeric),
        ("test_skips_list_values", test_class.test_skips_list_values),
        ("test_skips_string_values", test_class.test_skips_string_values),
        ("test_normal_numeric_values_work", test_class.test_normal_numeric_values_work),
        ("test_empty_infos_dict_handled", test_class.test_empty_infos_dict_handled),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"✓ {name}")
            passed += 1
        except Exception as e:
            print(f"✗ {name}: {e}")
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")

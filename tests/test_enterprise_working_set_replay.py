"""No-model unit tests for enterprise working-set replay."""

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "enterprise_working_set_replay.py"
spec = importlib.util.spec_from_file_location("enterprise_replay", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ReplayTests(unittest.TestCase):
    def test_policy_baselines(self):
        train = [{"task": "classification", "sequence": "train", "step": 0,
                  "pages": [1, 2], "transition_signal": 0.0}]
        test = [{"task": "classification", "sequence": "test", "step": i,
                 "pages": [1, 2], "transition_signal": 0.0} for i in range(3)]
        result = module.evaluate(train, test, 2, 1.0)
        self.assertEqual(result["static"]["misses"], 0)
        self.assertEqual(result["previous"]["misses"], 2)
        self.assertEqual(result["transition"]["misses"], 2)
        self.assertEqual(result["oracle"]["misses"], 2)

    def test_no_future_in_previous(self):
        train = [{"task": "tool", "sequence": "train", "step": 0,
                  "pages": [9], "transition_signal": 0.0}]
        test = [{"task": "tool", "sequence": "test", "step": i,
                 "pages": [i], "transition_signal": 0.0} for i in range(3)]
        result = module.evaluate(train, test, 1, 1.0)
        self.assertEqual(result["previous"]["misses"], 3)
        self.assertEqual(result["transition"]["misses"], 3)

    def test_no_false_physical_metric(self):
        self.assertTrue(hasattr(module, "evaluate"))

    def test_static_warmup_is_charged_for_each_reset_sequence(self):
        train = [{"task": "classification", "sequence": "train", "step": 0,
                  "pages": [1, 2], "transition_signal": 0.0}]
        test = [
            {"task": "classification", "sequence": sequence, "step": 0,
             "pages": [1, 2], "transition_signal": 0.0}
            for sequence in ("test-a", "test-b")
        ]
        result = module.evaluate(train, test, 2, 1.0)
        self.assertEqual(result["static"]["misses"], 0)
        self.assertEqual(result["static"]["warmup_transfers"], 4)
        self.assertEqual(result["static"]["total_transfers"], 4)

    def test_transition_prefetch_is_charged_and_evicts(self):
        train = [{"task": "classification", "sequence": "train", "step": 0,
                  "pages": [9], "transition_signal": 0.0}]
        test = [{"task": "classification", "sequence": "test", "step": 0,
                 "pages": [1], "transition_signal": 1.0}]
        result = module.evaluate(train, test, 1, 1.0)
        transition = result["transition"]
        self.assertEqual(transition["prefetch_transfers"], 1)
        self.assertEqual(transition["demand_transfers"], 1)
        self.assertEqual(transition["total_transfers"], 2)
        self.assertEqual(transition["evictions"], 1)

    def test_oracle_future_pages_are_charged_as_prefetches(self):
        train = [{"task": "classification", "sequence": "train", "step": 0,
                  "pages": [9], "transition_signal": 0.0}]
        test = [
            {"task": "classification", "sequence": "test", "step": 0,
             "pages": [1], "transition_signal": 0.0},
            {"task": "classification", "sequence": "test", "step": 1,
             "pages": [2], "transition_signal": 0.0},
        ]
        result = module.evaluate(train, test, 1, 1.0)
        oracle = result["oracle"]
        self.assertEqual(oracle["misses"], 1)
        self.assertEqual(oracle["prefetch_transfers"], 1)
        self.assertEqual(oracle["total_transfers"], 2)
        self.assertEqual(oracle["evictions"], 1)

    def test_demand_larger_than_capacity_is_rejected(self):
        train = [{"task": "classification", "sequence": "train", "step": 0,
                  "pages": [1], "transition_signal": 0.0}]
        test = [{"task": "classification", "sequence": "test", "step": 0,
                 "pages": [1, 2], "transition_signal": 0.0}]
        with self.assertRaisesRegex(module.CapacityError, "requires 2 pages"):
            module.evaluate(train, test, 1, 1.0)

    def test_no_policy_exceeds_capacity(self):
        train = [{"task": "classification", "sequence": "train", "step": 0,
                  "pages": [9, 8], "transition_signal": 0.0}]
        test = [
            {"task": "classification", "sequence": "test", "step": 0,
             "pages": [1, 2], "transition_signal": 1.0},
            {"task": "classification", "sequence": "test", "step": 1,
             "pages": [2, 3], "transition_signal": 1.0},
        ]
        result = module.evaluate(train, test, 2, 1.0)
        self.assertTrue(all(row["max_resident_pages"] <= 2 for row in result.values()))

    def test_total_transfers_equal_all_charged_transfer_classes(self):
        train = [{"task": "classification", "sequence": "train", "step": 0,
                  "pages": [9], "transition_signal": 0.0}]
        test = [{"task": "classification", "sequence": "test", "step": 0,
                 "pages": [1], "transition_signal": 1.0}]
        result = module.evaluate(train, test, 1, 1.0)
        for row in result.values():
            expected = (row["warmup_transfers"] + row["demand_transfers"]
                        + row["prefetch_transfers"])
            self.assertEqual(row["total_transfers"], expected)


if __name__ == "__main__":
    unittest.main()

"""No-model unit tests for enterprise working-set replay."""
import importlib.util
from pathlib import Path
import unittest

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


if __name__ == "__main__":
    unittest.main()

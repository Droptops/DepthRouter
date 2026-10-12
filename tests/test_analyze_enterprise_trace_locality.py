"""Tests for held-out enterprise trace locality analysis."""

import importlib.util
import unittest
from pathlib import Path

SCRIPT = (Path(__file__).resolve().parents[1] / "experiments"
          / "analyze_enterprise_trace_locality.py")
spec = importlib.util.spec_from_file_location("trace_locality", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class TraceLocalityTests(unittest.TestCase):
    def test_adjacent_comparison_uses_prompt_disjoint_control(self):
        rows = []
        for sequence, pages in (("a", [1, 2]), ("b", [3, 4])):
            for step in range(3):
                rows.append({
                    "task": "classification", "sequence": sequence, "step": step,
                    "pages": pages, "transition_signal": float(step),
                })
        result = module.analyze(rows)
        self.assertEqual(result["adjacent_pairs"], 4)
        self.assertEqual(result["same_task_same_step_control_pairs"], 3)
        self.assertEqual(result["adjacent"]["mean_jaccard"], 1.0)
        self.assertEqual(result["between_sequence_control"]["mean_jaccard"], 0.0)
        self.assertEqual(result["adjacent_jaccard_lift"], 1.0)

    def test_lagged_signal_correlation_is_reported_when_identifiable(self):
        rows = [
            {"task": "tool", "sequence": "a", "step": 0,
             "pages": [1], "transition_signal": 0.0},
            {"task": "tool", "sequence": "a", "step": 1,
             "pages": [1], "transition_signal": 0.0},
            {"task": "tool", "sequence": "a", "step": 2,
             "pages": [2], "transition_signal": 1.0},
        ]
        result = module.analyze(rows)
        self.assertEqual(
            result["lagged_signal_vs_current_new_page_fraction_pearson"], 1.0
        )

    def test_invalid_trace_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            module.analyze([{"pages": [1]}])


if __name__ == "__main__":
    unittest.main()

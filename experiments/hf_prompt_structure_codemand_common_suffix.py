"""Common-suffix falsification control for PR #11.

Run the existing co-demand experiment with every structure sharing the same
final instruction. Compare against the unmodified experiment with identical
model, semantic split, seed, and budgets. This is an oracle-demand diagnostic,
not a physical memory traffic benchmark.

Example:
 python experiments/hf_prompt_structure_codemand_common_suffix.py --json-out suffix.json
 python experiments/hf_prompt_structure_codemand.py --json-out original.json
"""
from __future__ import annotations

import runpy
from pathlib import Path

SOURCE = Path(__file__).with_name("hf_prompt_structure_codemand.py")
namespace = runpy.run_path(str(SOURCE), run_name="depthrouter_codemand_control")
suffix = (
    "\n\nFinal instruction for all tasks: Select the better option, "
    "then give exactly one concise reason."
)
namespace["STRUCTURES"].update(
    {name: template + suffix for name, template in namespace["STRUCTURES"].items()}
)
namespace["main"]()

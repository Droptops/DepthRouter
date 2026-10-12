"""Deterministic synthetic enterprise prompts for reproducible trace collection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CATEGORIES = ("billing", "technical_support", "account_access", "cancellation", "sales")
TOOLS = ("search_kb", "lookup_account", "create_ticket", "send_email", "schedule_meeting")


def examples(n=100):
    if n < 2:
        raise ValueError("need at least two examples per task")
    rows = []
    for i in range(n):
        category = CATEGORIES[i % len(CATEGORIES)]
        tool = TOOLS[i % len(TOOLS)]
        rows.append({"task": "classification", "sequence": f"class-{i:03d}",
                     "text": f"Classify the support ticket into billing, technical_support, account_access, cancellation, sales. Ticket: Case {i}: customer requests assistance with {category.replace('_', ' ')}. Respond with one category.",
                     "expected": category})
        rows.append({"task": "extraction", "sequence": f"extract-{i:03d}",
                     "text": f"Extract account_id and invoice_total from: Account AC-{1000+i} has invoice USD {10+i}.00. Respond as JSON.",
                     "expected": {"account_id": f"AC-{1000+i}", "invoice_total": f"{10+i}.00"}})
        rows.append({"task": "tool_selection", "sequence": f"tool-{i:03d}",
                     "text": f"Choose exactly one tool from {', '.join(TOOLS)}. The user explicitly requests {tool.replace('_', ' ')}. Respond with the tool name.",
                     "expected": tool})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--examples-per-task", type=int, default=100)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    rows = examples(args.examples_per_task)
    Path(args.out).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
    print(json.dumps({"total": len(rows), "per_task": args.examples_per_task}))


if __name__ == "__main__":
    main()

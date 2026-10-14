"""Runnable offline example: uv run python examples/support.py --store support.db."""

import argparse
import json
from dataclasses import asdict

from nudge import Nudge
from nudge.benchmark import SyntheticAnswerer, SyntheticProposer, synthetic_choice


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--decisions", type=int, default=400)
    args = parser.parse_args()
    with Nudge(
        synthetic_choice(),
        store=args.store,
        model=SyntheticAnswerer("context"),
        proposer=SyntheticProposer(),
        seed=8,
    ) as policy:
        for i in range(args.decisions):
            repeat = bool(i % 2)
            d = policy.ask(state=json.dumps({"ticket": i, "repeat": repeat, "urgent": False}))
            policy.feedback(d.id, resolved=float((d.action == "escalate") == repeat))
        print(json.dumps(asdict(policy.discover_questions()), indent=2))
        print(json.dumps(asdict(policy.learn()), indent=2))
        print(json.dumps(asdict(policy.ask(state='{"repeat": true, "urgent": false}')), indent=2))
        print(json.dumps(asdict(policy.report()), indent=2))


if __name__ == "__main__":
    main()

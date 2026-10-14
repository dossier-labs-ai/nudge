"""Reverse-engineer useful routing features from recorded RouterBench outcomes.

This is an offline benchmark-max diagnostic. It intentionally reports dataset
metadata features separately from prompt-observable features; family labels are
not available to a production router unless inferred from the prompt.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from .routerbench import write_json
from .routerbench_data import MODELS, aggregate, evaluate


def feature_values(row):
    state = row["state"]
    lower = state.lower()
    return {
        "family_metadata": row["family"],
        "prompt_length_bucket": "short" if len(state) < 900 else "medium" if len(state) < 2500 else "long",
        "multiple_choice_format": "mcq" if re.search(r"\b[A-D]\s*[\)\.]", state) else "open",
        "code_markers": "code" if re.search(r"\b(code|function|python|javascript|program|implement)\b", lower) else "noncode",
        "math_markers": "math" if re.search(r"\b(math|equation|calculate|number|percent|probability|solve)\b", lower) else "nonmath",
        "passage_markers": "passage" if re.search(r"\b(passage|article|story|paragraph|following text)\b", lower) else "nopassage",
    }


def reward(row, index, penalty):
    return row["quality"][index] - penalty * row["cost"][index]


def route_by_feature(train, test, feature, penalty):
    grouped = defaultdict(list)
    for row in train:
        grouped[feature_values(row)[feature]].append(row)
    global_means = {
        model: sum(reward(row, i, penalty) for row in train) / len(train)
        for i, model in enumerate(MODELS)
    }
    policies = {}
    for group, rows in grouped.items():
        means = {
            model: sum(reward(row, i, penalty) for row in rows) / len(rows)
            for i, model in enumerate(MODELS)
        }
        policies[group] = max(means, key=means.get)
    records = []
    for row in test:
        group = feature_values(row)[feature]
        action = policies.get(group, max(global_means, key=global_means.get))
        policy = {model: float(model == action) for model in MODELS}
        baseline = {model: 1.0 / len(MODELS) for model in MODELS}
        records.append({
            "family": row["family"],
            "rune": evaluate(row, baseline, penalty),
            "nudge": evaluate(row, policy, penalty),
        })
    return aggregate(records), policies


def compare(prepared):
    penalty = 10.0
    candidates = (
        "family_metadata",
        "prompt_length_bucket",
        "multiple_choice_format",
        "code_markers",
        "math_markers",
        "passage_markers",
    )
    results = {}
    for feature in candidates:
        metrics, policies = route_by_feature(prepared["train"], prepared["test"], feature, penalty)
        results[feature] = {"test": metrics, "training_selected_actions": policies}
    return {
        "training_prompts": len(prepared["train"]),
        "test_prompts": len(prepared["test"]),
        "features": results,
        "interpretation": [
            "Family metadata is a deliberate oracle diagnostic; it is not prompt-observable.",
            "Prompt-shape candidates are available at routing time but are intentionally crude.",
            "Each feature's action mapping is selected from training outcomes only.",
            "This diagnostic uses full recorded training outcomes, unlike Nudge's chosen-action feedback.",
            "The test split was previously inspected and is exploratory, not a fresh confirmation.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.prepared.read_text()))
    write_json(args.out, result)
    print(json.dumps({k: v["test"]["nudge"] for k, v in result["features"].items()}, indent=2))


if __name__ == "__main__":
    main()

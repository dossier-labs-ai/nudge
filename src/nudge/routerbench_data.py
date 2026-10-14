"""Pinned RouterBench ingestion and evaluator-only counterfactual metrics."""

from __future__ import annotations

import ast
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .types import canonical

DATA_REVISION = "784021482c3f320c6619ed4b3bb3b41a21424fcb"
DATA_SHA256 = "ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d"
MODELS = (
    "WizardLM/WizardLM-13B-V1.2",
    "claude-instant-v1",
    "claude-v1",
    "claude-v2",
    "gpt-3.5-turbo-1106",
    "gpt-4-1106-preview",
    "meta/code-llama-instruct-34b-chat",
    "meta/llama-2-70b-chat",
    "mistralai/mistral-7b-chat",
    "mistralai/mixtral-8x7b-chat",
    "zero-one-ai/Yi-34B-Chat",
)
FAMILIES = ("arc-challenge", "grade-school-math", "hellaswag", "mbpp", "mmlu", "winogrande")


def fingerprint(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def prompt_text(raw):
    """Decode the dataset's Python-list strings without executing input."""
    parts = ast.literal_eval(raw) if isinstance(raw, str) else raw
    if not isinstance(parts, list) or not parts or not all(isinstance(x, str) for x in parts):
        raise ValueError("RouterBench prompt must encode a list of strings")
    return "\n\n".join(parts)


def proportional_counts(sizes, count):
    """Largest-remainder allocation, with stable family-order tie breaking."""
    total = sum(sizes)
    if not 0 <= count <= total or total == 0:
        raise ValueError("sample count exceeds eligible population or population is empty")
    counts = [count * size // total for size in sizes]
    order = sorted(range(len(sizes)), key=lambda i: -(count * sizes[i] % total))
    for i in order[:count - sum(counts)]:
        counts[i] += 1
    return counts


def prepare(records, *, count=1000, holdout=200, seed=20261013):
    if count < 12 or not (holdout == 0 or 6 <= holdout <= count - 6):
        raise ValueError("need at least six training prompts and zero or at least six held-out prompts")
    pools, seen, exclusions = defaultdict(list), set(), Counter()
    for row in records:
        family = "mmlu" if row["eval_name"].startswith("mmlu-") else row["eval_name"]
        if family not in FAMILIES:
            exclusions["outside_six_task_families"] += 1
            continue
        state = prompt_text(row["prompt"])
        key = fingerprint(state)
        if key in seen:
            exclusions["duplicate_prompt"] += 1
            continue
        quality = np.array([float(row[model]) for model in MODELS])
        cost = np.array([float(row[model + "|total_cost"]) for model in MODELS])
        if (
            not np.isfinite(quality).all()
            or not np.isfinite(cost).all()
            or np.any((quality < 0) | (quality > 1))
            or np.any(cost < 0)
        ):
            exclusions["incomplete_or_invalid_outcomes"] += 1
            continue
        seen.add(key)
        pools[family].append(
            {
                "id": str(row["sample_id"]),
                "prompt_sha256": key,
                "family": family,
                "state": state,
                "quality": quality.tolist(),
                "cost": cost.tolist(),
            }
        )
    rng = np.random.default_rng(seed)
    train, test = [], []
    sizes = proportional_counts([len(pools[f]) for f in FAMILIES], count)
    test_sizes = proportional_counts(sizes, holdout)
    for i, family in enumerate(FAMILIES):
        size, test_size = sizes[i], test_sizes[i]
        pool = sorted(pools[family], key=lambda r: r["prompt_sha256"])
        picked = [pool[j] for j in rng.permutation(len(pool))[:size]]
        test.extend(picked[:test_size])
        train.extend(picked[test_size:])
    rng.shuffle(train)
    rng.shuffle(test)
    return {
        "format_version": 1,
        "source_revision": DATA_REVISION,
        "source_sha256": DATA_SHA256,
        "seed": seed,
        "sampling": "proportional_without_replacement",
        "evaluation": "online" if holdout == 0 else "heldout",
        "models": list(MODELS),
        "train": train,
        "test": test,
        "excluded": dict(exclusions),
        "eligible": {f: len(pools[f]) for f in FAMILIES},
    }


def load_pinned(path: Path, **kwargs):
    # The upstream release is pickle. Never unpickle an arbitrary CLI file:
    # require the exact independently published LFS digest before deserializing.
    if hashlib.sha256(path.read_bytes()).hexdigest() != DATA_SHA256:
        raise ValueError("dataset SHA256 does not match pinned public RouterBench release")
    import pandas as pd

    frame = pd.read_pickle(path)
    return prepare(frame.to_dict("records"), **kwargs)


def evaluate(row, probabilities, penalty):
    p = np.array([probabilities[m] for m in MODELS])
    if not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(sum(p), 1):
        raise ValueError("invalid evaluator policy")
    quality, cost = np.array(row["quality"]), np.array(row["cost"])
    utility = quality - penalty * cost
    return {
        "quality": float(p @ quality),
        "cost": float(p @ cost),
        "utility": float(p @ utility),
        "hindsight_regret": float(max(utility) - p @ utility),
    }


def aggregate(records, seed=20261013):
    if not records:
        return {}
    metrics = ("quality", "cost", "utility", "hindsight_regret")
    result = {
        policy: {metric: float(np.mean([r[policy][metric] for r in records])) for metric in metrics}
        for policy in ("rune", "nudge")
    }
    result["count"] = len(records)
    for policy in ("rune", "nudge"):
        result[policy]["cumulative_hindsight_regret"] = sum(
            r[policy]["hindsight_regret"] for r in records
        )
    # Stratified paired bootstrap: resample prompts within each task family.
    # This describes prompt uncertainty, not between-training-seed uncertainty.
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, row in enumerate(records):
        groups[row["family"]].append(i)
    indices = np.concatenate([rng.choice(v, (2000, len(v))) for v in groups.values()], axis=1)
    result["paired_nudge_minus_rune"] = {}
    for metric in metrics:
        delta = np.array([r["nudge"][metric] - r["rune"][metric] for r in records])
        result["paired_nudge_minus_rune"][metric] = {
            "mean": float(delta.mean()),
            "prompt_bootstrap_95": np.quantile(
                delta[indices].mean(axis=1), [0.025, 0.975]
            ).tolist(),
        }
    return result

"""Archived bounded-policy comparators only; never used by the live learner."""

import math
from types import SimpleNamespace
import numpy as np


def bounded_expert(config=None):
    return SimpleNamespace(**({'max_drift': 4.0, 'explore_floor': 0.05} | (config or {})))


def tilted(baseline, scores, scale, expert):
    if expert.max_drift == 1 or expert.explore_floor == 1:
        return dict(baseline)
    logs = {a: np.clip(scores[a] / scale, -1, 1) * math.log(expert.max_drift) / 2
            for a in baseline}
    offset = max(logs.values())
    weights = {a: p * math.exp(logs[a] - offset) for a, p in baseline.items()}
    total = sum(weights.values())
    return {a: (1-expert.explore_floor) * weights[a] / total + expert.explore_floor * p
            for a, p in baseline.items()}

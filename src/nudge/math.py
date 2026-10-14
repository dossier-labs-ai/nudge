"""Gaussian linear regression and unrestricted posterior action selection."""

from __future__ import annotations

import math
import hashlib
from typing import Sequence

import numpy as np

from .types import Expert, Question


def vector(probability: float, features: dict, questions: Sequence[Question]) -> np.ndarray:
    logit = (
        -4.0
        if probability <= 0
        else 4.0
        if probability >= 1
        else max(-4.0, min(4.0, math.log(probability) - math.log1p(-probability)))
    )
    return np.array(
        [1.0, logit, *(value for q in questions for value in q.feature_vector(features[q.name]))]
    )


def reward_scale(rewards: Sequence[float]) -> float:
    if not rewards:
        return 1.0
    spread = max(rewards) - min(rewards)
    # Constant outcomes still carry information. Use their magnitude, or one
    # for all-zero data, rather than a singular prior/noise covariance.
    scale = spread if spread > 0 else max(abs(rewards[0]), 1.0)
    if not math.isfinite(scale) or not 1e-12 <= scale <= 1e12:
        raise ValueError("reward scale must be within [1e-12, 1e12]; rescale feedback weights")
    return scale


def fit(
    rows: Sequence[dict], actions: Sequence[str], questions: Sequence[Question], scale: float
) -> dict:
    dimension = 2 + sum(
        len(q.categories) if q.kind == "categorical" and q.categories else 1
        for q in questions
    )
    result = {}
    # Solve in reward/scale units. This avoids tiny/huge prior precisions.
    # tau=scale/8, sigma=scale/2 => precision=64 I + 4 X' W X.
    for action in actions:
        selected = [row for row in rows if row["action"] == action]
        precision = 64.0 * np.eye(dimension)
        natural = np.zeros(dimension)
        if selected:
            x = np.array(
                [vector(row["jev"][action], row["features"], questions) for row in selected]
            )
            y = np.array([row["reward"] / scale for row in selected])
            weights = np.array([row["weight"] for row in selected])
            precision += 4.0 * x.T @ (weights[:, None] * x)
            natural = 4.0 * x.T @ (weights * y)
        mean = np.linalg.solve(precision, natural) * scale
        result[action] = {
            "mean": mean.tolist(),
            "precision": precision.tolist(),
            "n": sum(row["weight"] for row in selected),
        }
    return result


def coefficients(
    critic: dict, scale: float, expert: Expert, rng: np.random.Generator | None
) -> np.ndarray:
    mean = np.array(critic["mean"])
    if rng is None:
        return mean
    # P=L L'; L'^-1 z has covariance P^-1 (not L^-1 z).
    chol = np.linalg.cholesky(np.array(critic["precision"]))
    return mean + scale * np.linalg.solve(chol.T, rng.normal(size=len(mean)))


def policy_probabilities(critics, rows, questions, scale, expert):
    """Finite posterior ensemble policy shared exactly by serving and evaluation."""
    scores = []
    for action, critic in critics.items():
        mean = np.array(critic['mean'])
        seed = int.from_bytes(hashlib.sha256(action.encode()).digest()[:8], 'big')
        rng = np.random.default_rng(seed)
        half = (expert.posterior_draws + 1) // 2
        noise = rng.normal(size=(len(mean), half))
        noise = np.concatenate((noise, -noise), axis=1)[:, :expert.posterior_draws]
        chol = np.linalg.cholesky(np.array(critic['precision']))
        weights = mean[:, None] + scale * np.linalg.solve(chol.T, noise)
        x = np.array([vector(r['jev'][action], r['features'], questions) for r in rows])
        scores.append(x @ weights)
    values = np.stack(scores, axis=2)
    winners = values == values.max(axis=2, keepdims=True)
    probabilities = (winners / winners.sum(axis=2, keepdims=True)).mean(axis=1)
    return (1 - expert.explore_floor) * probabilities + expert.explore_floor / len(critics)


def predict(critics: dict, row: dict, questions: Sequence[Question]) -> float:
    action = row["action"]
    return float(
        np.array(critics[action]["mean"]) @ vector(row["jev"][action], row["features"], questions)
    )


def mse(critics: dict, rows: Sequence[dict], questions: Sequence[Question]) -> float:
    return float(
        np.average(
            [(row["reward"] - predict(critics, row, questions)) ** 2 for row in rows],
            weights=[row["weight"] for row in rows],
        )
    )

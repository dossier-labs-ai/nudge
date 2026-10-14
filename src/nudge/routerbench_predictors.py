"""Offline predictor comparison with training-only parameter selection; no inference."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .math import vector
from .historical import tilted
from .routerbench import write_json
from .routerbench_data import MODELS, aggregate, evaluate
from .routerbench_diagnose import load_recorded_run


def design(rows, questions, actions):
    x = np.array([vector(r["jev"][r["action"]], r["features"], questions) for r in rows])
    a = np.array([actions.index(r["action"]) for r in rows])
    return x, a


def polynomial(x):
    """Keep intercept/linear terms, then all degree-two terms excluding intercept."""
    return np.column_stack([x, *(x[:, i] * x[:, j]
                                for i in range(1, x.shape[1])
                                for j in range(i, x.shape[1]))])


def predict(x, actions, y, query, query_actions, *, family, alpha, bandwidth=1.0):
    if family == "shared_rbf":
        # Scale using only this fit's training rows. Keep the intercept out of distances.
        spread = np.maximum(np.std(x[:, 1:], axis=0), 0.1)
        train = x[:, 1:] / spread
        target = query[:, 1:] / spread

        def kernel(left, la, right, ra):
            distance = np.maximum(
                np.sum(left**2, axis=1)[:, None]
                + np.sum(right**2, axis=1)[None, :] - 2 * left @ right.T, 0
            ) / max(left.shape[1], 1)
            # Shared context plus action-specific deviations; positive semidefinite.
            return np.exp(-distance / (2 * bandwidth**2)) * (
                0.25 + 0.75 * (la[:, None] == ra[None, :])
            )

        weights = np.linalg.solve(
            kernel(train, actions, train, actions) + alpha * np.eye(len(x)), y
        )
        return kernel(target, query_actions, train, actions) @ weights
    if family in ("quadratic", "shared_quadratic"):
        x, query = polynomial(x), polynomial(query)
    if family == "shared_quadratic":
        count = max(int(actions.max()), int(query_actions.max())) + 1

        def expand(values, labels):
            deviations = (np.eye(count)[labels, :, None] * values[:, None, :]).reshape(
                len(values), -1
            )
            return np.column_stack([values, deviations])

        x, query = expand(x, actions), expand(query, query_actions)
        weights = np.linalg.solve(x.T @ x + alpha * np.eye(x.shape[1]), x.T @ y)
        return query @ weights
    if family not in ("linear", "quadratic"):
        raise ValueError("unknown predictor family")
    result = np.zeros(len(query))
    for action in np.unique(query_actions):
        local = x[actions == action]
        weights = np.linalg.solve(
            local.T @ local + alpha * np.eye(x.shape[1]), local.T @ y[actions == action]
        )
        result[query_actions == action] = query[query_actions == action] @ weights
    return result


def select(train, questions, actions, configurations):
    """Selection only sees chosen feedback; test rows/outcomes are not accepted."""
    x, a = design(train, questions, actions)
    y = np.array([r["reward"] for r in train])
    folds = np.array([int(hashlib.sha256(r["state"].encode()).hexdigest()[:8], 16) % 3
                      for r in train])
    if len(np.unique(folds)) != 3:
        raise ValueError("three nonempty training folds required")
    candidates = []
    for config in configurations:
        predictions = np.empty(len(train))
        for fold in range(3):
            held = folds == fold
            predictions[held] = predict(x[~held], a[~held], y[~held], x[held], a[held], **config)
        candidates.append({"config": config, "cv_chosen_reward_mse": float(np.mean((y-predictions)**2))})
    return min(candidates, key=lambda c: c["cv_chosen_reward_mse"])["config"], candidates


def compare(data, database):
    train, _, test, questions, expert, scale, penalty, _ = load_recorded_run(data, database)
    if len(test) != len(data["test"]):
        raise ValueError("predictor comparison requires the complete test partition")
    grids = {
        "original_linear_mean": [{"family": "linear", "alpha": 16.0}],
        "tuned_linear": [{"family": "linear", "alpha": a} for a in (0.1, 1., 16., 100.)],
        "quadratic": [{"family": "quadratic", "alpha": a} for a in (0.1, 1., 16., 100.)],
        "shared_quadratic": [{"family": "shared_quadratic", "alpha": a} for a in (0.1, 1., 16., 100.)],
        "shared_rbf": [{"family": "shared_rbf", "alpha": a, "bandwidth": b}
                       for a in (0.1, 1., 16.) for b in (0.5, 2.)],
    }
    selected, cv = {}, {}
    for name, grid in grids.items():
        selected[name], cv[name] = select(train, questions, MODELS, grid)
    # Freeze selection before touching any test outcomes.
    winner = min(selected, key=lambda name: min(c["cv_chosen_reward_mse"] for c in cv[name]))
    x, a = design(train, questions, MODELS)
    y = np.array([r["reward"] for r in train])
    queries = [{**row, "action": model} for row, _ in test for model in MODELS]
    qx, qa = design(queries, questions, MODELS)
    results = {}
    for name, config in selected.items():
        scores = predict(x, a, y, qx, qa, **config).reshape(len(test), len(MODELS))
        records, movement = [], []
        for (row, recorded), values in zip(test, scores):
            policy = tilted(row["jev"], dict(zip(MODELS, values)), scale, expert)
            records.append({"family": recorded["family"],
                            "rune": evaluate(recorded, row["jev"], penalty),
                            "nudge": evaluate(recorded, policy, penalty)})
            movement.append(sum(abs(policy[m] - row["jev"][m]) for m in MODELS) / 2)
        results[name] = {"selected": config, "metrics": aggregate(records),
                         "mean_probability_mass_moved": float(np.mean(movement))}
    return {"training_prompts": len(train), "test_prompts": len(test),
            "questions": [q.name for q in questions], "training_selected_winner": winner,
            "cv": cv, "results": results,
            "limitations": [
                "Offline refits on fixed logged chosen-action feedback; no new exploration trajectory.",
                "Five final questions were discovered using all training rows; CV is conditional on that representation, not nested discovery.",
                "Selection minimizes chosen-action reward MSE, not routing utility; no counterfactual training rewards are used.",
                "Same tilt, reward scale and posterior-mean evaluation for all predictors; no embeddings or new inference.",
                "Previously inspected test split is exploratory; do not select a winner from test scores.",
            ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--database", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = compare(json.loads(args.prepared.read_text()), args.database)
    write_json(args.out, result)
    print(json.dumps({"training_selected_winner": result["training_selected_winner"],
                      "results": {k: v["metrics"]["nudge"] for k, v in result["results"].items()}}, indent=2))


if __name__ == "__main__":
    main()

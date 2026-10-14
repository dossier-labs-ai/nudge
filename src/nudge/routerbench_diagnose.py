"""Offline diagnostic ablations of a stopped RouterBench pilot; no model calls."""

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

from .benchmark import tilt_oracle
from .discovery import question_key
from .math import fit, vector
from .historical import tilted, bounded_expert
from .routerbench import write_json
from .routerbench_data import MODELS, aggregate, evaluate
from .types import Question


def load_recorded_run(data, database):
    database = Path(database).resolve()
    if Path(str(database) + "-wal").exists():
        raise ValueError("stop the run and close its SQLite connections before diagnostics")
    connection = sqlite3.connect(database.as_uri() + "?immutable=1", uri=True)
    try:
        snapshot = json.loads(
            connection.execute(
                "SELECT payload FROM snapshots ORDER BY version DESC LIMIT 1"
            ).fetchone()[0]
        )
        decisions = [
            {"id": identifier, "state": state, **json.loads(payload)}
            for identifier, state, payload in connection.execute(
                "SELECT id,state,payload FROM decisions ORDER BY rowid"
            )
        ]
        feedback = {
            identifier: json.loads(fields)
            for identifier, fields in connection.execute(
                "SELECT decision_id,fields FROM feedback ORDER BY sequence"
            )
        }
        features = {}
        for identifier, key, value in connection.execute(
            "SELECT decision_id,question_key,value FROM feature_answers"
        ):
            features[(identifier, key)] = json.loads(value) if isinstance(value, str) else value
    finally:
        connection.close()
    if snapshot["recency"]["half_life_seconds"] is not None:
        raise ValueError("diagnostic currently requires the unit-weight pilot protocol")
    questions = [Question(**q) for q in snapshot["questions"]]
    expert = bounded_expert(snapshot['expert'])
    scale = snapshot["scale"]
    penalty = -snapshot["choice"]["reward"]["cost"]
    train_data = {r["state"]: r for r in data["train"]}
    test_data = {r["state"]: r for r in data["test"]}
    if train_data.keys() & test_data.keys():
        raise ValueError("training and test prompts overlap")
    train, full, test = [], [], []
    for decision in decisions:
        row = {
            **decision,
            "features": {q.name: features[(decision["id"], question_key(q))] for q in questions},
            "weight": 1.0,
        }
        if decision["id"] in feedback:
            observed = feedback[decision["id"]]
            row["reward"] = observed["quality"] - penalty * observed["cost"]
            train.append(row)
            recorded = train_data[decision["state"]]
            full.extend(
                {
                    **row,
                    "action": model,
                    "reward": recorded["quality"][j] - penalty * recorded["cost"][j],
                }
                for j, model in enumerate(MODELS)
            )
        else:
            test.append((row, test_data[decision["state"]]))
    if len(train) != len(data["train"]) or len(train) != snapshot["feedback_count"]:
        raise ValueError("training feedback is incomplete or differs from the final snapshot")
    return train, full, test, questions, expert, scale, penalty, snapshot


def diagnose(data, database):
    train, full, test, questions, expert, scale, penalty, snapshot = load_recorded_run(
        data, database
    )
    chosen = fit(train, MODELS, questions, scale)
    for model in MODELS:
        if not np.allclose(chosen[model]["mean"], snapshot["critics"][model]["mean"]):
            raise ValueError("offline refit does not reproduce the saved critic")
    variants = {
        "chosen_feedback_mean": (chosen, questions),
        "chosen_feedback_no_questions_mean": (fit(train, MODELS, [], scale), []),
        "all_training_outcomes_mean": (fit(full, MODELS, questions, scale), questions),
    }
    comparisons = {}
    movement = []
    for row, recorded in test:
        baseline = row["jev"]
        rewards = {
            m: recorded["quality"][j] - penalty * recorded["cost"][j] for j, m in enumerate(MODELS)
        }
        policies = {"logged_nudge": row["policy"]}
        for name, (critics, qs) in variants.items():
            scores = {
                m: float(np.array(critics[m]["mean"]) @ vector(baseline[m], row["features"], qs))
                for m in MODELS
            }
            policies[name] = tilted(baseline, scores, scale, expert)
        policies["perfect_test_rewards_same_tilt"] = tilted(baseline, rewards, scale, expert)
        policies["hindsight_best_allowed_tilt"] = tilt_oracle(baseline, rewards, expert)
        policies["rune_argmax"] = {m: float(m == max(baseline, key=baseline.get)) for m in MODELS}
        for model in ("gpt-3.5-turbo-1106", "gpt-4-1106-preview"):
            policies["fixed_" + model] = {m: float(m == model) for m in MODELS}
        for name, policy in policies.items():
            comparisons.setdefault(name, []).append(
                {
                    "family": recorded["family"],
                    "rune": evaluate(recorded, baseline, penalty),
                    "nudge": evaluate(recorded, policy, penalty),
                }
            )
        movement.append(sum(abs(row["policy"][m] - baseline[m]) for m in MODELS) / 2)
    return {
        "training_prompts": len(train),
        "completed_test_prompts": len(test),
        "planned_test_prompts": len(data["test"]),
        "feedback_counts": {m: Counter(r["action"] for r in train)[m] for m in MODELS},
        "mean_probability_mass_moved": float(np.mean(movement)),
        "questions": snapshot["questions"],
        "scale": scale,
        "comparisons": {name: aggregate(rows) for name, rows in comparisons.items()},
        "notes": [
            "Post-hoc diagnostics on completed test prompts; not an independent confirmation experiment.",
            "Full-information refit uses training outcomes only; questions, baseline, scale and tilt are held fixed.",
            "Mean variants omit posterior sampling; compare full-information mean with chosen-feedback mean.",
            "Perfect-test-reward and hindsight-tilt variants leak test outcomes deliberately as diagnostic oracles.",
            "No new discovery, online feedback trajectory, or model inference is simulated.",
            "Paired intervals describe prompt uncertainty conditional on this single trained run.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(json.loads(args.prepared.read_text()), args.database)
    write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

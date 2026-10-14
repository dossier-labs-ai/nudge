"""Reproducible synthetic contextual-bandit experiments, no external inference.

The learner sees state and chosen-action feedback only. Full reward vectors are
owned by this evaluator. Scripted proposers test the discovery mechanism, not
an LLM's ability to invent useful questions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import tempfile
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .core import Nudge
from .historical import tilted, bounded_expert
from .recency import DEFAULT_HALF_LIFE_SECONDS, validate_half_life
from .types import Answer, Choice, Expert, Proposal, Question, canonical, finite


SCENARIOS = (
    "calibration",
    "context",
    "no_signal",
    "weak_support",
    "noisy_features",
    "interaction",
    "nonstationary",
)
METHODS = ("baseline", "uniform", "no_features", "fixed_features", "discovery")


def synthetic_choice() -> Choice:
    return Choice(
        "synthetic_support",
        "Choose how to handle this support ticket",
        {"answer": "Answer directly", "escalate": "Escalate to a person"},
        {"resolved": "Observed binary resolution outcome"},
    )


QUESTIONS = (
    Question("repeat", "Has this issue happened before?", paraphrase="Is this a recurring issue?"),
    Question(
        "urgent", "Does this require attention today?", paraphrase="Is same-day help requested?"
    ),
    Question(
        "interaction",
        "Exactly one of recurring issue and same-day help?",
        paraphrase="Is either recurrence or urgency present, but not both?",
    ),
    Question("flat", "Is this a support ticket?", paraphrase="Does the state describe a ticket?"),
)


class SyntheticAnswerer:
    def __init__(self, scenario: str):
        if scenario not in SCENARIOS:
            raise ValueError("unknown synthetic scenario")
        self.scenario = scenario
        self.identity = f"synthetic-v1:{scenario}"
        self.calls = 0

    def ask(self, state, questions):
        self.calls += 1
        context = json.loads(state)
        repeat, urgent = context["repeat"], context["urgent"]
        if self.scenario == "noisy_features":
            # Repeatable noisy measurement; same feature at fit and decision time.
            flip = int(hashlib.sha256(state.encode()).hexdigest()[:8], 16) / 2**32 < 0.25
            repeat = bool(repeat) ^ flip
        available = {
            "repeat": float(repeat),
            "urgent": float(urgent),
            "interaction": float(bool(repeat) ^ bool(urgent)),
            "flat": 1.0,
        }
        probability = 0.995 if self.scenario == "weak_support" else 0.75
        return Answer(
            {"answer": probability, "escalate": 1 - probability},
            {q.name: available[q.name] for q in questions},
        )


class SyntheticProposer:
    identity = "synthetic-scripted-proposer-v1"

    def propose(self, context):
        existing = {q.name for q in context.questions}
        return [Proposal("add", question=q) for q in QUESTIONS if q.name not in existing]


def context_at(rng, identifier: int) -> str:
    return canonical(
        {"ticket": identifier, "repeat": bool(rng.integers(2)), "urgent": bool(rng.integers(2))}
    )


def reward_means(scenario: str, state: str, step: int, horizon: int) -> dict[str, float]:
    context = json.loads(state)
    if scenario == "no_signal":
        return {"answer": 0.5, "escalate": 0.5}
    if scenario in ("calibration", "weak_support"):
        return {"answer": 0.2, "escalate": 0.8}
    target = context["repeat"]
    if scenario == "interaction":
        target ^= context["urgent"]
    if scenario == "nonstationary" and step >= horizon // 2:
        target = not target
    return {"answer": 0.1 if target else 0.9, "escalate": 0.9 if target else 0.1}


def expected_reward(policy, rewards) -> float:
    return sum(policy[a] * rewards[a] for a in policy)


def tilt_oracle(baseline: dict, rewards: dict, expert: Expert) -> dict:
    """Exact best distribution in the clipped-score tilt family for this context.

    Linear-fractional optimization over positive weights attains a vertex;
    labels above a reward threshold use max weight and those below use min.
    Enumerate sorted thresholds (K+1), preserving the exploration mixture.
    """
    actions = sorted(baseline, key=lambda a: -rewards[a])
    candidates = [
        tilted(baseline, {a: 1.0 if a in actions[:k] else -1.0 for a in actions}, 1.0, expert)
        for k in range(len(actions) + 1)
    ]
    return max(candidates, key=lambda p: expected_reward(p, rewards))


def run_one(
    *,
    scenario: str,
    method: str,
    seed: int,
    steps: int,
    learn_every: int,
    evaluation_steps: int = 256,
    expert: Expert | None = None,
    half_life_seconds: float | None = DEFAULT_HALF_LIFE_SECONDS,
    step_seconds: float = 3600,
) -> dict:
    if scenario not in SCENARIOS or method not in METHODS:
        raise ValueError("unknown scenario/method")
    if min(steps, learn_every, evaluation_steps) < 1:
        raise ValueError("steps, learn_every and evaluation_steps must be positive")
    validate_half_life(half_life_seconds)
    if finite(step_seconds, "step_seconds") < 0:
        raise ValueError("step_seconds must be nonnegative")
    expert = expert or Expert(max_calls=10000, max_rounds=2)
    context_rng = np.random.default_rng(seed)
    outcome_rng = np.random.default_rng(seed + 100_000)
    action_rng = np.random.default_rng(seed + 200_000)
    eval_rng = np.random.default_rng(seed + 300_000)
    model = SyntheticAnswerer(scenario)
    choice = synthetic_choice()
    sequence = 0
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    current_time = epoch

    def identifier():
        nonlocal sequence
        sequence += 1
        return f"benchmark-{seed}-{sequence}"

    with tempfile.TemporaryDirectory(prefix="nudge-benchmark-") as directory:
        with Nudge(
            choice,
            store=Path(directory) / "nudge.db",
            model=model,
            proposer=SyntheticProposer() if method == "discovery" else None,
            questions=QUESTIONS[:3] if method == "fixed_features" else (),
            expert=expert,
            seed=seed + 200_000,
            id_factory=identifier,
            clock=lambda: current_time,
        ) as policy:
            regret = constrained = total_expected = baseline_expected = observed_reward = 0.0
            max_ratio = 1.0
            counts = Counter()
            curve = []
            learning_calls = 0
            second_half_regret = 0.0
            for step in range(steps):
                current_time = epoch + timedelta(seconds=step * step_seconds)
                state = context_at(context_rng, step)
                rewards = reward_means(scenario, state, step, steps)
                baseline = model.ask(state, ()).probabilities
                if method in ("baseline", "uniform"):
                    probabilities = (
                        dict(baseline) if method == "baseline" else {a: 0.5 for a in baseline}
                    )
                    action = str(
                        action_rng.choice(list(probabilities), p=list(probabilities.values()))
                    )
                    decision = None
                else:
                    decision = policy.ask(state=state)
                    probabilities, action = decision.policy, decision.action
                # Common potential-outcome randomness across compared methods.
                outcomes = {a: float(outcome_rng.random() < rewards[a]) for a in rewards}
                observed_reward += outcomes[action]
                counts[action] += 1
                value = expected_reward(probabilities, rewards)
                total_expected += value
                baseline_expected += expected_reward(baseline, rewards)
                regret += max(rewards.values()) - value
                if step >= steps // 2:
                    second_half_regret += max(rewards.values()) - value
                constrained += (
                    expected_reward(tilt_oracle(baseline, rewards, bounded_expert()), rewards) - value
                )
                max_ratio = max(
                    max_ratio, *(probabilities[a] / baseline[a] for a in baseline if baseline[a])
                )
                if decision:
                    policy.feedback(decision.id, resolved=outcomes[action])
                    if (step + 1) % learn_every == 0:
                        result = policy.learn(half_life_seconds=half_life_seconds, discover_questions=True)
                        learning_calls += result.model_calls
                if (step + 1) % learn_every == 0 or step == steps - 1:
                    curve.append(
                        {
                            "step": step + 1,
                            "regret": regret,
                            "tilt_oracle_regret": constrained,
                            "expected_reward": total_expected / (step + 1),
                        }
                    )
            if method not in ("baseline", "uniform") and steps % learn_every:
                learning_calls += policy.learn(half_life_seconds=half_life_seconds, discover_questions=True).model_calls
            final_expected = final_baseline = 0.0
            # Freeze learning before this independent test stream. No test feedback
            # is supplied to the learner and no discovery runs on these contexts.
            for step in range(evaluation_steps):
                state = context_at(eval_rng, 1_000_000 + step)
                rewards = reward_means(scenario, state, steps, steps)
                baseline = model.ask(state, ()).probabilities
                probabilities = (
                    baseline
                    if method == "baseline"
                    else {a: 0.5 for a in baseline}
                    if method == "uniform"
                    else policy.ask(state=state).policy
                )
                final_expected += expected_reward(probabilities, rewards)
                final_baseline += expected_reward(baseline, rewards)
            report = policy.report()
            return {
                "scenario": scenario,
                "method": method,
                "seed": seed,
                "steps": steps,
                "learn_every": learn_every,
                "evaluation_steps": evaluation_steps,
                "half_life_seconds": half_life_seconds,
                "step_seconds": step_seconds,
                "cumulative_pseudo_regret": regret,
                "second_half_regret": second_half_regret,
                "tilt_oracle_regret": constrained,
                "expected_reward": total_expected / steps,
                "baseline_expected_reward": baseline_expected / steps,
                "realized_reward": observed_reward / steps,
                "test_expected_reward": final_expected / evaluation_steps,
                "test_baseline_reward": final_baseline / evaluation_steps,
                "test_gain": (final_expected - final_baseline) / evaluation_steps,
                "action_counts": dict(counts),
                "max_probability_ratio": max_ratio,
                "learning_model_calls": learning_calls,
                "questions": [q.name for q in report.questions],
                "learned_weight": sum(a["learned_weight"] for a in report.actions.values()),
                "curve": curve,
            }


def summarize(runs):
    groups = {}
    for run in runs:
        groups.setdefault(
            (run["scenario"], run["method"], run["half_life_seconds"], run["step_seconds"]), []
        ).append(run)
    result = []
    for (scenario, method, half_life, step_seconds), members in groups.items():
        summary = {"scenario": scenario, "method": method, "seeds": len(members)}
        summary["half_life_seconds"] = half_life
        summary["step_seconds"] = step_seconds
        for metric in (
            "cumulative_pseudo_regret",
            "tilt_oracle_regret",
            "second_half_regret",
            "test_gain",
        ):
            values = [r[metric] for r in members]
            summary[metric] = {
                "mean": float(np.mean(values)),
                "seed_se": float(np.std(values, ddof=1) / math.sqrt(len(values)))
                if len(values) > 1
                else None,
            }
        result.append(summary)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learn-every", type=int, default=200)
    parser.add_argument("--seeds", default="0,1,2")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS))
    parser.add_argument("--methods", default="baseline,no_features,discovery")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--half-life-seconds",
        default=DEFAULT_HALF_LIFE_SECONDS,
        type=lambda value: None if value.lower() == "none" else float(value),
        help="training weight half-life; 'none' gives unit weights",
    )
    parser.add_argument(
        "--step-seconds",
        type=float,
        default=3600,
        help="simulated elapsed time between decisions (default: one hour)",
    )
    args = parser.parse_args()
    runs = [
        run_one(
            scenario=scenario,
            method=method,
            seed=int(seed),
            steps=args.steps,
            learn_every=args.learn_every,
            half_life_seconds=args.half_life_seconds,
            step_seconds=args.step_seconds,
        )
        for scenario in args.scenarios.split(",")
        for method in args.methods.split(",")
        for seed in args.seeds.split(",")
    ]
    result = {
        "format_version": 2,
        "note": "Synthetic scripted features/proposer; not an LLM benchmark. "
        "Pseudo-regret uses expected reward under the actual sampled distribution. "
        "Tilt oracle knows true rewards but respects score clipping and baseline mixing. "
        "Uniform can beat that constrained oracle. Seed SE is descriptive, not certification.",
        "summary": summarize(runs),
        "runs": runs,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result["summary"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()

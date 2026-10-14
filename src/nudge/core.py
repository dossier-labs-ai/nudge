"""The synchronous public feedback loop, with atomic fitted policy activation."""

from __future__ import annotations

import hashlib
import threading
import uuid
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from .discovery import CallBudget, discover, hydrate, question_key
from .math import fit, reward_scale, policy_probabilities, vector
from .recency import DEFAULT_HALF_LIFE_SECONDS, utc_timestamp, weight_rows, validate_half_life
from .store import Store
from .types import (
    Answerer,
    Choice,
    Decision,
    Expert,
    LearnResult,
    Proposer,
    Question,
    Report,
    text,
    validate_answer,
)


class Nudge:
    def __init__(
        self,
        choice: Choice,
        *,
        store: str | Path,
        model: Answerer,
        proposer: Proposer | None = None,
        expert: Expert | None = None,
        questions: Sequence[Question] = (),
        seed: int | None = None,
        id_factory: Callable[[], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ):
        if isinstance(model, str) or not callable(getattr(model, "ask", None)):
            raise ValueError(
                "v0 requires an Answerer object; model-name resolution is not implemented"
            )
        text(model.identity, "answerer identity")
        if proposer is not None:
            text(proposer.identity, "proposer identity")
        self.choice = choice
        self.model = model
        self.proposer = proposer
        self.expert = expert or Expert()
        questions = tuple(questions)
        if (
            len({q.name for q in questions}) != len(questions)
            or len(questions) > self.expert.max_questions
        ):
            raise ValueError("questions must be uniquely named and within max_questions")
        self._rng = np.random.default_rng(seed)
        self._rng_lock = threading.Lock()
        self._learn_lock = threading.Lock()
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._store = Store(store)
        # Reward weights may change through learn(reward=...). Everything else
        # affects the meaning of the log or split and is immutable for this choice.
        contract = choice.to_dict()
        contract.pop("reward")
        contract.update(model=model.identity, expert=asdict(self.expert),
                        policy='posterior_action_utility_v1', discovery='question_bank_v1')
        initial = {
            "version": 0,
            "choice": choice.to_dict(),
            "model": model.identity,
            "expert": asdict(self.expert),
            "questions": [q.to_dict() for q in questions],
            "question_bank": [q.to_dict() for q in questions],
            "question_evidence": {},
            "scale": 1.0,
            "critics": fit([], list(choice.actions), questions, 1.0),
            "feedback_count": 0,
            "feedback_cutoff": 0,
            "training_replay": {"capacity": self.expert.max_training_rows, "sample_count": 0,
                                "decision_ids": []},
            "recency": {"half_life_seconds": None, "reference_time": None},
        }
        try:
            self._store.initialize(choice.name, contract, initial)
            live = self._store.live(choice.name)
            if live["choice"]["reward"] != dict(choice.reward):
                raise ValueError(
                    "reopen with live reward weights; change weights explicitly via learn(reward=...)"
                )
            if questions and live["questions"] != [q.to_dict() for q in questions]:
                raise ValueError(
                    "reopening cannot replace live questions; omit questions to resume"
                )
        except BaseException:
            self._store.close()
            raise

    def ask(self, *, state: str) -> Decision:
        if not isinstance(state, str):
            raise ValueError("state must be a string")
        live = self._store.live(self.choice.name)
        questions = tuple(Question(**q) for q in live["questions"])
        answer = validate_answer(self.model.ask(state, questions), self.choice, questions)
        scores = {}
        contributions = {}
        labels = ["intercept", "baseline_logit"]
        for question in questions:
            if question.kind == "categorical":
                labels.extend(
                    f"question:{question.name}:{category}"
                    for category in (question.categories or ())
                )
            else:
                labels.append(f"question:{question.name}")
        with self._rng_lock:
            for action, probability in answer.probabilities.items():
                weights = np.array(live['critics'][action]['mean'])
                terms = weights * vector(probability, answer.features, questions)
                contributions[action] = dict(zip(labels, map(float, terms)))
                scores[action] = float(sum(terms))
            probabilities = policy_probabilities(
                live['critics'], [{'jev': answer.probabilities, 'features': answer.features}],
                questions, live['scale'], self.expert)[0]
            policy = dict(zip(live['critics'], map(float, probabilities)))
            actions = list(policy)
            action = str(self._rng.choice(actions, p=list(policy.values())))
        identifier = text(self._id_factory(), "decision id")
        split_value = int(hashlib.sha256(identifier.encode()).hexdigest()[:16], 16) / 2**64
        split = "selection" if split_value < self.expert.selection_fraction else "train"
        delta = policy[action] - answer.probabilities[action]
        if live["feedback_count"] == 0:
            why = f"{action}: posterior-prior exploration; no learned feedback."
        else:
            terms = contributions[action]
            dominant = max(terms, key=lambda key: abs(terms[key]))
            why = (
                f"{action}: probability {answer.probabilities[action]:.3f} → {policy[action]:.3f} "
                f"({delta:+.3f}); predicted utility {scores[action]:+.3f}; "
                f"largest mean term {dominant} {terms[dominant]:+.3f}. Posterior exploration."
            )
        decision = Decision(
            identifier,
            action,
            dict(answer.probabilities),
            policy,
            dict(answer.features),
            why,
            live["version"],
            policy[action],
            scores,
            contributions,
        )
        self._store.insert_decision(
            self.choice.name,
            state,
            split,
            asdict(decision),
            {question_key(q): answer.features[q.name] for q in questions},
            utc_timestamp(self._clock()),
        )
        return decision

    def feedback(self, decision_id: str, **fields: float) -> None:
        values = self.choice.fields(fields)
        self.choice.utility(values)
        self._store.feedback(self.choice.name, decision_id, values)

    def decision(self, decision_id: str) -> Decision:
        """Read the original committed decision; never rerun the answerer."""
        return Decision(**self._store.decision(self.choice.name, decision_id))

    def learn(
        self,
        *,
        reward: Mapping[str, float] | None = None,
        discover_questions: bool = False,
        half_life_seconds: float | None = DEFAULT_HALF_LIFE_SECONDS,
    ) -> LearnResult:
        with self._learn_lock:
            return self._learn(
                reward=reward,
                discover_questions=discover_questions,
                half_life_seconds=half_life_seconds,
            )

    def discover_questions(self, *, half_life_seconds=DEFAULT_HALF_LIFE_SECONDS) -> LearnResult:
        """Update the question bank without changing serving features or critics."""
        with self._learn_lock:
            return self._learn(reward=None, discover_questions=True,
                               half_life_seconds=half_life_seconds, train_policy=False)

    def _learn(self, *, reward, discover_questions, half_life_seconds, train_policy=True) -> LearnResult:
        validate_half_life(half_life_seconds)
        live = self._store.live(self.choice.name)
        choice = Choice(**live["choice"])
        if reward is not None:
            choice = replace(choice, reward=reward)
        questions = [Question(**q) for q in live["question_bank"]]
        replay, observed = self._store.training_rows(choice.name)
        cutoff = replay['cutoff']
        observation_boundary = replay['decision_boundary']
        rows = [
            {**row, "reward": choice.utility(row["fields"]),
             'observation_boundary': observation_boundary}
            for row in observed
            if row["fields"] is not None
        ]
        rows, reference_time = weight_rows(rows, half_life_seconds)
        if not rows and reward is None:
            return LearnResult(
                live["version"],
                0,
                tuple(questions),
                0,
                (),
                False,
                "No feedback; baseline/live policy unchanged.",
            )
        run = uuid.uuid4().hex
        self._store.start_run(run, choice.name, live["version"], cutoff)
        budget = CallBudget(self.expert.max_calls)
        try:
            rows = hydrate(
                rows,
                questions,
                store=self._store,
                answerer=self.model,
                choice=choice,
                budget=budget,
            )
            questions, rows, events = discover(
                rows,
                questions,
                choice=choice,
                expert=self.expert,
                answerer=self.model,
                proposer=self.proposer if discover_questions else None,
                store=self._store,
                budget=budget,
                run=run,
                evidence_history=live['question_evidence'],
            )
            from .question_evidence import update_evidence
            evidence_history = update_evidence(rows, questions, list(choice.actions),
                                               live['question_evidence'], choice.to_dict())
            scale = reward_scale([row["reward"] for row in rows]) if train_policy else live['scale']
            snapshot = {
                "version": live["version"] + 1,
                "choice": choice.to_dict(),
                "model": self.model.identity,
                "expert": asdict(self.expert),
                "questions": [q.to_dict() for q in questions],
                "question_bank": [q.to_dict() for q in questions],
                "question_evidence": evidence_history,
                "scale": scale,
                "critics": (fit(rows, list(choice.actions), questions, scale)
                            if train_policy else live['critics']),
                "feedback_count": replay['feedback_count'],
                "training_replay": {"capacity": self.expert.max_training_rows,
                                    "sample_count": len(rows),
                                    "decision_ids": [r['id'] for r in rows]},
                "feedback_cutoff": cutoff,
                "selection_objective": "question_action_discrimination_v1",
                "run": run,
                "recency": {
                    "half_life_seconds": half_life_seconds,
                    "reference_time": reference_time,
                },
            }
            if not train_policy:
                for key in ('questions', 'critics', 'scale', 'feedback_count', 'feedback_cutoff',
                            'recency', 'training_replay'):
                    snapshot[key] = live[key]
            self._store.activate(choice.name, run, live["version"], snapshot)
        except BaseException as error:
            self._store.fail_run(run, type(error).__name__)
            raise
        self.choice = choice
        return LearnResult(
            snapshot["version"],
            replay['feedback_count'],
            tuple(questions),
            budget.used,
            tuple(events),
            train_policy,
            ('Activated on explicit learn(); question evidence is exploratory.' if train_policy else
             'Question bank updated; serving features and critic unchanged.'),
            training_sample_count=len(rows),
        )

    def report(self) -> Report:
        live = self._store.live(self.choice.name)
        choice = Choice(**live["choice"])
        _, rows = self._store.rows(choice.name)
        questions = tuple(Question(**q) for q in live["questions"])
        actions = {}
        for action in choice.actions:
            selected = [r for r in rows if r["action"] == action]
            labeled = [r for r in selected if r["fields"] is not None]
            actions[action] = {
                "decisions": len(selected),
                "feedback_count": len(labeled),
                "learned_weight": live["critics"][action]["n"],
                "missing_feedback_rate": 1 - len(labeled) / len(selected) if selected else None,
                "mean_reward": float(np.mean([choice.utility(r["fields"]) for r in labeled]))
                if labeled
                else None,
                "mean_fields": {
                    f: float(np.mean([r["fields"][f] for r in labeled])) if labeled else None
                    for f in choice.feedback
                },
                "mean_logged_probability_delta": float(
                    np.mean([r["policy"][action] - r["jev"][action] for r in rows])
                )
                if rows
                else None,
            }
        labels = ["intercept", "baseline_logit"]
        for question in questions:
            if question.kind == "categorical":
                labels.extend(
                    f"question:{question.name}:{category}"
                    for category in (question.categories or ())
                )
            else:
                labels.append(f"question:{question.name}")
        weights = {}
        for action, critic in live["critics"].items():
            covariance = (
                np.linalg.solve(critic["precision"], np.eye(len(labels))) * live["scale"] ** 2
            )
            weights[action] = {
                name: {"mean": critic["mean"][i], "posterior_sd": float(np.sqrt(covariance[i, i]))}
                for i, name in enumerate(labels)
            }
        return Report(
            choice.name,
            live["version"],
            len(rows),
            sum(r["fields"] is not None for r in rows),
            actions,
            questions,
            weights,
            live["feedback_count"],
            live["feedback_cutoff"],
            tuple(Question(**q) for q in live['question_bank']),
            tuple(live['question_evidence'].values()),
            training_sample_count=live['training_replay']['sample_count'],
        )

    def close(self) -> None:
        self._store.close()

    def __enter__(self) -> Nudge:
        return self

    def __exit__(self, *_):
        self.close()

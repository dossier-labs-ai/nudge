"""Public contracts. Feedback is supplied by callers, never inferred by Nudge."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Literal, Mapping, Protocol, Sequence


def finite(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


@dataclass(frozen=True)
class Choice:
    name: str
    situation: str
    actions: Mapping[str, str]
    feedback: Mapping[str, str]
    reward: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        text(self.name, "choice name")
        text(self.situation, "situation")
        for name, values in (("actions", self.actions), ("feedback", self.feedback)):
            if not values or (name == "actions" and len(values) < 2):
                raise ValueError("declare at least two actions and one feedback field")
            for key, description in values.items():
                text(key, name)
                text(description, f"{name} description")
            object.__setattr__(self, name, MappingProxyType(dict(values)))
        weights = self.reward
        if weights is None:
            if len(self.feedback) != 1:
                raise ValueError("multiple feedback fields require explicit reward weights")
            weights = {next(iter(self.feedback)): 1.0}
        if not weights or not set(weights) <= set(self.feedback):
            raise ValueError("reward weights must reference declared feedback fields")
        object.__setattr__(
            self,
            "reward",
            MappingProxyType(
                {key: finite(weights.get(key, 0.0), "reward weight") for key in self.feedback}
            ),
        )

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "situation": self.situation,
            "actions": dict(self.actions),
            "feedback": dict(self.feedback),
            "reward": dict(self.reward),
        }

    def fields(self, values: Mapping[str, float]) -> dict[str, float]:
        if set(values) != set(self.feedback):
            raise ValueError("feedback must contain exactly all declared fields; reports replace")
        return {key: finite(values[key], key) for key in self.feedback}

    def utility(self, fields: Mapping[str, float]) -> float:
        value = sum(self.reward[key] * fields[key] for key in self.feedback)
        return finite(value, "weighted reward")


@dataclass(frozen=True)
class Question:
    name: str
    prompt: str
    kind: Literal["binary", "ordinal", "categorical"] = "binary"
    paraphrase: str | None = None
    levels: tuple[str, ...] | None = None
    categories: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        text(self.name, "question name")
        text(self.prompt, "question prompt")
        if self.kind not in ("binary", "ordinal", "categorical"):
            raise ValueError("question kind must be binary, ordinal, or categorical")
        if self.paraphrase is not None:
            text(self.paraphrase, "paraphrase")
        if self.levels is not None:
            if self.kind != "ordinal" or len(self.levels) != 5:
                raise ValueError("levels require an ordinal question and five descriptions")
            for level in self.levels:
                text(level, "level description")
            object.__setattr__(self, "levels", tuple(self.levels))
        if self.categories is not None:
            if self.kind != "categorical" or not 2 <= len(self.categories) <= 32:
                raise ValueError("categories require a categorical question and 2..32 labels")
            if len(set(self.categories)) != len(self.categories):
                raise ValueError("categorical labels must be unique")
            for category in self.categories:
                text(category, "category description")
            object.__setattr__(self, "categories", tuple(self.categories))
        if self.kind == "binary" and (self.levels is not None or self.categories is not None):
            raise ValueError("binary questions cannot declare levels or categories")
        if self.kind == "ordinal" and self.categories is not None:
            raise ValueError("ordinal questions cannot declare categories")
        if self.kind == "categorical" and self.levels is not None:
            raise ValueError("categorical questions cannot declare levels")

    def normalize(self, value: object) -> float:
        if self.kind == "categorical":
            if self.categories is None:
                raise ValueError("categorical questions require categories")
            if isinstance(value, Mapping):
                if set(value) != set(self.categories):
                    raise ValueError(f"{self.name} probabilities must cover every category")
                probabilities = [finite(value[c], c) for c in self.categories]
                if any(p < 0 for p in probabilities) or not math.isclose(
                    sum(probabilities), 1.0, abs_tol=1e-6, rel_tol=0
                ):
                    raise ValueError(f"{self.name} category probabilities must sum to one")
                index = sum(i * p for i, p in enumerate(probabilities))
            else:
                index = finite(value, self.name)
            if not 0 <= index <= len(self.categories) - 1:
                raise ValueError(f"{self.name} category index is out of range")
            return index / (len(self.categories) - 1)
        value = finite(value, self.name)
        low, high = (0.0, 1.0) if self.kind == "binary" else (1.0, 5.0)
        if not low <= value <= high:
            raise ValueError(f"{self.name} must be in [{low}, {high}]")
        return (value - low) / (high - low)

    def feature_vector(self, value: object) -> tuple[float, ...]:
        if self.kind != "categorical":
            return (self.normalize(value),)
        if self.categories is None:
            raise ValueError("categorical questions require categories")
        if isinstance(value, Mapping):
            self.normalize(value)
            return tuple(float(value[c]) for c in self.categories)
        index = finite(value, self.name)
        if not index.is_integer() or not 0 <= index < len(self.categories):
            raise ValueError(f"{self.name} numeric category values must be integer indices")
        return tuple(float(i == int(index)) for i in range(len(self.categories)))

    def to_dict(self) -> dict:
        value = asdict(self)
        if self.levels is not None:
            value["levels"] = list(self.levels)
        return value


@dataclass(frozen=True)
class Expert:
    explore_floor: float = 0.05
    posterior_draws: int = 64
    max_rounds: int = 3
    max_questions: int = 32
    max_calls: int = 100000
    selection_fraction: float = 0.25
    min_split: int = 10
    min_gain: float = 0.001
    min_feature_std: float = 0.03
    max_paraphrase_mae: float = 0.15
    stability_samples: int = 16
    proposer_examples: int = 256
    max_training_rows: int = 1024

    def __post_init__(self) -> None:
        for name in (
            "explore_floor",
            "selection_fraction",
            "min_gain",
            "min_feature_std",
            "max_paraphrase_mae",
        ):
            finite(getattr(self, name), name)
        if not 0 < self.explore_floor <= 1:
            raise ValueError("explore_floor must be in (0, 1]")
        if not 0 < self.selection_fraction < 1:
            raise ValueError("selection_fraction must be in (0, 1)")
        if min(self.min_gain, self.min_feature_std, self.max_paraphrase_mae) < 0:
            raise ValueError("discovery thresholds must be nonnegative")
        for name in (
            "posterior_draws",
            "max_rounds",
            "max_questions",
            "max_calls",
            "min_split",
            "stability_samples",
            "proposer_examples",
            "max_training_rows",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if (
            min(self.posterior_draws, self.min_split, self.stability_samples, self.proposer_examples)
            < 1
        ):
            raise ValueError("sample/split/example counts must be positive")
        if self.max_training_rows < 3:
            raise ValueError("max_training_rows must be at least 3")


@dataclass(frozen=True)
class Answer:
    probabilities: Mapping[str, float]
    features: Mapping[str, object] = field(default_factory=dict)


def validate_answer(answer: Answer, choice: Choice, questions: Sequence[Question]) -> Answer:
    if not isinstance(answer, Answer):
        raise ValueError("answerer must return Answer")
    if set(answer.probabilities) != set(choice.actions):
        raise ValueError("answer probabilities must cover exactly the declared actions")
    probabilities = {a: finite(answer.probabilities[a], a) for a in choice.actions}
    if any(p < 0 or p > 1 for p in probabilities.values()):
        raise ValueError("probabilities must be in [0, 1]")
    total = sum(probabilities.values())
    if not math.isclose(total, 1.0, abs_tol=1e-8, rel_tol=0):
        raise ValueError("probabilities must sum to one")
    # Normalize only rounding error; use this canonical baseline throughout.
    probabilities = {a: p / total for a, p in probabilities.items()}
    if set(answer.features) != {q.name for q in questions}:
        raise ValueError("feature answers must cover exactly the requested questions")
    features = {}
    for question in questions:
        question.normalize(answer.features[question.name])
        if question.kind == "categorical":
            if not isinstance(answer.features[question.name], Mapping):
                raise ValueError("categorical feature answers must be probability mappings")
            features[question.name] = dict(answer.features[question.name])
        else:
            features[question.name] = float(answer.features[question.name])
    return Answer(probabilities, features)


class Answerer(Protocol):
    """Bind to a Choice on construction; identity pins model/prompt/decoding.

    Baseline probabilities must be independent of the requested question list.
    """

    identity: str

    def ask(self, state: str, questions: Sequence[Question]) -> Answer: ...


@dataclass(frozen=True)
class Proposal:
    operation: Literal["add", "revise", "drop", "add_bundle"]
    question: Question | None = None
    target: str | None = None
    questions: tuple[Question, ...] = ()

    def __post_init__(self) -> None:
        if self.operation not in ("add", "revise", "drop", "add_bundle"):
            raise ValueError("unknown proposal operation")
        if self.operation in ("add", "revise") and not isinstance(self.question, Question):
            raise ValueError("add/revise requires a question")
        if self.operation in ("revise", "drop"):
            text(self.target, "proposal target")
        if self.operation == 'add_bundle':
            if (not 2 <= len(self.questions) <= 16 or self.question is not None
                    or self.target is not None
                    or not all(isinstance(q, Question) for q in self.questions)):
                raise ValueError('add_bundle requires 2..16 questions and no question/target')
        elif self.questions:
            raise ValueError('questions only valid for add_bundle')

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ProposalContext:
    choice: dict
    questions: tuple[Question, ...]
    examples: tuple[dict, ...]
    round: int
    history: tuple[dict, ...] = ()
    question_diagnostics: tuple[dict, ...] = ()
    pending: tuple[dict, ...] = ()

    def to_dict(self) -> dict:
        return {
            "choice": self.choice,
            "questions": [q.to_dict() for q in self.questions],
            "examples": list(self.examples),
            "round": self.round,
            "history": list(self.history),
            "selection_objective": "question_action_discrimination_v1",
            "question_diagnostics": list(self.question_diagnostics),
            "pending": list(self.pending),
        }


class Proposer(Protocol):
    identity: str

    def propose(self, context: ProposalContext) -> Sequence[Proposal]: ...


@dataclass(frozen=True)
class Decision:
    id: str
    action: str
    jev: dict[str, float]
    policy: dict[str, float]
    features: dict[str, object]
    why: str
    version: int
    propensity: float
    scores: dict[str, float]
    contributions: dict[str, dict[str, float]]


@dataclass(frozen=True)
class LearnResult:
    version: int
    feedback_count: int
    questions: tuple[Question, ...]
    model_calls: int
    discovery: tuple[dict, ...]
    activated: bool
    note: str
    training_sample_count: int = 0


@dataclass(frozen=True)
class Report:
    choice: str
    version: int
    decisions: int
    feedback_count: int
    actions: dict
    questions: tuple[Question, ...]
    coefficients: dict
    learned_feedback_count: int
    feedback_cutoff: int
    question_bank: tuple[Question, ...] = ()
    question_evidence: tuple[dict, ...] = ()
    training_sample_count: int = 0
    note: str = (
        "Observed outcomes and model uncertainty; no causal or certified policy-value claims."
    )

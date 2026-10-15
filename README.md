# Nudge

Canonical repository: [dossier-labs-ai/nudge](https://github.com/dossier-labs-ai/nudge).
The website source lives in [`site/`](site/README.md).

Nudge learns action utilities from feedback and readable questions about the
state. It is an experimental contextual-bandit library: no model
fine-tuning, one SQLite file, and an explicit `learn()` call to activate updates.

Requires Python 3.11+. Install with `pip install nudge-policy`, or clone and
run `uv sync`. The distribution is `nudge-policy`; the import is `nudge`.
The PyPI name `nudge` is held by an unrelated, inactive project (Evite, last
released 2012); a PEP 541 transfer request is the path to it.

```python
from nudge import Answer, Choice, Nudge

choice = Choice(
    name="support",
    situation="A support ticket and conversation so far",
    actions={"answer": "Reply now", "escalate": "Hand to a person"},
    feedback={"resolved": "1 if resolved, otherwise 0"},
)

class Baseline:
    identity = "example-baseline-v1"  # pin model, prompt, and decoding revisions

    def ask(self, state, questions):
        # A runnable fixed baseline, not a model integration.
        return Answer({"answer": 0.7, "escalate": 0.3})

with Nudge(choice, store="support.db", model=Baseline(), seed=42) as policy:
    decision = policy.ask(state="My order is damaged")
    policy.feedback(decision.id, resolved=1)
    policy.learn()
    print(policy.report())
```

Your answerer must return answers for every requested feature question. Add a
proposer to enable discovery; without one, Nudge still learns action intercepts
and weights on baseline confidence. Caller-supplied initial `Question`s also work.

## Semantics

- `ask(state=...)` makes one answerer call, samples an action, and records it
  before returning. It does not execute the action. Feedback must refer to the
  sampled action actually taken; overrides require separate treatment.
- `feedback(id, **fields)` requires all declared fields as finite numbers. Every
  report is retained; the latest report replaces the previous one. Missing
  reports never mean zero reward. Booleans are rejected; use numeric 0/1.
- `learn()` fits from a fixed latest-feedback snapshot and immediately activates
  on success. By default it trains from the retained question bank without discovery.
  It blocks its caller; other threads/instances can continue asking.
  It is not scheduled automatically. Concurrent fits use atomic compare-and-swap.
- `discover_questions()` updates the question bank and evidence without fitting or
  changing the serving critic or serving questions. Bounded replay samples inform
  discovery. Call this frequently as feedback arrives, then call `learn()` when
  ready to train the routing policy. `learn(discover_questions=True)` explicitly
  combines both operations. `report().question_bank` and `.question_evidence`
  expose retained hypotheses separately from active `.questions`.
- Learning visits labeled decisions newest first. By default a decision's weight
  halves every seven days relative to the newest labeled decision. Set
  `learn(half_life_seconds=3 * 86400)` for a three-day half-life, or `None` for
  unit weights. Fitting and question selection use the same decay weights.
  Late/corrected feedback retains the original decision's age.
- `Expert(max_training_rows=1024)` caps each learning/discovery update, including
  feature hydration and evidence, at that many unique labeled decisions. Feedback
  maintains recent, uniform-history and action-balanced surprise pools in SQLite;
  learning does not scan the audit log. `training_sample_count` on learn results
  and reports distinguishes fitted sample size from total feedback count.
  Schema v2 requires a fresh store and refit when upgrading; preserve old stores
  for auditing. Full audit reporting still reads history. See [design](docs/design.md).
- `learn(reward={...})` explicitly changes weights and refits the logged fields.
  Reopen with those live weights. Changing the action/feedback/model contract
  requires a new choice name or store in v0.
- `decision(id)` reads the exact original sampled distribution and explanation.
- `report()` makes no model calls. It shows observed action outcomes, missing
  feedback, logged probability changes, model coefficient uncertainty, and each
  action's `learned_weight` (sum of weights in the last fitted snapshot).

Baseline probabilities are input features, not constraints on the chosen action.
`Expert(posterior_draws=64, explore_floor=0.05)` averages posterior winner votes
and mixes a uniform action distribution. Exploration starts before any feedback;
every action receives at least `explore_floor / number_of_actions` probability,
including baseline-zero actions. Coefficients describe predictions, not causes.
Discovery supplies action-by-answer outcome comparisons to the LLM and supports
atomic question bundles. Answerable proposals are retained as hypotheses without
a regressor-utility gate. Later decisions provide separate evidence for refinement.

This policy breaks the previous bounded-tilt contract. Start a new choice name or
SQLite store; preserve old stores for audit. The separate-bank contract also
requires a new choice/store when upgrading earlier unrestricted experiments.
Removed `max_drift` and `sample_after`
settings are not accepted. Historical analysis tools retain the archived rule.

See [the design](docs/design.md), [benchmarks](docs/benchmarks.md),
[adapters](docs/adapters.md), and [contributing](CONTRIBUTING.md).

## Development

```sh
uv sync --group dev
uv run python -m unittest discover -s tests -v
uv run ruff check src tests
uv build
uv run nudge-benchmark --steps 1000 --seeds 0,1,2 --output benchmark-results/run.json
```

The core has only NumPy as a runtime dependency and imports no host application
code. Tests use synthetic states and local SQLite; no credentials are required.
Database files contain verbatim states and feedback. Keep them out of source
control and distribute only synthetic fixtures.

## RouterBench

The [RouterBench integration](docs/routerbench.md) compares Rune alone with
Nudge + Rune on a proportionally sampled online stream. It scores both policies
before revealing only the chosen action's feedback, learns every 200 prompts,
and exports rolling utility and cumulative advantage as CSV, JSON and PNG.
It includes TensorZero gateway configuration, bounded concurrent replay,
checkpoint recovery, and a dedicated VM runner with provider cleanup.

```sh
uv sync --frozen --group dev --extra routerbench --extra operations
uv run python -m nudge.routerbench --help
uv run python -m nudge.routerbench_report --help
uv run python -m nudge.routerbench_vm --help
```

Real inference requires the explicit gateway and Secret Manager configuration
described in the runbook. No credentials are needed for unit tests.
See [the RouterBench runbook](docs/routerbench.md) for the live pilot and
its artifact locations.

MIT licensed. This is research software, with exploratory selection scores and
no automatic statistical certification of policy improvement.

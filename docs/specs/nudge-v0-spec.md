# Nudge v0

Nudge is an isolated Python experiment in this repository. Its portable
package documentation owns the API and local SQLite schema. It has no runtime
dependency on any host application, no staff UI/tool workflow, and no customer
deployment. This experimental placement is explicitly requested by the user;
it is not a new production shared-library location.

## Contract

`Choice` declares actions, feedback fields, and linear reward weights. An
answerer returns baseline action probabilities and question answers from the
state. `ask()` samples from learned action utilities and commits the entire decision
before returning. `feedback()` appends a complete replacement report.
`learn()` fits on a consistent snapshot of the latest feedback and immediately
activates on success. By default it trains the retained question bank without
discovery. `discover_questions()` separately updates the bank and evidence while
preserving the serving critic and features. `learn(discover_questions=True)` is
an explicit combined operation. `report()` performs no inference. Nothing computes the
user's feedback or automatically triggers learning.

The live snapshot contains configuration, question definitions, critic posterior,
reward scale, expert settings, and a feedback cutoff. Decisions refer to exact
snapshots and include their actual distributions and selected-action propensity.
An interrupted/failed learning run cannot replace the live snapshot. Concurrent
learning uses compare-and-swap activation to reject stale fits.

## Learning and scope

Per-action Gaussian linear regression uses intercept, clipped baseline logit,
and normalized question answers. Binary and ordinal answers contribute one
numeric feature; categorical answers contribute a soft one-hot probability
vector over their declared labels. Average winner votes from a fixed finite
posterior ensemble, then mix a uniform action floor. Serving and selection use
the same policy function. No score clipping, drift cap, or delayed exploration.
Baseline-zero actions can win; cold start explores the prior over every action.
The new policy requires a fresh choice name/store; old bounded-policy contracts
cannot reopen under the new runtime. Preserve their stores as historical artifacts.

Learning processes labeled decisions newest first and weights them with
`2 ** (-age_seconds / half_life_seconds)`, anchored to the newest labeled
decision timestamp. Default half-life is seven days; callers configure it on
`learn()`. Feedback arrival/correction does not refresh decision age. Fitting,
feature stability/variance checks, and held-out utility comparisons use the same weights.
Snapshots record recency settings; report exposes weighted evidence separately
from raw outcome counts. Existing decision timestamps and unit-weight snapshot
counts require no schema migration or rewriting historical data.

Discovery proposes from bounded training replay and filters flat/unstable features,
with defaults of 256 replay-sampled proposer examples, 32 retained questions and three
rounds; the supplied proposer accepts up to 16 proposals per round. This question
representation is intended to replace input embeddings, not defer rich context
representation to an embedding-based follow-up.
The LLM retains/revises/drops hypotheses using historical action-by-answer outcome
evidence. No reward-MSE or routing-policy improvement gate controls retention.
Use propensity-weighted action means within fixed answer buckets; report variation
in pairwise action advantage across buckets and coverage. Missing outcomes stay
unknown. Evidence is descriptive, not certified usefulness. All answerable proposed
questions can be retained for measurement, including complementary bundles.
Persist an initial evidence profile and decision-insertion boundary for each exact
question/objective. Only decisions arriving after this boundary count as subsequent
validation; corrections or late feedback on earlier decisions do not. Subsequent
profiles and direction comparisons guide LLM refinement. The separate policy fit
uses the same capped replay sample. Final benchmark outcomes never enter discovery or fitting.
Keep rounds, questions, examples and inference calls bounded.

Training uses at most `Expert.max_training_rows` (default 1024) unique labeled
decisions per update. Feedback incrementally maintains bounded recent, uniform
historical and action-balanced surprise reservoirs as references to the durable
decision log. Learning reads bounded membership and indexed payloads, never a
full-history scan. The cap applies to discovery, feature hydration, evidence and
regression. Total audit counts and sample counts remain separate. Schema v2
requires a fresh store/refit; preserve v1 stores as historical evidence. Proposal
history is indexed and pending state is capped at 64. Live proposers do not
prefetch features outside the learning budget. Full audit reporting is not a
bounded training operation. Fixed-size row work is not a fixed latency guarantee.

Within the capped training set, discovery replay uses the durable decision log, not a second copy of experience.
Approximately equal recent, uniform historical and informative samples are drawn
without replacement from positive-weight sampled decisions. Informative sampling
favors residual surprise and rare actions/answer groups. The fit still uses each
decision once with its original recency weight. Proposers receive sampled IDs,
original action propensities and snapshot versions, plus bounded prior proposal
diagnostics for the same choice/reward. History supports targeted refinement.
Proposer examples contain observed feedback and question answers, not the serving
regressor's coefficient contributions, predicted scores or explanatory text.
Project records after replay sampling so this context bound preserves sample
membership, full prompt text, selected-action propensities and provenance.
Action outcome differences and effective counts accompany new comparisons.
Edit results explicitly report the retained bank after every attempt. Rejected
bundles install none of their members; retrying an absent question requires an
addition, while revisions and removals target only current retained questions.
Old MSE/policy-utility history retains its original objective label. Initial and
subsequent evidence profiles guide revisions. Rejections can trigger
another bounded round; an empty proposal list or exhausted budget stops discovery. Corrections
replace outcomes on the next snapshot; exact-question answer caches survive restart.

Question and model identities are versioned and cached. Providers must preserve
baseline probabilities independently of the feature question list. Literal
historical regeneration of model responses is not promised.

The new `discovery=question_bank_v1` contract requires a new choice name/store
when upgrading an older run. Preserve old stores as audit artifacts; do not
silently treat old policy-selected questions as independently validated hypotheses.

## Boundaries

- Real inference must use TensorZero. A standalone example gateway configuration
  belongs to the package; it does not change or deploy the production gateway.
- The live RouterBench pilot uses an isolated local gateway and native Rune
  protocol provider bridge, plus GLM-5.2 with explicit thinking enabled. Its
  runbook owns the pinned partial-feedback protocol and experiment artifacts.
- The core is model-independent. A batched log-score adapter accepts scores
  from HF or other backends without requiring their runtimes in the package.
  Pure TypeSafe request/response codecs follow the supplied documentation;
  no direct provider endpoint is required for core operation. Real checkpoint
  execution and live gateway testing require an explicitly configured backend.
- Pure Python tests are exercised by a co-located Vitest launcher and can also
  run independently when the directory becomes a separate repository.
- No CI publishing, PyPI name reservation, or public release is authorized.
- Utility estimation is exploratory off-policy evaluation, not statistically
  certified OPE after adaptive selection. No shared-action model or automatic scheduling.

## Acceptance

Demonstrate persistence/restart, prior exploration, baseline-zero action override,
feedback corrections, atomic activation and concurrent asks, reproducible
benchmarks including no-signal/weak-support/noisy/interaction conditions,
and independently buildable/installable source and wheel distributions.

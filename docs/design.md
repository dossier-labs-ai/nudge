# Design and v0 boundaries

## Policy

For action a, the regression features are an intercept, the baseline logit
clipped to [-4,4], and normalized question answers. Binary questions return
probabilities in [0,1]. Ordinal answers use [1,5], mapped to [0,1] for regression;
categorical questions return a probability vector over their declared labels and
contribute one feature per label. The public Decision retains the original
answer representation. Feature coefficient names have a `question:` prefix to
avoid collision with `intercept` or `baseline_logit`.

Each action has a Gaussian linear reward model, prior standard deviation R/8
and observation-noise standard deviation R/2. R is the observed reward range.
With constant outcomes it is max(abs(outcome), 1); with no outcomes it is 1.
Extreme reward units outside the supported numerical range must be rescaled.
Precision is stored in reward/R units and means in reward units. Cholesky solves
produce posterior samples without explicitly inverting a covariance matrix.

Use a fixed finite ensemble of posterior coefficients, including before any
feedback. For each of 64 draws, the highest predicted utility wins; divide tied
votes equally. Average these votes and mix `explore_floor / number_of_actions`
into every action (default uniform mixture share 0.05). Antithetic draws use a
stable seed per action name, independent of dictionary serialization order.
Serving and evaluation call exactly the same function. The ensemble is the
actual policy, not a separate approximation of the serving distribution.

Baseline probability is only a regression feature. There is no score clipping,
drift limit, baseline mixture, or twenty-observation threshold. A baseline-zero
action can dominate once its learned utility warrants it. Cold start explores
the prior, not the baseline. The uniform floor implies ongoing exploration cost;
this is not a claim of asymptotic best-action optimality. Historical tilt
comparators live in `historical.py` and are not used by serving or discovery.

## Learning lifecycle

`learn()` reads a fixed latest-feedback snapshot, fills missing feature answers,
optionally discovers questions, refits, then atomically installs a new immutable
snapshot. New feedback arriving mid-run is used next time. A concurrent learner
that loses compare-and-swap gets `ConcurrentLearnError`; no implicit retry or
overwriting another fit. `ask()` pins the policy before inference, so an in-flight
ask may correctly return the previous version after activation.

Labeled decisions are processed newest first. With half-life H seconds, weight
`w_i = 2 ** (-(t_newest - t_i) / H)`, where t_i is decision time and t_newest is
the newest labeled decision time. The default H is seven days. Set
`learn(half_life_seconds=...)` per fit; `None` gives unit weights. Record the
half-life and reference timestamp in the immutable snapshot's `recency` field.
Late or corrected feedback does not rejuvenate an old decision. Idle time alone
does not change relative evidence. Equal timestamps have equal weights; insertion
order breaks chronological ties. No observations are deleted.

In reward/R units the weighted posterior precision is `64 I + 4 X' W X` and
the natural parameter is `4 X' W y`. Decay reduces both evidence and certainty;
weights are not renormalized back to the original row count. The stored critic
`n` is the sum of weights, reported as evidence mass. This is
weighted evidence mass, not the importance-sampling effective sample size.
Raw feedback counts and observed outcome means remain unweighted in reports.
`actions[action].learned_weight` exposes the last fitted evidence mass.
The historical reward range still defines R for prior/noise scale; recency
does not change that separate scale convention.

No SQL transaction spans inference or fitting. Database transactions can briefly
contend; this is not a hard real-time nonblocking guarantee. Caller-provided
answerers must support concurrent calls if the caller uses concurrent asks.
The sampler seed controls action randomness within one instance;
it is not persisted. For deterministic experiments, also supply unique seeded
`id_factory` values, since decision IDs determine the split. The benchmark does
this. A `clock` callback supplies timezone-aware datetimes for reproducible time
decay experiments; production defaults to UTC wall time. Production IDs default
to UUID4. Duplicate IDs fail rather than overwrite.

No-feedback `learn()` is a no-op unless explicitly changing reward weights.
Reward changes create fitted snapshots. Actions, feedback definitions, situation,
answerer identity, and expert settings are fixed for a named choice in one store.
Use a new name/store for those changes. Reopen with live reward weights; omit
initial questions to resume the learned list.

## Discovery

Question discovery is independent of routing-model training. `discover_questions()`
updates `question_bank` and `question_evidence` in an immutable snapshot, preserving
the serving questions, critic, scale and fitted-feedback cutoff. `learn()` trains
the bank on the bounded replay sample; discovery is off by default. The explicit combined
operation is `learn(discover_questions=True)`. `report()` exposes both bank and
serving questions. Discovery-only changes increment the snapshot version but do
not change serving probabilities. Compare-and-swap protects both operations.

Labeled prompts retained in bounded replay participate in discovery. The old deterministic
75/25 split remains in logs for historical evaluation tools, not as a discovery
data exclusion. Proposer subsample surprise is relative to observed action-average reward;
it does not depend on the routing regressor. Final benchmark prompts remain
outside the learner. No classifier is fitted or evaluated during discovery-only work.

The SQLite decision log remains the complete audit store. Training draws at most
`Expert.max_training_rows` (default 1024, minimum 3) unique labeled decisions.
Feedback maintains three bounded candidate pools as ID references in the same
database: newest B by decision timestamp, uniform historical bottom-B hash
priorities, and per-action informative reservoirs of max(1, floor(B/A)) members.
Candidate membership is at most 2B + A * max(1, floor(B/A)); for A <= B this is
at most 3B. No prompt or outcome payloads are duplicated in these pools.
Selection draws roughly thirds recent/uniform/informative, excludes duplicate
IDs, round-robins informative action strata, and fills overlap from remaining
candidates. Small histories (N <= B) are used in full.

Informative admission uses a deterministic exponential race weighted by
1 + min(10, abs(observed utility - decision-time predicted utility) / decision-time
reward scale). Corrections recompute that row's priority but do not change its
timestamp or increase the unique-feedback count. Informative membership is a
heuristic: evicted candidates are not rescanned after a correction or reward
change. Uniform history remains independently retained. Sampling does not turn
logged action propensity into a replay inclusion probability; evidence describes
this biased replay mixture, not unbiased population performance.

Each learning call reads only bounded membership plus indexed decisions/latest
feedback. Hydration, all discovery rounds, evidence and fitting use that same
sample. Complete logs grow for auditing; `report()` and offline diagnostics may
scan them. The cap bounds row/model work for fixed questions, actions, rounds,
and input lengths, not exact latency: SQLite index lookups grow logarithmically,
and variable prompt lengths/provider latency still matter. Seven-day decay still
applies by default; use `half_life_seconds=None` for equally weighted old examples.

Within this training sample, each proposer sample has approximately equal parts:
newest decisions, uniform historical decisions, and informative historical
decisions. Draw without replacement across all parts; small buffers return every
eligible decision once. Zero-weight and unlabeled decisions are excluded.
Informative priority is one plus residual-magnitude rank, inverse action frequency,
and mean inverse answer-bucket frequency, with inverse frequencies scaled by the
eligible row count. Categorical buckets use argmax labels; scalar answers use four
bins. These are discovery heuristics, not estimates of treatment advantage.
All positive-weight training-sample rows remain eligible; decay still controls
regression evidence. The recent portion separately guarantees recent examples.

Sampling is deterministic from ordered decision IDs, exact questions and round
number, independent of live action randomness. Each request records sampled IDs,
sampling source, original action propensity and policy version. Original propensity
is not a replay inclusion probability. Regression uses the bounded sample and
original decay weights, without duplicate rows. Subsequent learns
read latest corrected outcomes, while original timestamps and decisions persist.

Proposal context includes diagnostics from up to the latest sixteen proposal
batches for the choice, filtered to the same declared reward/objective. Nested
history and raw prior examples are excluded. Run status distinguishes failed,
in-progress and activated attempts. History survives restart and includes earlier
rounds, so the proposer can revise rejected wording or avoid redundant edits.
The v6 proposer prompt reasons about relative action advantage, missing outcome
coverage and targeted refinement. Historical records retain their original
objective labels: older reward-MSE gains must not be interpreted as utility gains.
Before each round, current questions receive the descriptive evidence below.
The proposer chooses additions, revisions and drops. Answerable proposals are
retained as hypotheses; neither MSE nor immediate routing utility vetoes them.

Proposals can add/revise/drop questions, or add a bundle of 2..16 questions
atomically. Every new bundle member must pass the answer checks. The linear critic remains additive;
nonlinear interactions require an appropriate composite question.
Pending proposals are incrementally maintained in a bounded table (latest 64)
independently of the sixteen-batch history window and included in the next proposer context. An
uncertain result is not a demonstrated failure. Later measured results resolve
the same proposal; the LLM chooses when new evidence warrants revisiting it.
New questions must vary on observed
rows and survive a distinct paraphrase check on at most sixteen observed rows.
Feature variance and paraphrase error are time-weighted. Categorical checks use
the complete probability vector: variation is the root mean squared Euclidean
distance from the mean vector, and paraphrase error is total variation distance.
They do not collapse unordered labels to a scalar category index.

### Question evidence and subsequent observations

`question_evidence.py` uses chosen feedback and logged propensities, without a
regressor. Bucket categorical answers by argmax label, scalar answers into four
fixed intervals. Within each bucket/action, compute a self-normalized weighted
mean using recency weight divided by logged action propensity. Report counts and
ESS. A comparison requires at least two effective observations per cell and no
structural zero logging support for that action in the bucket. Unsupported cells
remain unknown; they do not prevent retaining a question.

For each action pair observed in at least two buckets, compute the difference
between action means per bucket and subtract its average across those buckets.
`discrimination_rms` summarizes the resulting contrasts. A question that merely
makes all actions equally better or worse has zero contrast in a balanced example.
This is descriptive evidence, not an unbiased finite-sample estimate or a
significance test. Coarse buckets, sparse observations, selective feedback and
changing context mix can mislead it; soft answer variation is partly lost.
Questions can be redundant. The LLM should consider mechanisms and coverage, not
maximize this score or fill a quota. Retaining a hypothesis is not proving it useful.

At first retention, freeze the exact question's initial profile and maximum
decision insertion sequence, including already logged but unlabeled decisions.
On subsequent runs, update the current profile from all latest feedback and a
separate subsequent profile using only decisions inserted after that boundary.
Old feedback corrections do not become new validation. Compare pairwise contrasts
on common answer groups and report sign agreement and mean cross-product. These
are correlated descriptive comparisons, not independent trials or certified gains.
The initial profile survives restart. Changing wording or objective starts a new
profile. New bank questions are backfilled on subsequent observations during
discovery; serving only asks the active policy's questions until the next fit.

### Archived policy-utility evaluator

The following describes `utility.py`, retained for reproducing the previous
experiment. It is no longer called by question discovery or retention.

Candidates compete using the declared `Choice.utility(fields)` reward, without
task-specific actions or metrics. Both critics fit only training rows. Evaluate
on the original selection partition, never the prioritized replay sample.
For each selection row, with chosen action a, observed reward r, logged propensity
mu(a|x), and candidate policy pi, compute:

`DR(pi) = sum_b pi(b|x) * q(x,b) + pi(a|x)/mu(a|x) * (r - q(x,a))`.

Use the same current training-only critic q as the control variate for both
policies. This keeps identical policies' paired difference exactly zero even
when candidate reward predictions improve. No unchosen reward is supplied to the
learner or inferred as an observed fact. The regression model is a variance
reduction device; logged propensities provide the observed residual correction.
Report recency-weighted means of both estimates and their paired difference.
MSE remains in proposal records solely as a regression diagnostic.

Policies use the same finite posterior winner ensemble as serving, without
consuming the serving RNG. Each comparison holds draws fixed and shares random
numbers where dimensions align. The training-only policy is not a prediction of
the final all-data refit.

With normalized recency weights alpha_i and paired gains d_i, estimate the
standard error as `sqrt(sum(alpha_i^2 * (d_i - mean(d))^2) /
(1 - sum(alpha_i^2)))`. Record the approximate interval `gain +/- 1.96 * SE`.
Accept an edit only if `(gain - 1.96 * SE) / training_reward_scale > min_gain`.
`min_gain` now means normalized utility improvement, not normalized MSE reduction.
ESS is `(sum w)^2 / sum(w^2)`. Require selection-weight ESS at least
`max(2, min_split)` and policy-change ESS at least `min_split` unless policies
are identical. Change weights are `w * abs(pi_new(a|x)-pi_old(a|x))/mu(a|x)`.
At most 10% of total absolute probability change may concern actions with less
than two effective observations. An action below ten observations is a diagnostic,
not a blanket veto. Structural zero logging support blocks changes at that row;
unchanged common probabilities cancel in the paired comparison. Policy ESS and
importance ratios remain diagnostics. Missing/inconsistent propensities fail.
Insufficient coverage gives `deferred_coverage`; an interval that still permits
a meaningful gain gives `deferred_uncertainty`. A supported interval whose upper
end is below the gain threshold gives `no_utility_gain`. Importance ratios are
not clipped or silently self-normalized.

These are exploratory diagnostics, not a certified confidence bound. The interval
does not account for repeated adaptive selection, model/question search, temporal
correlation. Correct logging probabilities and
representative observed feedback are necessary; selective missing feedback can
bias estimates. Keep a separate final evaluation partition. The proposer sees
aggregate selection metrics, not selection examples or final-test outcomes.

Defaults: at most three rounds, 32 live questions, 100,000 logical model calls,
256 replay examples, and ten units of weighted evidence in total before discovery.
The supplied TensorZero proposer allows up to 16 proposals per round.
These are ceilings, not quotas: discovery stops on an empty proposal list or
budget exhaustion. Rejected edits are fed back into the next bounded round,
allowing the LLM to revise an unsuccessful hypothesis.
A call budget
includes proposer calls, feature backfills, and paraphrase checks. Transport
retries can produce additional HTTP attempts. Budget exhaustion during a
candidate discards that candidate and fits with already accepted questions.
Budget exhaustion filling mandatory live features fails without activation.
Bank questions are refitted on bounded replay only when policy training is requested.

New snapshots record `selection_objective=question_action_discrimination_v1`.
The runtime contract includes `policy=posterior_action_utility_v1` and
`discovery=question_bank_v1`. To migrate from either previous experiment,
retain the old database for audit and start a new choice name or store with the
new Expert settings. Do not relabel old propensities, transplant old snapshots,
or reopen an old bounded-policy choice under the new runtime. `max_drift` and
`sample_after` were removed. Historical analysis explicitly uses archived math.

Feature answers are cached by decision ID and exact question definition. A
model identity pins checkpoint, tokenizer/prompt and decoding conventions;
changing only the proposer does not change the live policy. Retrospective
features are new measurements, not promised reproductions of hypothetical old
outputs. The answerer must keep baseline probabilities independent of the
feature list.

## SQLite schema (schema version 2)

| Table | Contents |
| --- | --- |
| choices | Immutable contract and live snapshot pointer |
| snapshots | Complete immutable configuration, questions, critic means/precisions/weighted counts, reward scale, feedback cutoff, recency settings |
| decisions | Verbatim state, decision timestamp, stable split, exact snapshot, distributions, propensity, scores and explanation |
| feedback | Append-only complete reports ordered by monotonic insertion sequence |
| feature_answers | Cached retrospective and original feature measurements |
| learning_runs | Starting snapshot/cutoff, run status, activated version or sanitized error type |
| proposals | Proposer identity, complete structured prompt inputs, parsed proposals, selection diagnostics |
| replay_meta | Capacity, action list, total unique feedback count, cutoff and decision insertion boundary |
| latest_feedback | Indexed latest fields per labeled decision, updated transactionally with feedback |
| training_replay | Bounded lane memberships and priorities; references immutable decisions |
| pending | Latest 64 unresolved proposals per choice, maintained on proposal results |

The proposer system prompt is versioned in the adapter source/identity. Gateway
transport failures and malformed proposals fail the run; arbitrary provider error
bodies are not copied into the log. A process killed during learning can leave a
`running` audit row but cannot partially activate a policy. The prior snapshot
and immutable historical feedback remain available for inspection.

Upgrade procedure: preserve v1 stores unchanged as historical artifacts; create
a new store and refit by collecting/replaying authorized feedback through the
public API. There is no implicit full-history backfill on open or learn. Schema
v1 is rejected with this instruction before changing the database. Changing B
also changes the immutable Expert contract and requires a fresh choice/store.
Snapshots record replay capacity, sample count and selected IDs; total feedback
counts still describe the audit history. `LearnResult.training_sample_count`
reports the current operation's sample; `Report.training_sample_count` reports
the activated policy's sample. Discovery-only updates preserve the policy sample.

## Interpretation

`report()` shows observed outcomes for each selected action, not comparable
randomized-treatment means. Missing feedback can bias the observed sample.
Coefficient uncertainty is conditional on the Gaussian model and selected
features; it does not incorporate adaptive feature-selection uncertainty.
`why` explains mean coefficient terms and predicted utility, while probabilities
reflect posterior winner votes across all actions and uniform exploration.
Neither surface claims causal effects, similar-case counts, or certified uplift.

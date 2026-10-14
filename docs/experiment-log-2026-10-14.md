# Nudge RouterBench experiment log

This is the handoff record for the RouterBench experiments run on 2026-10-13
and 2026-10-14. It is intentionally explicit about which runs generated fresh
Rune traffic and which runs replayed a local cache.

## Current state

Question revision: replaced `option_distinction` with `has_answer_options` and
conditional `option_difficulty` (17 semantic dimensions). Rune measures presence
first, asks difficulty only when choices exist, and otherwise stores
`not_applicable`. GLM validation enforces the same relationship. Historical
artifacts remain unchanged; the revised rubric needs fresh annotations and
measurements in a new output directory before reporting new accuracy numbers.

Latest update (Rune semantic answerability follow-up): the deployment was found
ACTIVE with two replicas and min/max 2/2. That live state supersedes the older
inactive state listed below. The follow-up requested 4/4, observed three active
replicas, then restored 2/2 and left it ACTIVE. Owned local gateway/bridge were
stopped. See the saved `rune-semantic-*.json` management evidence.

The completed fresh run is
`benchmark-results/routerbench/categorical-1000-fresh2/`.

The Baseten deployment used for the experiment is restored to:

- deployment: the pilot's dedicated Baseten model and deployment
- status: `INACTIVE`
- active replicas: `0`
- autoscaling: `min_replica=1`, `max_replica=2`
- other autoscaling settings unchanged

The local TensorZero gateway and Rune bridge are stopped. No production model
or customer traffic was changed.

## Benchmark contract

The pinned prepared data is `benchmark-results/routerbench/prepared.json`:
800 training prompts and 200 frozen test prompts, balanced across six
RouterBench families. The routing objective is quality minus `10 * cost_usd`.
The benchmark records historical downstream outcomes; it does not regenerate
all candidate-model answers for every prompt.

Nudge receives feedback only for the action it selected. The report can compute
counterfactual comparisons from the dataset for evaluation, but those values do
not enter the learner.

## What worked

### Core Nudge library

The library is implemented under `src/nudge/`. It supports:

- bounded feedback learning around a baseline Rune policy;
- recency weighting;
- binary, ordinal, and categorical questions;
- categorical answers as validated soft probability vectors;
- persistent policy and inference caches;
- TensorZero-only live inference boundaries;
- RouterBench reports with paired quality, cost, utility, and regret metrics.

Categorical support is wired through the public question type, proposal schema,
feature vectors, persistence, TensorZero/TypeSafe/log-score adapters, and the
native Rune answerer.

### Categorical smoke pilot

Run: `benchmark-results/routerbench/categorical-pilot/`

- 80 training prompts, 20 test prompts;
- Rune quality: 59.988%; Nudge quality: 61.256%; gain: +1.268 pp;
- paired bootstrap interval: +0.196–2.504 pp;
- three categorical questions retained: `prompt_type`, `output_format_type`,
  and `domain_knowledge`;
- six additional binary/ordinal questions retained;
- 3,064 successful Rune calls, zero errors;
- estimated GLM proposer cost: USD 0.120.

This proved that categorical proposals can be generated, answered, persisted,
and retained. It is only an exploratory n=20 smoke test.

### Fresh 1,000-prompt run

Run: `benchmark-results/routerbench/categorical-1000-fresh2/`

Settings:

- 800 train / 200 test;
- four H100:2 replicas during execution;
- 32 inference workers;
- one discovery round;
- four proposals per learning block;
- fresh inference cache, not a replay of an earlier cache.

Results on the frozen 200-prompt test set:

| Metric | Rune | Nudge | Difference |
| --- | ---: | ---: | ---: |
| Quality | 58.438% | 59.016% | **+0.578 pp** |
| Utility | 0.579479 | 0.585129 | **+0.005650** |
| Recorded cost/prompt | 0.0004904 | 0.0005035 | +0.0000131 |
| Cumulative hindsight regret | 65.646 | 64.516 | -1.130 |

The paired quality bootstrap 95% interval is +0.371–0.795 pp. The run made
10,256 Rune calls and recorded zero inference errors. Four GLM proposer calls
were made, with estimated GLM cost USD 0.424 under the documented price
assumptions. The retained questions were:

- `task_type` (categorical), retained after 200 training examples;
- `output_format` (categorical), retained after 800 training examples.

The report is `benchmark-results/routerbench/categorical-1000-fresh2/report.json`.

## What failed or misled

### First expanded categorical attempt

Run: `benchmark-results/routerbench/categorical-1000/`

This used three learning rounds, 16 proposals, and concurrency 16. It completed
baseline prefetch and failed during candidate evaluation with a transient Rune
502. It produced no test result. Its cache contains partial successful
requests and three recorded errors.

### Cache-replay retries

Runs `categorical-1000-retry/` and `categorical-1000-retry2/` reused the first
attempt's inference cache. They were useful for recovery debugging, but they
were not fresh endpoint experiments. During cache hits, Baseten was correctly
idle; local replay should never be described as GPU utilization or fresh Rune
traffic. Both were stopped before a valid test summary.

### Fresh run retry behavior

The live HTTP layer already retried some transient statuses, but
`routerbench_live.Gateway` used only two attempts. That made a pair of transient
502 responses fatal to a long run. It now uses five bounded attempts with 0.5 s
exponential backoff. The completed fresh run used this change and had zero
inference errors.

Validation completed after the change: the Nudge unittest suite, Ruff, and
repository docs check passed. The full repository build had passed before this
retry-only change; rerun it before shipping the package.

## Why the improvement stayed small

The completed run did retain categorical questions, but retention did not mean
that the questions were sufficient model-selection features.

1. **Chosen-arm feedback is the dominant limit.** Nudge sees quality and cost
   only for the selected model. The supervised all-outcomes diagnostic, which
   is allowed to see every model's outcome during training, reaches 63.246%
   test quality on this split. Nudge reaches 59.016%, leaving a 4.23-point
   information gap.
2. **The categories are coarse.** `task_type` distinguishes broad prompt forms,
   while the useful RouterBench distinctions are family-level differences such
   as MMLU, ARC, HellaSwag, MBPP, and Winogrande. Many prompts collapse into
   `other`; `domain` was proposed but rejected as no-gain.
3. **Useful features arrived late.** `task_type` was retained after 200 rows,
   but `output_format` was not retained until the final 800-row update. Most
   training decisions therefore used only the first feature.
4. **The completed run deliberately used a small discovery budget.** One round
   and four proposals per block made the experiment finish reliably, but it was
   less search than the earlier three-round/16-proposal attempt.
5. **The policy is a bounded correction to Rune.** It can tilt Rune's action
   distribution, but it cannot make large reallocations from sparse evidence.
   Some arms received very little feedback; Yi-34B had one observation and
   GPT-4 had 44.
6. **The predictor objective is indirect.** Question retention minimizes
   predictive error on observed chosen-arm rewards. It does not directly select
   questions that maximize counterfactual routing utility or identify the best
   model family.

The correct interpretation is a statistically positive but modest gain, with
substantial headroom. The categorical implementation worked; the information
available to the learner and the discovery objective remain the blockers.

## Next-session starting points

### Full-sample oracle factor follow-up

At the user's request, a separate offline search used all 1,000 prompts and
all recorded model outcomes. The unrestricted oracle is 91.8% quality at
USD 0.230650944 total downstream cost, mean utility 0.91569349056.

The new `nudge.routerbench_factors` audit found a 40-factor prompt-text policy
that reproduces all three numbers exactly. It combines 15 grouped keyword or
structural factors with 25 literal token-presence factors and maps their bit
patterns to models. It creates 987 cells, including 976 singletons: exact
sample reconstruction, not a compact semantic explanation. A 24-factor prefix
already reaches 91.8% quality at USD 0.307940812. The topic/structure-only search
uses 37 factors and reaches 89.4% at USD 0.359953844.

Exact rules and counts are in `benchmark-results/routerbench/oracle-factors-lexical.json`;
structural and quality-prefix artifacts are adjacent. See `routerbench.md` for
the full factor set, reproducible command, and method. No inference calls or
online-policy changes were made. Selection is greedy, with redundant-factor
pruning; the resulting factor count is not proven globally minimal.

### Semantic-factor follow-up

Following rejection of literal word factors, GLM-5.2 thinking annotated all
1,000 prompts against 16 fixed semantic dimensions through a separate local
TensorZero gateway. Inputs contained prompt text only. All labels completed;
two bounded retries corrected the last batch's seven-for-eight count mismatch.
There were 127 real serverless calls and zero transport failures. The gateway
is stopped. No Rune activation or online policy changes occurred.

All 16 semantic dimensions jointly reach quality 88.275%, utility
0.87634671916, total historical downstream cost USD 0.640328084. This mapping
still has 516 cells / 350 singletons. The best supported sample-fit tree uses
seven dimensions and nine rules (minimum 22 prompts/rule), reaching 81.05%
quality / 0.791792478 utility. Its five-fold out-of-fold utility is 0.772484208.
The measured semantic vocabulary therefore does not explain the full 91.8%
oracle, and the unrestricted score must not be presented as a compact router.

Useful distinctions include code data structure (strings vs numeric/sequences),
quantitative operation (multiplicative vs proportional vs formal math), edge
cases, constraints, and reasoning depth in combination. Exact results, rubric,
selection order, model assignments and reproduction are in `routerbench.md` and
`benchmark-results/routerbench/semantic-analysis.json`.

### Rune semantic answerability follow-up

Completed all 1,000 prompts / 16,000 native Rune Choice labels through
TensorZero. Batched-head versus single-head probe agreed on all 128 top labels;
probabilities differed by as much as 0.0926666. Full run uses 16 heads/call,
32 workers. Total 1,128 real requests including probes, zero failed logical
calls. Rune was already ACTIVE; user-authorized scaling requested 4/4 and
cleanup restores the captured 2/2 ACTIVE state, not the older inactive state.

Agreement with GLM semantic judgments: 70.84375% overall. Strongest useful
dimensions include reference tracking (90.6%), knowledge depth (81.6%), subject
(79.9%), and code structure within coding tasks (87.06%). Weak dimensions include
reasoning steps (55%), constraints (53%), and option distinction (27.1%). The
last recognizes only 6/334 prompts with no offered answer options.

Frozen GLM nine-rule policy with Rune labels: utility 0.764927 versus 0.791792
with GLM labels. Refitting supported rules on Rune labels reaches 0.791018 on
this sample. Four-leaf Rune tree out-of-fold utility is 0.773656. Complete
16-label grouping reaches 0.895224 but has 773 cells/647 singletons, so it is
not evidence of better semantics or generalization. No learner policy changed.

Report: `benchmark-results/routerbench/rune-semantic-report.json`; per-prompt
probabilities: `rune-semantic-measurements.json`. Full metric table, command,
boundaries, and implications are recorded in `routerbench.md`.

### Earlier proposed follow-ups (prior to factor analysis)

Use the completed report and this log as the baseline. Do not count cache
replays as fresh Rune traffic. The highest-value experiments are:

- add richer prompt-derived task/family classifiers whose labels align with
  capability boundaries, while keeping labels observable at decision time;
- score candidate questions by held-out routing utility or regret, not only
  chosen-arm reward MSE;
- expose more counterfactual feedback where the serving/evaluation system can
  provide it, or use a principled off-policy estimator with logged propensities;
- give discovery enough training blocks and proposals to stabilize the feature
  set before the test pass;
- keep the five-attempt bounded retry policy and require one completed test pass
  before reporting an uplift.

### Frozen 800/200 semantic regressor

Measured the existing Gaussian linear critic using saved original 16-question
Rune probabilities. On the 200 test prompts, chosen-feedback bounded tilt gives
0.585693 utility; chosen-feedback greedy selection gives 0.669325. Full training
outcomes give 0.587040 with bounded tilt and 0.778440 with greedy selection
(79.625% quality). Rune baseline on the same test set is 0.579479 utility.

Saved `semantic-regressor-test200.json`; reproduction and information-setting
boundaries are in `routerbench.md`. No live calls or policy changes. All 80 unit
tests pass, including a test that changes test outcomes and verifies identical
predictions and reward scale. Ruff passes for the evaluator and its tests.

### Discovery replay and proposal memory

Added `SamplingReplayBuffer` over the existing decision log. Proposal examples
mix recent, uniform historical and informative training decisions without
replacement. Informative priority uses residual rank and rare action/answer
buckets. Fits and selection retain their original rows/weights; replay does not
multiply evidence. Original propensities, policy versions, sampled IDs and sample
sources accompany examples. The latest sixteen proposal batches provide bounded
history across rounds and restarts, filtered to the same choice/reward objective.
The v3 proposer prompt reasons about relative action advantage and targeted
refinement using that history. Current selection still uses held-out reward MSE;
this change is not a measured routing-utility improvement.

Verified 85 Python tests, including actual SQLite learn/reopen, corrected outcomes,
cache reuse, history persistence, distinct replay and unchanged evidence counts.
Ruff and the full repository build pass. SmartChoices source was not located in
the checked repositories, so this implements the discussed sampling semantics
rather than claiming exact parity with that implementation. No live inference,
new RouterBench utility measurement, scheduler or deployment was performed.

### General decision-utility feedback for question discovery

Replaced MSE acceptance with paired doubly robust estimates of the declared
Choice reward, using logged chosen-action propensities and a common training-only
critic. The evaluator integrates the bounded serving policy with 64 deterministic
antithetic posterior draws. It records paired utility gains, approximate intervals,
importance-weight ESS, action coverage and policy changes. Acceptance requires a
positive normalized gain after a 1.96-SE margin and sufficient support. These are
exploratory selection estimates; adaptive reuse and finite policy-integration error
are not covered by the reported interval. No RouterBench-specific objective or
model list enters the core mechanism.

The v4 proposer receives historical metrics under their original objective labels,
current-question removal effects, answer distributions and proposal diagnostics.
Rejection no longer immediately stops the loop: the LLM can revise in the next
bounded round. Categorical variation/stability use full answer vectors. MSE remains
a diagnostic only. New snapshots record the selection objective; no SQLite schema
migration or live policy mutation is required before the next explicit learn().

Validation: 96 Python tests pass. Tests cover known utility recovery from partial
feedback, unequal propensities, no gain for identical policies despite lower MSE,
weak coverage, reward units, categorical distributions, preserved historical labels
and the actual reject/refine/keep/removal-feedback learning path. Full repository
build passes (557 test files, 4,412 tests; two files/three tests skipped), as do Ruff
and docs checks. This verifies the mechanism, not a new live benchmark uplift.

### Live matched-budget utility-discovery measurement

Completed 800/200 RouterBench with the reference seed, reward, policy and
one-round/four-proposal budget, freezing reference Rune baseline probabilities.
Current test utility 0.5846528864 vs previous 0.5851290806; delta -0.0004761942
(prompt-bootstrap 95% [-0.0018062633, +0.0008217319]). Current quality 58.9647%
vs previous 59.0164%. Rune baseline utility 0.5794789649.

No questions retained. Twelve proposals rejected for coverage, four unstable;
zero scored candidates pass the uncertainty-adjusted utility threshold even with
coverage ignored. Therefore the outcome is not explained solely by the coverage
gate. This is the whole-loop comparison, not an isolated selector ablation.
4 GLM calls, 7,256 Rune feature calls, zero inference errors, 1,017 seconds.
Evidence under `benchmark-results/routerbench/utility-discovery-1000/` includes
report, learning diagnostics, decisions, SQLite state, cache/audit and source
archive. Shared model lifecycle/scaling unchanged; local gateway/bridge stopped.
99 Python tests and full build passed; paired comparison permits only 1e-12
roundoff in baseline vectors (observed maximum 1.11e-16).

## Unrestricted policy and known-question check

Removed bounded tilt, clipped utility scores and delayed exploration. Jev is an
input feature only. Serving/evaluation share a fixed posterior winner ensemble
plus a uniform action floor. Action-name seeds keep behavior invariant to SQLite
JSON dictionary ordering. New runtime requires a fresh named choice/store.
Discovery supports atomic bundles and distinguishes deferred coverage/uncertainty
from demonstrated lack of utility. Pending hypotheses survive the history window.

Offline original-16 Rune answer check, 800 training / 200 test, no live calls:

| Feedback | Retained | Selected test utility | All 16 test utility |
| --- | --- | --- | --- |
| Archived chosen outcomes | 0 | 0.651992 | 0.616917 |
| New exploration replay chosen outcomes | 0 | 0.653526 | 0.611841 |

Sixteen individual candidates and one bundle measured. Original feedback defers
all 17 for coverage; exploration defers 16 for uncertainty and one for coverage.
No selection used final-test outcomes. The replay collects one outcome per prompt
with a no-question policy, refitting every 200. These are utility-retention
diagnostics using saved answers, not live proposer/answer-stability measurements.
The first implementation check also retained none; a serialization-order bug in
posterior draw seeding was corrected before the recorded final measurement.
No hyperparameters were tuned to test utility. Evidence:
`benchmark-results/routerbench/unrestricted-retention.json`; reproduction and
interpretation are in [RouterBench](routerbench.md).

Validation: all 106 Python tests pass through the full repository build; Ruff and
Python source/wheel builds pass. No deploy, commit or push performed for this change.

## Decoupled question bank

Implemented `discover_questions()` separately from `learn()`. Discovery uses all
labeled prompts and no routing fit, MSE or policy-gain gate. LLM-proposed questions
that pass answer checks are retained hypotheses. Action-by-answer means use logged
propensities; contrasts distinguish relative action advantage from common difficulty.
Freeze initial evidence and a decision-insertion boundary; only later decisions
count as subsequent validation. Corrections and late labels on existing decisions
do not count as new evidence. Evidence, bank and serving questions persist separately.
`learn()` defaults to training only; callers can explicitly combine the operations.
The v6 prompt receives bounded evidence summaries and historical labels. New
`question_bank_v1` contract requires a fresh choice/store.

Saved-original-16 diagnostic: all 16 retained as hypotheses, using all 800 chosen
outcomes. In the retrospective first-400/next-400 exploration comparison, eight
have positive cross-products and >50% directional agreement. Reference tracking
80.0%, precision requirement 73.1%, task type 72.4%; knowledge depth 23.3% and
temporal order 20.0%. These are descriptive correlated contrasts, not independently
validated useful questions. No fresh LLM discovery or paraphrase measurements.
The separate routing result remains 0.611841 for all 16 versus 0.653526 without
questions. Preserved the earlier report; new evidence is in
`benchmark-results/routerbench/question-bank-retention.json` with reproduction in
[RouterBench](routerbench.md).

Validation for the separate bank: 112 Python tests, full repository build, Ruff,
docs checks and source/wheel build pass. Discovery-only leaves serving probabilities
unchanged, default fitting does not invoke the proposer, and final-test rewards
cannot alter the bank evidence. No live inference, deploy, commit or push.

## Live loop debugging: completed v8 experiment

Fresh v6, v7 and v8 runs start with an empty bank; reference questions are never
passed to discovery. Real GLM proposals and Rune answers exposed three operational
failures: oversized decision records in proposer input, revisions of questions
that an atomic bundle rejection never installed, and a GLM response exhausting
16,384 output/reasoning tokens without producing JSON. Compact evidence rows,
explicit retained-bank feedback, bounded parsed-output retries and checked resume
address those failures. Partial runs and failed calls remain in the audit.

These fixes establish execution, not useful learning. Retention means a proposer
hypothesis passed answer checks; it does not certify discrimination or routing gain.
The completed v8 run below distinguishes those execution checks from routing benefit.

Matched controls on the original recorded feedback, identical fit and baselines:

| Training feedback | Routing | No questions | Known reference questions |
| --- | --- | --- | --- |
| Chosen outcomes | Greedy | 0.676555 | 0.669325 |
| Chosen outcomes | Posterior | 0.651992 | 0.616917 |
| All training outcomes, diagnostic only | Greedy | 0.742267 | 0.778440 |
| All training outcomes, diagnostic only | Posterior | 0.725929 | 0.697653 |

All-outcome fitting is a separate diagnostic, never discovery feedback. The known
representation helps greedy routing with full training outcomes but does not help
this fitter on the recorded chosen outcomes. This implicates feedback/fitting and
posterior routing as additional obstacles; it does not prove that fresh discovery
has succeeded. Evidence: `live-bank-debug-v7/reference-control-with-full-diagnostic.json`
under `benchmark-results/routerbench/`. The eventual fresh comparison must use
v8's own feedback for all three question sets.

Final v8 results: 800 chosen-feedback training prompts and 200 frozen evaluation
prompts, fresh discovery from an empty bank. The final bank contains 15 questions:
three categorical (`task_type`, `domain_expertise`, `num_answer_choices`), six binary
and six ordinal. Exact wording, categories and paraphrases are in
`live-bank-debug-v8/learned-questions.json`. The proposer dropped `is_scenario_based`,
added `has_context_passage`, revised coreference/numerical questions and added
negation reasoning. Activated updates contain 21 accepted edits and ten unstable
rejections; accepted edits include revisions and a drop, not 21 unique questions.

Matched utility on v8's own chosen feedback (quality minus 10 times recorded USD):

| Routing | No questions | Fresh 15 | Reference 16 |
| --- | --- | --- | --- |
| Greedy | 0.671517 | 0.676220 | 0.693846 |
| Posterior | 0.655931 | 0.637639 | 0.612183 |

Fresh-minus-none greedy gain is 0.004703, prompt-bootstrap 95% interval
[-0.043174, 0.052747]. Fresh-minus-reference is -0.017626, interval
[-0.058419, 0.026309]. Neither establishes improvement or equivalence. Fresh
posterior routing beats reference posterior by 0.025457 [0.013678, 0.038585],
but remains below the no-question point estimate. The online final policy matches
the matched fresh posterior result exactly. Beating the original Rune baseline
(0.579479) alone does not establish question usefulness.

Separate all-training-outcomes diagnostic, same frozen questions and chosen-feedback
reward scale: greedy none/fresh/reference = 0.742267/0.765232/0.778440; posterior
= 0.723997/0.723946/0.693431. Fresh greedy gain over none is 0.022965 with interval
[-0.012949, 0.056697]. Reference greedy gain is 0.036173 [0.002979, 0.070310].
The full-outcome diagnostic never feeds discovery. These results suggest some
signal but do not demonstrate that discovery found an equally discriminative set.
The selected-action feedback and posterior routing remain additional obstacles.

Evidence under `benchmark-results/routerbench/live-bank-debug-v8/`: `matched.json`,
`report.json`, `final-snapshot.json`, `discovery-trace.json`, `learned-questions.json`,
`discovery-boundary-audit.json`, source archives, gateway image identity, execution
notes, logs, decisions, SQLite state and inference audit. The boundary audit covers
all 16 proposer calls: zero final-test examples, zero initial questions. Thirteen
calls returned usable proposals and three exhausted output without JSON. The first
failure required checked resume; two later failures recovered on the third bounded
attempt. All 24,896 audited Rune calls completed without a logical-call error.
Internal HTTP retries are not separate audit calls. Reported GLM token/cost totals
cover successful calls only; failed-call usage remains in the inference audit.

Limits: one seed and a reused exploratory test split; reference answers were saved
16-head batches while fresh questions used single-head calls; baselines were frozen
from the reference run. Later proposer inputs still reached about 266k tokens,
largely answer vectors, prompts and accumulated evidence. Compacting decision rows
helped initial calls but did not solve total context growth. No tuning used this
final evaluation result. Partial v6/v7 runs remain explicitly incomplete.

Validation: 116 Python tests; full repository build (4,412 tests passed, three
skipped); Ruff and Python source/wheel build. Owned local gateway and bridge stopped
after evaluation. Shared Rune scaling/lifecycle unchanged. No commit, push or deploy.

# RouterBench live pilot

This is a partial-feedback adaptation of
[RouterBench](https://github.com/withmartian/routerbench), not a reproduction of
the paper's supervised training experiment or a full AIQ cost sweep.

## Proportional preparation

### Live exports and dedicated VM

`run --export-dir=PATH` atomically exports `utility.csv`, `utility.json`,
`utility.png` and `status.json` every 50 decisions, before/after learning, and
on completion/failure. Install the `operations` extra for plotting. CSV columns
are prompt count, trailing-window count, Rune utility, Nudge utility and cumulative
Nudge-minus-Rune utility. JSON also includes overall means, objective, update time,
stage and policy version. Utility is probability-weighted recorded quality minus
the declared cost penalty, computed before feedback; it excludes inference overhead.

`python -m nudge.routerbench_vm inspect --model=<model-id> --environment=eval --secret-project=... --secret=...`
checks the management endpoint using an explicit Secret Manager reference.
`routerbench_vm run` supervises a fresh run, holds that environment at a minimum
of one replica, runs Rune through a local bridge and TensorZero, and uses GLM-5.2
serverless at `https://inference.baseten.co/v1` with thinking enabled. Required
flags specify root, prepared sample, gateway config/image and GCS output URI.
The bridge and runner use the same endpoint and the inspected deployment ID.
It publishes utility files and `lifecycle.json` to `OUTPUT/live/` every 30 seconds;
the lifecycle timestamp is separate from the last utility observation timestamp.
`routerbench_vm sync --root=LOCAL --output-uri=GCS` mirrors those live files locally.

On completion, failure, SIGTERM or the default 24-hour run limit, the supervisor
stops inference, requests minimum zero and deactivates the specified environment,
uploads a complete artifact archive, and powers off the dedicated VM when
`--shutdown` is supplied. Cleanup attempts are bounded and errors are recorded;
`deactivation_requested` means the API accepted the request, not independent proof
that all replicas are gone. Inspect the management state afterward. Persistent
disk and private bucket artifacts remain available after VM shutdown.
The management API is separate from inference; see the
[Baseten management reference](https://docs.baseten.co/reference/management-api/deployments/activate/activates-a-deployment).
The committed `ops/bootstrap-vm.sh` installs the package and starts the supervisor
on a dedicated Debian VM. Never put credentials into VM metadata or command flags.

For continuous bandit evaluation use `prepare --count=10000 --holdout=0`.
Every prompt is scored before its chosen-action feedback is revealed. Learning
and discovery run every 200 prompts, including a final update whose benefit is
only observable on future prompts. The run has no frozen evaluation partition.
The reporter exports trailing 200-prompt expected utility and cumulative utility
gain over Rune, including exploration losses, and can plot incomplete streams.
These are historical downstream utilities; inference overhead is separate.
Block prefetch begins online learning after the first 200 prompts, without a
full-stream baseline prefetch. Original held-out experiments remain supported.

New preparation samples without replacement proportionally across the six eligible
task families, after deduplication and outcome validation. Integer family quotas
use largest remainders; holdout quotas follow the selected family sizes.
`prepare --count=10000 --holdout=2000` creates 10,000 total unique prompts,
including 2,000 evaluation prompts: 8,000 feedback opportunities, not 10,000.
`--max-training-rows` independently controls the fit/replay cap (default 1,024);
set it explicitly for a larger experiment. Historical prepared files remain
unchanged. Their balanced-family metrics are not directly comparable to a new
proportional mix. The frozen protocol below describes historical runs only.

Systems status: Rune batches question heads within one prompt. Core discovery
hydrates rows concurrently using the adapter's configured limit; adaptive candidate
edits remain sequential. GLM missing-JSON
output receives at most three attempts, with failures audited. Retry recovery
does not bound proposer context size or guarantee completion; the v8 audit
contained contexts around 266k tokens.

Before the concurrency fix, the VM measured this bottleneck at its 2,000-prompt update on Oct 15:
8,320 successful single-head Rune calls consumed 1,687 seconds of summed latency;
four GLM attempts (one failed/retried) consumed 371 seconds. The roughly 35-minute
update completed normally. Eight candidates each received 1,024 historical answers
before 16 paraphrase checks; six were then rejected as unstable. Routing prefetch
concurrency 16 did not yet parallelize discovery. After the fix, full 1,024-call
passes measured 53–59 seconds, and recent full updates took approximately 5–8
minutes with varying proposal workloads. Supervisor heartbeat is
not evidence of learner progress; inspect inference audit timestamps and learning
activation when diagnosing apparent stalls.

## Frozen original pilot protocol

- Zero-shot dataset revision `784021482c3f320c6619ed4b3bb3b41a21424fcb`, SHA256
  `ba4f77f19517610a707c374e99322d7750c30fc4ae7ff5527888595a1e65d36d`.
  The loader rejects any other bytes before reading the upstream pickle.
- Seed 20261013; 1,000 unique prompt texts balanced across ARC, GSM8K,
  HellaSwag, MBPP, MMLU and WinoGrande. 800 learning / 200 held out.
  Prompts are deduplicated before splitting; outcomes never enter model state.
- All eleven recorded models are actions. Rune receives model identifiers as
  descriptions and an explicit quality/cost objective. It gets no per-example
  outcome, response, oracle label, or benchmark family label.
- Utility = recorded quality minus 10 times recorded USD cost. This is one
  declared operating point, not a tuned cost frontier. Scores need not be binary.
- Nudge receives only the action it sampled and that action's quality/cost.
  Learn after every 200 observations; one discovery round, up to two proposals,
  four retained questions, 32 recent training examples per proposer call.
  Existing Nudge selection and paraphrase checks remain in effect.
- Unit weights: the shuffled source does not supply real decision chronology.
  Synthetic timestamps advance hourly, but this run makes no decay/drift claim.
- Freeze the learned snapshot for all 200 held-out prompts; no test feedback.
  Score both Rune and Nudge probability distributions against the full outcome
  table in the evaluator. This removes extra action-sampling noise from reported
  quality/cost/regret without exposing counterfactual feedback to the learner.
- Report paired prompt-bootstrap intervals, task breakdowns, recorded-outcome
  hindsight regret and fixed models. A single training seed does not measure
  variability across learning runs. Current LLM benchmark contamination remains
  possible; public historical tasks are not proof of fresh production quality.

## Partial run evidence (2026-10-13 local date)

The run is **incomplete**: 800 training prompts and 194 of 200 frozen test
prompts completed. Rune inference then failed with HTTP 502 and a provider
network/timeout error. A deterministic replay from the shared cache failed on
the same uncached request again. No prompts were removed or replaced.

| Policy | Quality on 194 completed test prompts | Historical USD / 1,000 prompts |
| --- | ---: | ---: |
| Rune probability sampling | 58.63% | 0.485 |
| Nudge probability sampling | 59.02% | 0.496 |

The observed quality gain is only 0.39 percentage points, with slightly higher
recorded cost. This is descriptive partial evidence, not a completed pilot or
a demonstrated large improvement. Outcomes for the six missing prompts may
change the comparison. The two retained questions are code generation and
multiple choice; six additions failed the minimum prediction-improvement test.
No retained question was removed.

On the completed 800 training prompts, Rune sampling scored 61.47%, Rune's
highest-probability action scored 65.41%, and always selecting GPT-3.5 Turbo
scored 66.44%. Their recorded costs per 1,000 prompts were 0.476, 0.400, and
0.227 USD respectively. Thus there is headroom beyond the sampling effect;
these training diagnostics are not held-out generalization evidence.

The audit contains 6,313 successful Rune calls, two failed logical calls
(each has bounded retries), and five successful GLM calls including the probe.
GLM reported 10,678 reasoning tokens and approximately 0.097 USD of inference
under the explicit pricing assumptions below. Rune GPU cost remains unattributed.
Local artifacts are `benchmark-results/routerbench/pilot/`, `pilot-retry/`,
the shared `cache/`, and `partial-status.json`. Preserve them. Once Rune
inference is available, replay into a new run directory with the same prepared
data/configuration/cache to complete the missing requests and final report.

## Offline diagnosis of the partial run

Run the reusable diagnostic on a stopped run (no open SQLite WAL):

```sh
uv run python -m nudge.routerbench_diagnose \
  --prepared benchmark-results/routerbench/prepared.json \
  --database benchmark-results/routerbench/pilot/policy.db \
  --out benchmark-results/routerbench/diagnosis.json
```

This makes no model calls or database writes. It verifies that refitting the
saved training feedback reproduces the saved critic. Full-information refits
use only the 800 training prompts' outcomes. All variants are evaluated on the
same 194 completed test prompts. These are post-hoc diagnostic comparisons, not
new independent experiments; they do not simulate different online exploration
or question discovery trajectories.

| Diagnostic policy | Test quality | Gain over Rune sampling |
| --- | ---: | ---: |
| Rune probability sampling | 58.63% | — |
| Logged Nudge with posterior sampling | 59.02% | +0.39 pp |
| Chosen-feedback refit, posterior mean | 59.12% | +0.49 pp |
| Chosen-feedback refit, no questions, posterior mean | 59.09% | +0.46 pp |
| All 8,800 training-model outcomes, same questions/scale/tilt, posterior mean | 59.26% | +0.63 pp |
| Perfect recorded test rewards through the same tilt (oracle) | 65.05% | +6.42 pp |
| Best permitted clipped-score tilt with test outcomes (oracle) | 72.46% | +13.83 pp |
| Rune highest-probability model | 60.95% | +2.33 pp |
| Always GPT-3.5 Turbo | 61.98% | +3.36 pp |
| Always GPT-4 | 79.64% | +21.01 pp |

The oracles deliberately use test outcomes and are unattainable diagnostics,
not trained policies. Fixed-model costs differ: GPT-3.5 costs 0.233 USD per
1,000 prompts, GPT-4 3.747 USD, and Rune sampling 0.485 USD. Even after the
declared `10 * USD` penalty, fixed GPT-4 utility is 0.7589 versus Rune 0.5814.
The pilot baseline was therefore far from a strong policy for this operating
point; model identifiers alone were its capability/cost descriptions.

Measured implications:

- Logged Nudge moved only 2.87% of probability mass per prompt on average.
  GPT-4's mean probability moved from 5.68% to 6.09%, while GPT-3.5 moved from
  48.85% to 50.53%. The adjustment left most baseline behavior intact.
- Feedback was uneven: GPT-3.5 had 389 observations, GPT-4 47, Claude v2 15,
  and Yi none. Separate per-action regressions do not share evidence. The
  baseline-mixture exploration floor does not provide uniform action coverage.
- Nevertheless, exposing all training outcomes improved the fixed-feature
  posterior-mean policy by only 0.14 pp. Feedback scarcity alone does not explain
  the weak result under the current representation and adjustment rule.
- Removing and refitting without both learned questions reduced posterior-mean
  quality by only 0.028 pp. Most measured gain here came from the intercept and
  baseline-probability features, not language question discovery. Binary code
  and multiple-choice questions did little to identify which model would win.
- Perfect reward scores improve quality by 6.42 pp under the same rule. Thus
  the allowed policy changes leave substantial room for better scores. The
  conservative rule also limits performance: even its hindsight best tilt
  falls below always-GPT-4 quality and utility at the selected cost penalty.
- Posterior sampling cost about 0.10 pp relative to using the saved posterior
  mean in this realized run. This is secondary to the representation limitation.
- Admission optimizes selected-action reward MSE rather than policy utility;
  its two accepted features did not translate into much test improvement.
  This is evidence of a weak proxy in this run, not proof it always fails.

The paired prompt-bootstrap interval for logged Nudge's quality gain is
approximately +0.19 to +0.57 pp. It describes prompt uncertainty conditional on
one fitted run and the completed subset; it does not cover training-seed
variation or remove the six missing test prompts. The result is a small positive
change, not evidence of a large gain. New questions, exploration settings, or
stronger policy updates should be tested on a fresh split or declared as further
exploratory analyses; do not tune repeatedly against these 194 test outcomes.

## Question selection

GLM proposes additions, revisions, or removals; it does not decide which edits
become active. Nudge assigns about 25% of learning decisions to an internal
selection split by a stable hash of decision IDs. It fits current and candidate
reward predictors on the other rows and compares their selected-action reward
MSE on those selection rows. An edit is accepted only when
`(current_mse - candidate_mse) / reward_scale**2 > 0.001`.
An accepted removal therefore needs to improve prediction without the question.
A rejected addition never becomes part of the retained question set.

Additions and revisions must also have normalized feature standard deviation
above 0.03 and mean absolute paraphrase disagreement at most 0.15 on 16 training
examples. Accepted edits are evaluated sequentially, with at most four retained
questions. These are exploratory heuristics: lower prediction error does not
certify lower routing regret, and repeated use of the selection split can
overfit it. The final 200 test prompts are separate and never used for selection.

## Inference path

Application -> local TensorZero `/inference` -> local Rune provider bridge ->
the operator-selected native Baseten Rune endpoint. The bridge translates wire
formats; all action and feature probabilities come from Rune's Choice heads.
The baseline request is separate from feature requests and cached by complete
input, so adding features cannot change the baseline distribution.

The pilot used a dedicated Baseten model in environment `rune-dst-v1`, passed
as `--rune-endpoint`. The `--rune-revision` value comes from the operator's
committed deployment contract; native responses identify `/tmp/bei-model` and
do not independently attest the running weight revision.

The proposer uses `zai-org/GLM-5.2` through TensorZero with
`chat_template_kwargs.thinking=true`, 16,384 output tokens, temperature 0.1.
The real probe returned reasoning-token usage as well as two valid proposals.
Reasoning text is not saved; usage and its presence/length are recorded.

Secrets are read into memory from explicit Secret Manager project/reference
flags, then passed as TensorZero dynamic credentials. No secret files or
environment-variable configuration are introduced. Provider bodies containing
credentials are never persisted. The native bridge listens only on loopback;
Docker Desktop reaches it through `host.docker.internal`.

## Expanded discovery defaults

Following review of the original pilot, new runs use 256 newest training
examples per proposer call, up to 32 retained questions, three discovery rounds
per update, and up to 16 proposals per round. These replace the original
32-example/four-question/one-round/two-proposal pilot limits; the historical
results above still describe the original settings. Available training rows may
be fewer than 256, and selection rows are never supplied to the proposer.

The CLI exposes `--proposer-examples`, `--max-questions`, `--max-rounds`, and
`--max-proposals` (1–16). Logical learning calls are bounded by
`--max-learning-calls` (100,000 per update), and uncached gateway calls by
`--max-inference-calls` (200,000 per run). These larger ceilings accommodate
candidate feature backfills; they are not expected usage estimates. Discovery
can stop earlier when no proposal is accepted. The resolved settings are saved
in the manifest, and proposer/cache identities distinguish this revision.

No expanded live run has been performed. Use a new policy run directory for
the expanded experiment. Preserve the original artifacts for comparison; their
cache identity belongs to the old protocol. For other existing library stores,
reopen with the exact `Expert` settings from their stored snapshot; changing
expert settings requires a new store/choice rather than silently mutating an
existing policy. The supplied gateway schema must be reloaded to allow 16
proposals before using the new proposer.

## Running

From the repository root, install optional ingestion dependencies with
`uv sync --extra routerbench`.
Download the pinned `routerbench_0shot.pkl` from the
[public dataset](https://huggingface.co/datasets/withmartian/routerbench/tree/784021482c3f320c6619ed4b3bb3b41a21424fcb)
into `benchmark-results/routerbench/`.

Start the provider bridge with `python -m nudge.rune_gateway --endpoint=URL`
using the package environment. Start an isolated local TensorZero container
with `gateway/routerbench.toml`, mounting the gateway directory at `/app/config`
and publishing container port 3000 to loopback port 3118. The bridge uses port
3119. No production gateway config or endpoint lifecycle changes are needed.

Use `python -m nudge.routerbench --help` for the three commands:

- `prepare` validates the source checksum and writes the frozen split.
- `probe` verifies real Rune Choice and GLM proposal responses.
- `run` evaluates the frozen split and writes the policy log and metrics.

Required flags are `--prepared`, `--run-dir`, `--cache-dir`, `--gateway-config`;
`prepare` additionally needs `--dataset`. Credentials accept independent
`--rune-secret-project`, `--rune-secret`, `--glm-secret-project`, `--glm-secret`
flags; all four are required. The pilot used one shared Baseten API key secret
for both providers.

Successful inference responses are cached in SQLite by configuration identity,
function, system prompt and full input. Failed calls remain in the JSONL audit
and are never cached as success. A rerun uses a **new policy run directory** and
can reuse the inference cache to deterministically replay completed work.
Native concurrency defaults to two; set runner `--inference-concurrency` and
bridge `--max-concurrency` together to increase it (1–128). The runner prefetches
each learning block's states using only the already fitted questions, and then
processes decisions and feedback sequentially. Test prefetch occurs only after
the final fit. This preserves the seeded decision sequence while parallelizing
independent feature inference. Concurrency is recorded as execution provenance
and candidate paraphrases are prefetched on the same newest training examples
used by the core stability check. These speculative measurements do not change
selection or expose test outcomes. Execution settings are recorded
without invalidating model-input caches. The gateway bounds calls and timeouts. Errors stop
the run rather than silently dropping examples or inventing fallback outcomes.

`python -m nudge.routerbench_report --run-dir=DIR --cache-dir=CACHE` reads live
progress without invoking models. After completion, add `--prepared=FILE` for
a context-free zero-router comparator trained with the full training outcome
table (explicitly a stronger information setting than Nudge), `--out=FILE` for
the combined JSON report, and `--plot=FILE.png` for regret and quality figures.
Plotting requires Matplotlib, e.g. `uv run --with matplotlib ...`. Optional
`--glm-*-per-million` flags override the report's explicit price assumptions.

## Artifacts and costs

### Completed expanded run (2026-10-13 local date)

`expanded-recovery3/summary.json` and `expanded-report.json` contain the completed
800-training / 200-test run. Recovery reused the shared `expanded-cache` without
dropping prompts. Earlier expanded attempts stopped on gateway timeouts; those
failures do not imply the Baseten deployment was inactive. Restarting the local
gateway and replaying cached inputs ultimately completed the protocol.

| Policy | Test quality | Historical USD / 1,000 prompts | Mean utility regret |
| --- | ---: | ---: | ---: |
| Rune probability sampling | 58.455% | 0.49049 | 0.32807 |
| Nudge probability sampling | 59.103% | 0.50447 | 0.32173 |

Quality improved by 0.648 percentage points (paired prompt-bootstrap 95% interval
0.410–0.887 pp); mean utility improved by 0.006338. This remains a modest gain,
with higher historical downstream cost. The interval is conditional on one
learned run. This reused split is exploratory, not a fresh confirmation set, and
its 200-prompt result is not directly comparable to the original partial 194.

Five questions survived: domain specialization, expected answer brevity,
Winograd-style format, factual recall, and how-to continuation. The expanded
limits allowed 32 questions, 256 proposer examples, 16 proposals per round and
three rounds; the existing selection checks retained only five.

The library now supports categorical questions as soft probability vectors. A
task-identity question can use mutually exclusive labels such as coding,
quantitative reasoning, multiple-choice knowledge, passage continuation, and
commonsense reasoning without forcing the learner to treat labels as an
ordinal scale. The completed live run predates this capability and did not use it.

### Categorical-question smoke pilot

After adding categorical support, a paid 100-prompt smoke pilot used 80 training
and 20 frozen test prompts from the same prepared source. The ordinary GLM
proposer proposed and retained three categorical questions: `prompt_type`,
`output_format_type`, and `domain_knowledge`. It also retained six binary or
ordinal context questions. This confirms the intended task-classification heads
are discoverable and persist correctly.

On the 20-prompt test slice, Rune scored 59.988% quality and Nudge scored
61.256% (+1.268 pp); utility improved by 0.012691 with essentially unchanged
recorded cost. The paired bootstrap interval was +0.196–2.504 pp, but this is a
smoke test with n=20 and the slice is exploratory. It is evidence that the
categorical path works, not a claim of production-sized improvement. The run had
3,064 successful Rune calls and no inference errors; observed GLM overhead was
estimated at USD 0.120 under the documented price assumptions. The deployment
was restored to its original 1/2 scaling and deactivated after the run.

Shared expanded-cache evidence records 32,318 logical Rune calls (12 errors)
and eight successful GLM calls, including interrupted attempts/probes. Observed
GLM usage estimates USD 0.663 under the assumptions below. Dedicated H100 cost
is additional and unattributed; interrupted in-flight requests may be absent
from the audit. Question batching was not used in this run.

### Offline predictor comparison

The completed run's cached feature answers and policy database support a
predictor-only comparison without additional Rune or GLM calls:

```sh
uv run python -m nudge.routerbench_predictors \
  --prepared benchmark-results/routerbench/prepared.json \
  --database benchmark-results/routerbench/expanded-recovery3/policy.db \
  --out benchmark-results/routerbench/predictor-comparison.json
```

The comparison selects regularization and model family by three-fold training
cross-validation on observed chosen-action rewards, then applies the same
bounded tilt used by Nudge. It compares the original per-action linear model
with tuned linear, quadratic interaction, shared quadratic, and shared RBF
predictors. Shared RBF was selected by training cross-validation. On the
already inspected 200-prompt test split, posterior means produced:

| Predictor | Test quality | Test utility | Mean probability mass moved |
| --- | ---: | ---: | ---: |
| Original linear | 59.028% | 0.585238 | 3.40% |
| Tuned linear | 59.028% | 0.585238 | 3.40% |
| Quadratic | 58.971% | 0.584688 | 3.18% |
| Shared quadratic | 58.976% | 0.584731 | 3.26% |
| Shared RBF (training-selected) | 59.066% | 0.585582 | 3.69% |

The nonlinear predictor improves the posterior-mean result by only 0.038 pp
over the original predictor. This is exploratory because the test split was
already inspected for the pilot, and the question representation was discovered
before this cross-validation. The comparison does not justify activating the
RBF predictor; it shows that predictor capacity alone is not the main blocker.

### Oracle feature audit

#### Full 1,000-prompt factor reconstruction (2026-10-14)

The user requested discriminating factors for the entire existing sample and
its oracle, explicitly including all 800 training and 200 test prompts.
`nudge.routerbench_factors` implements that separate full-sample diagnostic.
It fits routing cells on all 1,000 rows and evaluates those same rows; the
historical training/test split is not used as a holdout in this analysis.

Each selected Boolean factor splits the existing cells. Each resulting cell
routes to the model with greatest aggregate `quality - 10 * cost`. Greedy
forward selection maximizes that routing objective, allows zero-gain splits
to resolve interactions, and stops when the oracle is reached or no candidate
can separate unresolved rows. Backward pruning removes factors that do not
change attained utility. This is not a proof of a globally smallest set.

The input to factor extraction is prompt text only. IDs, family labels,
outcomes, and generated model answers are excluded. Math worked-example
preambles and the common answer-format suffix are stripped before keyword,
token, length, and number features are measured. Structural format detectors
can inspect the full prompt. Topic labels denote explicit regex word groups,
not model-produced semantic judgments.

| Factor policy | Factors after pruning | Routing cells | Quality | Total historical USD for 1,000 | Mean utility |
| --- | ---: | ---: | ---: | ---: | ---: |
| Best fixed model, GPT-4 | 0 | 1 | 80.25% | 3.770440 | 0.764796 |
| Topic/structure/length/number indicators | 37 | 746 | 89.40% | 0.359954 | 0.890400 |
| First lexical combination reaching quality ceiling | 24 | 847 | 91.80% | 0.307941 | 0.914921 |
| Full lexical combination matching quality and cost oracle | 40 | 987 | 91.80% | 0.230651 | 0.915693 |

The exact-oracle set has 15 grouped or structural factors: algebra/arithmetic;
pronouns; length at least 100 characters; household/food; causal words;
negation/exception; arrays/collections; earth/astronomy; comparison/superlative;
sorting/searching; length at least 200 characters; time/rates; medicine/biology;
question mark; chemistry. Its other 25 factors are presence of the tokens
`b`, `are`, `in`, `and`, `to`, `of`, `is`, `the`, `on`, `find`, `list`, `one`,
`with`, `given`, `into`, `for`, `by`, `difference`, `two`, `nth`, `so`, `from`,
`race`, `hexagonal`, and `did`.

The exact-oracle policy contains 976 single-prompt cells, nine two-prompt cells,
and two three-prompt cells. This is an exact sample reconstruction, not a
discovered compact semantic explanation of model capability. The 37-factor
topic/structure result also fits this same sample. Neither result measures
prediction on unseen prompts. No live Nudge policy was modified.

Reproduce from the repository root:

```sh
uv run python -m nudge.routerbench_factors \
  --prepared benchmark-results/routerbench/prepared.json \
  --out benchmark-results/routerbench/oracle-factors-lexical.json \
  --lexical
```

Omit `--lexical` for the structural-only candidate set, or add
`--max-factors 24` to reproduce the first quality-ceiling combination. The
JSON artifacts preserve the exact selected factors, bit-pattern-to-model map,
cell counts, full selection trajectory, and prepared-data fingerprint
`81e39768f364f05e45b5d9c4fed1d24c6bfeada4d1301544d6e64ed0db251b7e`.
A local-artifact regression test replays the saved exact-oracle policy using
only prompt-derived bits for model selection and verifies all three oracle
metrics independently. These commands make no inference calls.

#### Semantic factor analysis

Current rubric revision: the original `option_distinction` has been replaced
with two fields, bringing the rubric to 17 dimensions:

- `has_answer_options`: "Does the actual question explicitly provide candidate
  answers for the respondent to choose from? Ignore worked examples." Labels
  are `no` / `yes`; producing a number or code without supplied candidates is `no`.
- `option_difficulty`: only for supplied choices, distinguish `obvious`,
  `conceptual`, and `subtle` alternatives. Rune asks this head only after the
  presence head selects `yes`; it does not offer an absence category in that
  call. Otherwise the stored difficulty is deterministically `not_applicable`.

GLM's fixed response schema includes `not_applicable`, with cross-field
validation enforcing its use exactly when presence is `no`. Both classifiers
use the same rubric definitions. The generated schema is updated. This change
has unit/transport coverage but has not been rerun against live models.

Migration: preserve the historical 16-factor artifacts below. Rerun annotations,
Rune measurements, and policy fitting into a new directory such as
`benchmark-results/routerbench/semantic-v2/`, with new `--out`, `--cache-dir`,
`--reference`, `--annotations`, `--measurements`, and `--semantic-report` paths
as appropriate. Reload the local gateway's updated schema before GLM annotation.
Rubric fingerprints reject old annotations, measurements, or fitted policies;
do not rename old labels or reuse old tree feature indices. Current Rune reports
use `factor_count` and `all_labels_match_prompts` instead of a fixed 16-label key.
The measurements and scores below describe the original rubric.

`nudge.routerbench_semantic` replaces literal word-presence checks with a fixed
16-dimension semantic rubric. GLM-5.2 thinking annotates prompt meaning through
the isolated `gateway/semantic.toml` TensorZero function. The rubric covers task,
subject, knowledge depth, reasoning steps, abstraction, choice subtlety,
quantitative operation, code structure, algorithm, constraint load, edge cases,
fact integration, reference tracking, causal reasoning, temporal order, and
required precision. Labels are categorical judgments, not regex matches.

The classifier receives only prompt strings and the rubric, never outcomes,
model names, family metadata, or sample IDs. It ignores worked examples when
judging the actual question. Batches contain eight prompts; responses must
contain all fields and exactly eight annotations in input order (or the size
of the final batch). Prepared-data, rubric, prompt, schema, and gateway config
identities protect alignment/cache reuse. Persisted caches support restart;
malformed annotations receive at most two validation retries with distinct
cached request identities. Every validation failure is recorded in the annotation
artifact; an exhausted retry stops analysis instead of silently omitting prompts.

Generate the strict gateway schema from the rubric with:

```sh
uv run python -m nudge.routerbench_semantic schema \
  --out gateway/schemas/semantic.json
```

Run a separate local TensorZero container with that configuration, mounting the
gateway directory at `/app/config`, exposing loopback port 3120 to container
3000. No Rune bridge or dedicated deployment activation is needed. Then:

```sh
uv run python -m nudge.routerbench_semantic annotate \
  --prepared benchmark-results/routerbench/prepared.json \
  --out benchmark-results/routerbench/semantic-annotations.json \
  --gateway-config gateway/semantic.toml \
  --cache-dir benchmark-results/routerbench/semantic-cache \
  --credential-source application-default

uv run python -m nudge.routerbench_semantic analyze \
  --prepared benchmark-results/routerbench/prepared.json \
  --annotations benchmark-results/routerbench/semantic-annotations.json \
  --out benchmark-results/routerbench/semantic-analysis.json
```

Secret references require `--secret-project` and `--secret`. Values stay in
memory.
Concurrency defaults to eight, bounded to 32; batch size defaults to eight,
bounded to 16; the logical uncached call ceiling defaults to 150. Each request
uses the existing bounded five-attempt gateway transport with a 300-second
timeout. Stop the owned local gateway when finished. Analysis itself is offline.

Analysis measures each dimension alone, full-sample combinations selected by
utility, and interpretable decision trees whose leaves contain at least 20
prompts. Trees have caps of 4, 8, 16, and 32 leaves. The saved report includes
every routing condition, support count and model assignment, and all-model
quality/cost/utility breakdowns for each semantic value. Five-fold out-of-fold
tree routing is reported separately as a check on whether model preferences
transfer across these rows. It is not the earlier untouched 200-prompt test.
Unrestricted combinations still report singleton counts; semantic labels do
not by themselves prevent sample memorization.

Completed on 2026-10-14: 1,000/1,000 annotated using 127 serverless GLM calls,
zero transport errors, and two annotation-count validation retries for the last
batch (seven returned for eight requested). The successful retry supplied all
eight labels. The first probe's eight annotations were reused, and the recovery
reused the other completed cached annotations. Observed usage: 267,312 input
tokens (116,928 reported cached) and 279,982 output tokens. The local semantic
gateway was stopped after completion; Rune was not activated for this analysis.

| First semantic dimensions selected | Cells | Singleton cells | Quality | Mean utility | Total historical downstream USD |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 128 | 42 | 82.550% | 0.813395 | 1.210457 |
| 8 | 280 | 146 | 85.325% | 0.844589 | 0.866060 |
| 12 | 392 | 231 | 87.325% | 0.865392 | 0.785838 |
| 16 | 516 | 350 | 88.275% | 0.876347 | 0.640328 |
| Unrestricted per-prompt oracle | — | — | 91.800% | 0.915693 | 0.230651 |

Greedy semantic selection order: (1) offered-answer distinction, (2) task type,
(3) subject, (4) edge-case sensitivity, (5) causal/physical reasoning,
(6) integration of multiple facts, (7) algorithmic requirement,
(8) dependent reasoning steps, (9) code data structure, (10) reference tracking,
(11) quantitative operation, (12) temporal ordering, (13) constraint load,
(14) abstraction, (15) required precision, (16) knowledge depth.
This order measures incremental improvement within this vocabulary; it does
not establish that these are the globally best semantic dimensions.

The best full-sample supported tree stops at nine leaves using seven factors:
offered-answer distinction, edge-case sensitivity, quantitative operation,
constraint load, reasoning steps, subject, and code structure. All leaves have
at least 22 prompts. Its quality is 81.05%, utility 0.791792478, and total
downstream cost USD 1.8707522. Five-fold out-of-fold quality/utility are
78.925% / 0.772484208; fixed GPT-4 is 80.25% / 0.7647956 on the full sample.
The 4-leaf tree has the best out-of-fold utility among the tested caps:
0.77441901 (78.65% quality). Report this separately from the same-sample fit.

Concrete differences in these recorded outcomes:

- String-processing code: GPT-4 wins utility on 32 prompts (0.63825 vs GPT-3.5
  0.590518). Numeric-scalar code (55 prompts) and sequence code (67) favor
  GPT-3.5 (0.742223 and 0.623372, respectively).
- Multiplicative problems (107 prompts) favor Claude Instant (0.590351).
  Proportional problems (49) favor Claude v1 (0.634911). Formal math (39)
  favors GPT-4 (0.721246).
- GPT-4 remains the utility winner in each offered-choice subtlety category,
  so that factor's standalone gain mainly distinguishes offered-choice tasks
  from open responses. It does not show subtlety itself changes the winner.
- Knowledge depth, fact integration, reference tracking, causality and temporal
  order each leave GPT-4 as the best model in every one-dimensional category.
  Their additional sample-fit value arises through interactions; do not describe
  them as independently demonstrated model switches.

The 16-factor score still uses 516 model-assignment cells, including 350
singletons. It is semantic sample reconstruction, not evidence that a compact
semantic router reaches that score on unseen prompts. The gap to the oracle
remains 3.525 quality points and 0.03934677 utility. The narrower supported
tree reveals much less transferable routing gain than unrestricted grouping.

Artifacts: `semantic-annotations.json` stores all labels and retry evidence;
`semantic-analysis.json` stores full model breakdowns, the selection trajectory,
supported tree rules, cross-validation, and all 516 semantic-combination model
assignments. The real-artifact regression test replays those assignments from
labels alone and independently verifies all reported metrics. No labels or
policies were activated in the Nudge learner.

#### Rune answerability of the semantic questions

On 2026-10-14, `nudge.routerbench_rune_semantic` measured the same 16 semantic
dimensions with native Rune Choice heads through TensorZero. All 1,000 prompts
completed (16,000 labels). The cache contains 1,128 real requests: 1,000
16-head requests, including the eight-prompt probe, plus 128 single-head probe
requests. There were zero failed logical inference calls. Successful transport
retries, if any, are inside the bounded HTTP layer and not separately counted.

The eight-prompt probe gave identical top categories for all 128 comparisons
between single-head and 16-head calls. The largest probability difference was
0.0926666, so this is evidence for category agreement on the probe, not identical
soft features in the existing single-question Nudge loop. The full diagnostic
uses 16 heads per call, concurrency 32, rubric definitions in the instructions,
and human-readable category names as Choice criteria. No GLM labels or model
outcomes were sent to Rune.

GLM labels are a reference annotator, not independently verified truth. Mean
agreement is 70.84375%; only two prompts match all 16 reference labels.

| Factor | Agreement | Majority-reference baseline |
| --- | ---: | ---: |
| Task type | 74.4% | 20.1% |
| Subject | 79.9% | 23.2% |
| Knowledge depth | 81.6% | 53.0% |
| Reasoning steps | 55.0% | 65.6% |
| Abstraction | 61.8% | 52.3% |
| Offered-answer distinction | 27.1% | 33.4% |
| Quantitative operation | 87.1% | 76.7% |
| Code structure | 80.6% | 83.0% |
| Algorithmic requirement | 80.8% | 83.0% |
| Constraint load | 53.0% | 84.1% |
| Edge-case sensitivity | 73.5% | 76.1% |
| Fact integration | 69.0% | 52.7% |
| Reference tracking | 90.6% | 76.8% |
| Causal/physical model | 71.8% | 78.9% |
| Temporal ordering | 73.2% | 84.7% |
| Required precision | 74.1% | 56.8% |

Applicable-subset agreement avoids letting the `not_code`/`none` labels dominate:
code structure 87.06% and algorithmic requirement 75.88% on 170 GLM-coded tasks;
quantitative operation 63.95% on 233 GLM-quantitative tasks. On offered-choice
prompts alone, option distinction is 39.79% (666 rows).

The largest concrete failure is the option question: Rune recognizes only six
of 334 `no_options` prompts, calling 283 of them `obvious` instead. This
explains why the GLM-derived supported tree does not transfer unchanged.

| Supported routing comparison | Quality | Mean utility | Historical USD / 1,000 |
| --- | ---: | ---: | ---: |
| Nine-rule GLM-label fit | 81.050% | 0.791792 | 1.870752 |
| Same frozen rules, Rune labels | 80.225% | 0.764927 | 3.732292 |
| Best supported sample-fit tree refitted on Rune labels | 80.750% | 0.791018 | 1.648155 |

The refitted Rune 4-leaf tree has out-of-fold utility 0.773656 (79.0% quality);
larger tested caps yield lower out-of-fold utility. Fitting all 16 hard-label
combinations reaches 89.95% quality / 0.895224 utility but creates 773 cells,
647 singletons. Do not interpret that larger sample reconstruction score as
better semantic accuracy or transferable routing. The immediate implication is
to refine/gate the weak questions and fit on Rune's own measurements, rather
than copying a policy fitted on GLM labels. Soft-probability learning remains
a separate experiment.

Run against the existing local Rune gateway and bridge:

```sh
uv run python -m nudge.routerbench_rune_semantic measure \
  --prepared benchmark-results/routerbench/prepared.json \
  --out benchmark-results/routerbench/rune-semantic-measurements.json \
  --gateway-config gateway/routerbench.toml \
  --cache-dir benchmark-results/routerbench/rune-semantic-cache

uv run python -m nudge.routerbench_rune_semantic report \
  --prepared benchmark-results/routerbench/prepared.json \
  --reference benchmark-results/routerbench/semantic-annotations.json \
  --measurements benchmark-results/routerbench/rune-semantic-measurements.json \
  --semantic-report benchmark-results/routerbench/semantic-analysis.json \
  --out benchmark-results/routerbench/rune-semantic-report.json
```

`--heads-per-call 1` measures the existing single-head style;
`--limit 8` reproduces the probe. Secrets default to the previous experiment's
Secret Manager reference, read with application-default authentication and
kept in memory. The report contains full confusion matrices, per-class recall,
reference probabilities, applicable-subset metrics, frozen-policy transfer and
refitted analyses. No semantic labels or policies are activated in Nudge.

Live state was captured before changing scaling: ACTIVE, two replicas, min/max
2/2, unlike the earlier inactive 1/2 log. The authorized experiment requested
4/4; observations during and after inference showed three active replicas.
Cleanup restores the observed 2/2 settings and leaves the deployment ACTIVE;
local owned bridge/gateway processes are stopped. Retain the before/running/
afterrun/restored management responses beside the report as operation evidence.

#### Earlier family and marker comparison

`nudge.routerbench_feature_search` is a separate reverse-engineering diagnostic.
It groups the full recorded training outcomes by prompt metadata or simple
prompt-observable markers, chooses one historical model per group, and evaluates
that mapping on the test prompts. The family result is an oracle upper-bound
diagnostic because `family` comes from the dataset manifest and was withheld
from Nudge. It must not be wired into a production router as if it were prompt
input. The prompt-marker results are exploratory and use the full training
outcome matrix, so they are not comparable to Nudge's chosen-action feedback.

The family mapping found on training data was: `mmlu → GPT-4`,
`arc-challenge → GPT-4`, `grade-school-math → Claude Instant`,
`winogrande → GPT-4`, `mbpp → GPT-3.5`, and `hellaswag → GPT-4`.

This explains why the oracle diagnostic looked dramatically better: it mostly
selects a strong fixed model using leaked benchmark task identity. It does not
show that the five learned Nudge questions can recover that identity.

`manifest.json` pins configuration/data identity; `decisions.jsonl` contains
per-prompt distributions and evaluator metrics; `learning.json` contains
proposals/selection results; `policy.db` preserves chosen feedback and snapshots;
`summary.json` contains the completed comparison. Shared cache
`inferences.jsonl` records real request IDs, latency, token usage and failures.
Keep downloaded data and raw runtime artifacts out of Git. Publish only the
derived aggregate report and reproducible code.

Recorded downstream costs use RouterBench's historical pricing. GLM runtime
cost can be estimated using reported usage and explicit price assumptions:
Baseten lists $1.40 input, $0.14 cached input and $4.40 output per million tokens
for GLM-5.2 ([source](https://www.baseten.co/library/glm-52/), checked 2026-10-13).
Dedicated Rune GPU dollars cannot be inferred from token counts or request
latency: report them as unattributed unless a hardware billing/allocation model
is explicitly supplied. Never count them as zero or claim a measured end-to-end
dollar saving from the historical downstream table alone.

### Fresh 1,000-prompt categorical run

The completed fresh run used the pinned 800-train/200-test split, four H100:2
replicas during execution, 32 inference workers, one discovery round, and four
proposals per learning block. It retained two categorical questions:
`task_type` and `output_format`. All 1,000 prompts completed with zero inference
errors. Artifacts are under `benchmark-results/routerbench/categorical-1000-fresh2/`.

On the frozen 200-prompt test partition, Rune quality was 58.438% and Nudge
quality was 59.016%, a paired gain of **+0.578 percentage points** (bootstrap
95% interval +0.371–0.795 pp). Utility rose from 0.579479 to 0.585129
(+0.005650); recorded downstream cost rose by USD 0.0000131 per prompt. Nudge
test cumulative hindsight regret was 64.516 versus Rune's 65.646. The
supervised all-outcomes diagnostic reached 63.246% on the same test slice, so
this is an exploratory improvement with substantial headroom.

The live adapter now retries transient 408/429/5xx responses and network
timeouts five times with bounded exponential backoff. The first fresh attempt
was stopped after a transient 502; the completed run used the retrying adapter
and recorded zero errors. Baseten scaling was restored to 1/2 and the
deployment was deactivated after completion.

## Frozen 200-prompt semantic regressor measurement

Train on the existing 800 prompts using archived original 16-question Rune
probability vectors, plus the original per-model Rune baseline logit. Use the
existing per-action Gaussian linear critic (ridge alpha 16), training-only
reward scale, and posterior means. No tuning on test outcomes.

| Training feedback | Routing rule | Test utility | Test quality |
| --- | --- | ---: | ---: |
| Original Rune baseline | Original distribution | 0.579479 | 58.438% |
| Logged chosen model | Current bounded tilt | 0.585693 | 59.077% |
| Logged chosen model | Highest predicted utility | 0.669325 | 68.750% |
| All 11 training model outcomes | Current bounded tilt | 0.587040 | 59.213% |
| All 11 training model outcomes | Highest predicted utility | 0.778440 | 79.625% |

Utility is quality minus 10 times downstream model cost in USD. Full-outcome
training is a stronger information setting than Nudge's chosen-action feedback;
greedy routing removes the current bounded adjustment. These are offline fits,
not changes to the deployed learner. The revised 17-question rubric has not been
measured. Questions were designed after inspecting this sample, so this reused
split is exploratory even though test outcomes do not enter regression fitting.

Result artifact: `benchmark-results/routerbench/semantic-regressor-test200.json`,
including predictions, exact rubric, input fingerprints and paired bootstrap
metrics. Reproduce from repository root:

```sh
uv run python -m nudge.routerbench_semantic_regressor \
  --prepared benchmark-results/routerbench/prepared.json \
  --measurements benchmark-results/routerbench/rune-semantic-measurements.json \
  --semantic-report benchmark-results/routerbench/semantic-analysis.json \
  --decisions benchmark-results/routerbench/categorical-1000-fresh2/run/decisions.jsonl \
  --manifest benchmark-results/routerbench/categorical-1000-fresh2/run/manifest.json \
  --out benchmark-results/routerbench/semantic-regressor-test200.json
```

## Matched decision-utility discovery experiment

The utility selector comparison uses the same prepared 800/200 split, seed
20261013, quality-minus-10-times-cost objective, 200-feedback learning cadence,
32-question cap, and one round/four proposals per checkpoint as
`categorical-1000-fresh2/run`. It compares the whole current discovery loop
(replay, historical diagnostics, v4 proposer and utility acceptance), not the
selector in isolation. The final test split has already been inspected.

`--reference-run` validates the reference data fingerprint, objective, prompt IDs
and partitions, then freezes its Rune baseline distributions. Only new question
answers and proposals require live inference. Unknown prompts fail rather than
silently receiving a fresh baseline. Fresh runtime/caching artifacts are under
`benchmark-results/routerbench/utility-discovery-1000/`; `source.tar.gz` preserves
source/tests/gateway configuration used for reproduction.

```sh
uv run python -m nudge.routerbench run \
  --prepared benchmark-results/routerbench/prepared.json \
  --run-dir benchmark-results/routerbench/utility-discovery-1000/run \
  --cache-dir benchmark-results/routerbench/utility-discovery-1000/cache \
  --gateway-config gateway/routerbench.toml \
  --reference-run benchmark-results/routerbench/categorical-1000-fresh2/run \
  --max-rounds 1 --max-proposals 4 --inference-concurrency 16 \
  --credential-source application-default \
  --rune-endpoint <rune-endpoint> --rune-revision <revision> \
  --rune-secret-project <secret-project> --rune-secret <secret-ref> \
  --glm-secret-project <secret-project> --glm-secret <secret-ref>
```

After completion, `nudge.routerbench_report --reference-run=REFERENCE` adds a
paired current-versus-previous test comparison. It refuses different data,
training settings, prompt order or baseline vectors. Reported intervals describe
prompt uncertainty at this seed, not training-seed variation.

### Measured result: utility discovery, matched budget

Completed 800 training and 200 test prompts in 1,017 seconds. GLM made four
proposal calls; Rune made 7,256 new feature/paraphrase calls. No inference errors.
Baseline vectors match the reference within 1.11e-16 from normalizing floating-point
roundoff (comparison tolerance 1e-12). No shared deployment/scaling state changed;
the owned local bridge and gateway were stopped after completion.

| Policy | Test utility | Test quality | Retained questions |
| --- | ---: | ---: | ---: |
| Rune baseline | 0.579479 | 58.438% | — |
| Previous categorical/MSE Nudge | 0.585129 | 59.016% | 2 |
| Current utility-discovery Nudge | 0.584653 | 58.965% | 0 |

Current-minus-previous utility is -0.000476, paired prompt-bootstrap 95% interval
[-0.001806, +0.000822]. There is no measured improvement. Twelve of sixteen
proposals failed the per-action coverage gate; four failed paraphrase stability.
None of the twelve utility-scored proposals passed the uncertainty-adjusted gain
threshold even if coverage were ignored. Removing the coverage gate alone would
therefore not have accepted these proposals under the current evaluator.

At the final update, selection action counts were 104 for GPT-3.5, 21 for Mistral,
19 for Claude Instant, 15 for Mixtral, 14 for GPT-4, and fewer than ten for every
other model, including zero for Yi. The gate treats any average probability
change above 1e-10 as requiring ten effective observations for that action, so
small normalization-induced changes can block an otherwise localized proposal.
This exposes a restrictive gate, but does not establish useful gain for the
rejected proposals. The earlier full-information semantic results are a different
information/policy setting. A test of known questions under this exact feedback
and serving policy would separate representation from evaluation limitations.

Full evidence: `utility-discovery-1000/report.json`, `run/learning.json`,
`run/decisions.jsonl`, `run/policy.db`, and inference audit/cache. The frozen
baseline and paired-comparison helpers passed 99 package tests; full build passed.

## Unrestricted policy and saved-question retention (2026-10-14)

The runtime now selects from posterior action utilities with a uniform floor;
Jev probabilities are features only. No drift cap, score clipping, zero-probability
veto or delayed exploration remains. Serving and utility comparisons share the
same deterministic posterior ensemble. Bundle proposals and persistent pending
hypotheses are supported. See [design](design.md) for the new policy, coverage
diagnostics and required new choice/store migration.

Before another live discovery run, measured the original 16 saved Rune semantic
heads, individually and as one atomic bundle. Selection used the stable internal
train/selection split of the 800 training prompts. Questions were then frozen,
critics refitted on 800, and policies evaluated on the existing 200 test prompts.
The test split did not select questions or tune parameters. This is a saved-answer
utility diagnostic, not a new LLM discovery run; it does not fabricate or remeasure
paraphrase stability.

```sh
uv run python -m nudge.routerbench_retention \
  --prepared benchmark-results/routerbench/prepared.json \
  --measurements benchmark-results/routerbench/rune-semantic-measurements.json \
  --semantic-report benchmark-results/routerbench/semantic-analysis.json \
  --decisions benchmark-results/routerbench/categorical-1000-fresh2/run/decisions.jsonl \
  --out benchmark-results/routerbench/unrestricted-retention.json
```

| Training feedback | Activated questions | No-question / selected test utility | Fixed 16-question test utility |
| --- | --- | --- | --- |
| Original logged chosen outcomes | 0 | 0.651992 | 0.616917 |
| New exploration replay, one chosen outcome per prompt | 0 | 0.653526 | 0.611841 |

New exploration replay refits every 200 observations and uses no questions while
collecting feedback. It reveals only the selected model's outcome to fitting and
selection. Full outcome vectors stay in the replay/evaluation harness. Utility
is quality minus ten times dollar cost, as before.

All 17 original-feedback candidates remain pending for coverage. With exploration,
16 remain pending for uncertainty and one for coverage. For example `task_type`
has estimated selection gain +0.043919, with approximate interval
[-0.010640, +0.098478]. This is unresolved evidence, not established improvement.
Removing Jev's veto did not make the saved representation beneficial under this
posterior policy; forcing all questions reduced test utility. Earlier 0.669325
chosen-feedback results used greedy prediction, a different decision rule.
This run does not establish a ceiling or successful discovery. Synthetic tests
separately verify that clearly useful questions and bundles are retained, pending
records survive restart/history truncation, and test outcomes cannot influence
selection or predictions. No live inference was used.

## Separate question bank (2026-10-14)

Question retention no longer requires the routing regressor to demonstrate an
immediate policy gain. The LLM proposes bank additions/revisions/drops; the library
checks answer variation and stability, records action-by-answer outcome evidence,
and retains hypotheses. Subsequent decisions provide separate evidence. Default
`learn()` now trains the bank without discovery; `discover_questions()` leaves
the serving critic and features unchanged. The live replay runner explicitly
requests the combined operation. Upgrade with a new choice/store contract.

Saved-answer measurement:

```sh
uv run python -m nudge.routerbench_question_bank \
  --prepared benchmark-results/routerbench/prepared.json \
  --measurements benchmark-results/routerbench/rune-semantic-measurements.json \
  --semantic-report benchmark-results/routerbench/semantic-analysis.json \
  --decisions benchmark-results/routerbench/categorical-1000-fresh2/run/decisions.jsonl \
  --out benchmark-results/routerbench/question-bank-retention.json
```

All 16 saved questions vary enough to remain hypotheses in both feedback regimes.
All 800 chosen outcomes inform their descriptive evidence. No question was kept
because of regressor MSE or policy-value gain. This is a fixed-question test:
no new LLM proposals or live paraphrase checks were performed. Retaining all 16
is therefore a mechanism check, not a claim that all 16 are useful.

As an additional retrospective check, compare relative action-outcome patterns
in the first 400 and next 400 training prompts. With exploration feedback, eight
questions have positive mean cross-product of centered action advantages and
greater-than-half direction agreement: task type, subject, abstraction, option
distinction, quantitative operation, code structure, reference tracking and
precision requirement. Example direction agreements: reference tracking 80.0%
(20 comparable contrasts), precision requirement 73.1% (26), task type 72.4%
(29). Knowledge depth agrees only 23.3% (30); temporal order 20.0% (20).
Contrasts are correlated and some are sparse; these descriptive results should
guide LLM refinement, not certify usefulness. The questions were originally
designed after inspecting this dataset, so this is not fresh prospective validation.
Real prospective boundaries are implemented and tested separately in the library.

## Live loop debugging and matched comparisons (2026-10-14)

The v6 live debug run started with an empty bank and real GLM/Rune inference.
Its first proposal bundled task type, reasoning depth, code output and constrained
answer format; all four passed actual answer/paraphrase checks. This partial run
is preserved under `benchmark-results/routerbench/live-bank-debug-v6/` and was
intentionally interrupted before final evaluation to fix excessive proposer input.
It is not a completed routing result.

The first two GLM calls reported 254,496 and 297,466 input tokens. Inspection of
the first saved context found 607,814 characters, including 139,890 characters
of coefficient contributions, 97,158 of baseline probabilities, 96,409 of policy
probabilities and 60,800 of predicted scores; prompt text accounted for 111,381.
The question-bank implementation had passed entire decision records. v7 projects
the sampled records onto observed evidence: prompt, chosen action/outcome,
selection propensity, question answers, residual, weight and replay provenance.
It preserves sample membership and full prompts. This fixes input bloat, not a
demonstrated question-quality or routing improvement. The v7 run starts empty
with its own cache; no reference questions enter either proposer.

`nudge.routerbench_matched` compares reference, generated and no-question feature
sets using the same chosen-feedback trajectory, baseline vectors, reward scale
and Gaussian linear fit. It reports greedy and posterior routing separately.
Generated features must come from a completed live store aligned with the supplied
decision log; mismatched feedback, incomplete measurements and split overlap fail.
Test outcomes do not affect fitting or policy construction. Run after discovery:

```sh
uv run python -m nudge.routerbench_matched \
  --prepared benchmark-results/routerbench/prepared.json \
  --measurements benchmark-results/routerbench/rune-semantic-measurements.json \
  --semantic-report benchmark-results/routerbench/semantic-analysis.json \
  --decisions benchmark-results/routerbench/live-bank-debug-v8/run/decisions.jsonl \
  --database benchmark-results/routerbench/live-bank-debug-v8/run/policy.db \
  --all-training-outcomes \
  --out benchmark-results/routerbench/live-bank-debug-v8/matched.json
```

Omit `--database` for a reference/no-question comparison alone. On original
archived feedback, no questions score 0.676555 greedy utility versus 0.669325
with the reference 16; posterior utilities are 0.651992 and 0.616917 respectively.
Thus the earlier 0.669325 result alone did not establish improvement from these
questions in this regressor. This does not negate their sample-fit discrimination
under other predictors. The saved comparator is
`live-bank-debug-v6/reference-control-original-feedback.json`. All final-test
comparisons remain exploratory because this split has been examined previously.

Pass `--all-training-outcomes` to additionally produce a separately labeled
full-information diagnostic. It fits all eleven recorded outcomes for each
training prompt with the same scale, predictor and question representations.
Those extra outcomes never enter discovery. On the original trajectory, reference
questions improve this diagnostic's greedy utility from 0.742267 to 0.778440,
while posterior utility decreases from 0.725929 to 0.697653. This establishes
useful reference-feature signal under the stronger feedback setting and shows
why posterior routing alone is not a sufficient question-quality measurement.

The v7 live trace exposed a second problem in the first 200 training examples:
one unstable member rejected an entire bundle, then the proposer attempted to
revise members that had never entered the bank. Two such invalid revisions were
observed across the three rounds. v8 explicitly states atomic rejection semantics,
requires `add` when retrying an absent question, and reserves bundles for questions
that must be retained together. Every edit result now includes the actual
`retained_question_names`, including after rejection. This preserves atomic bundle
semantics and the existing answer thresholds; it makes the actual bank state
clear to the proposer. A separate empty-bank v8 run uses a fresh cache in
`live-bank-debug-v8/`. The optional v7 diagnostic was stopped after its completed
400-observation update to free shared bridge capacity for v8; v7 has no final
test result. Compare each representation
on that run's own identical feedback; do not treat different trajectories as a
controlled question-only comparison.

### Interrupted discovery recovery

Learning hydration honors the live Rune adapter's `--inference-concurrency`
limit, including historical replay and paraphrase checks. At most that many
prompt requests are outstanding; all missing question heads for each prompt are
sent together. Budget reservations, SQLite writes and replay results remain
ordered on the caller thread. Stateful adapters without an explicit concurrency
attribute remain serial. Proposal evaluation order and answer checks are unchanged.
Each noncached hydration pass logs row/head/call counts, concurrency and wall time.
This fixes the measured 2,000-prompt update's 28 minutes of serial Rune time;
GLM's three adaptive rounds remain sequential and contribute additional time.

The VM supervisor accepts `--resume` (also the optional sixth argument to
`ops/bootstrap-vm.sh`) with the original run root, prepared sample and output URI.
It removes only the stopped task-owned gateway container, probes the providers,
then invokes the runner's checked resume. `executions.jsonl` records supervisor
segments. Utility exports continue on the original stream. Final archives are
built outside the run tree so tar cannot reject its own directory mutation.

The v8 400-prompt update failed when GLM used its entire 16,384 output-token
allowance for reasoning and emitted no JSON (212,459 input tokens). This is an
actual failed proposer call, not an empty proposal or successful discovery stop.
Live proposals now retry missing parsed JSON at most twice, with the same inputs;
every failed attempt stays in the inference audit and consumes the call budget.
Other failures keep the existing bounded transport behavior.

The runner supports explicit `run --resume` with the same arguments and existing
run/cache directories. It verifies the sample fingerprint, configuration, exact
logged decisions, chosen feedback and partition order. Replaying the seeded draws
must reproduce every saved action before any new inference. It then restores
decision IDs/clock, training-only proposer states and the random stream. An
interrupted learning update is retried from the last activated snapshot; previous
candidate measurements remain cached but an unactivated bank is not promoted.
Recovery handles exactly one extra trailing SQLite decision when interruption
preceded its feedback/export: it verifies the original prompt, live policy version,
entire saved prefix and seeded action sequence before restoring that decision's
chosen feedback and deterministic utility export. It never reruns or resamples the
committed decision. Other ledger mismatches and activation/log disagreement fail
closed; recovery actions are recorded in `resumes.jsonl`.
Already-completed runs cannot be resumed. Prior calls count against the same
inference ceiling. `resumes.jsonl` records recovery boundaries, and summary
`elapsed_seconds` covers only the last process segment.

The v8 recovery verified 400 decisions with the last activated 200-prompt bank.
Source at initial launch and at recovery is archived separately. The optional
v7 run had collected 600 decisions by interruption, with its last completed
update at 400; its last proposer request was interrupted. The reporter separates
legacy incomplete audit records from successful calls, and future interruption
records explicitly contain the exception type. Neither v6 nor v7 has a completed
final-test result.

Policy training is measured separately. On the existing 200 test prompts, fitting
all 16 gives 0.611841 utility with exploration feedback versus 0.653526 without
questions; original logged feedback gives 0.616917 versus 0.651992. These match the
previous fixed-16 comparator: the routing regressor has not changed. The change
allows questions with promising evidence to survive and be refined independently
of that regressor. Final-test outcomes do not enter question evidence or retention.

### Completed fresh v8 result

The run completed 800 training and 200 evaluation prompts with 15 fresh questions
(three categorical, six binary, six ordinal). On its own matched chosen feedback,
greedy utility for none/fresh/reference is 0.671517/0.676220/0.693846; posterior
utility is 0.655931/0.637639/0.612183. Fresh greedy gain over none has a 95% interval
of [-0.043174, 0.052747]. This does not demonstrate improvement or equivalence to
the reference set. Full-feedback diagnostics, actual question wording, failure
counts, provenance and limitations are recorded in the
[experiment log](experiment-log-2026-10-14.md#live-loop-debugging-completed-v8-experiment).
The final 800-prompt update recovered two further missing-JSON responses on its
third bounded attempt. Local services were stopped after the completed evaluation.

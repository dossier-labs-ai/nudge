# Plan: Rune and Nudge RouterBench pilot

Status: Blocked on Rune inference availability; 994/1,000 prompts complete
Owner: Codex
Last updated: 2026-10-14 UTC

## Goal

User: "start a 1000 prompt pilot and let's measure", with GLM-5.2 thinking on
for question generation and the Rune DST Baseten endpoint for classification.

## Done Criteria

- Run 1,000 distinct public RouterBench prompts through real Rune inference;
  800 sequential training examples and 200 frozen held-out examples.
- Compare Rune probabilities and Nudge probabilities against recorded model
  outcomes, revealing only the selected action's reward to the learner.
- Record model/config/data identities, generated questions, quality, cost,
  hindsight regret, inference overhead, errors, and paired uncertainty.
- Verify the proposer sends thinking=true, and report any inference failure
  without substituting synthetic outcomes or a different model.

## Context

Read repository architecture, harness, Nudge spec, native Rune contracts in the
operator's checkout, and RouterBench's public dataset and reference evaluator.
Baseline: 38 package tests pass. All work stays in the experimental package.

## Policy / Mechanism Boundary

User specifies pilot size, Rune, and thinking-enabled GLM-5.2. Implementation
choices: pinned zero-shot dataset; deterministic stratified 800/200 split;
unit learning weights for shuffled, stationary data; 200-step learning cadence.
Use historical dataset costs, not claims about current downstream prices.
No customer data or production deployment/config changes. Native Rune requests
go through a local TensorZero function and its local protocol provider bridge.
The bridge translates OpenAI chat transport to native Choice heads; it does
not generate numeric confidence with a chat model.

## Plan

1. Prepare pinned data and tested replay/metric helpers.
2. Configure isolated local TensorZero functions and the Rune protocol bridge.
3. Verify real Rune and GLM thinking, run the pilot, preserve resumable artifacts.
4. Summarize measurements and complete package and repository validation.

Parallelism pass: batch discovery and baseline checks; download public data
while implementing adapters. Shared replay/contracts are edited serially.

## Validation

Native-protocol HTTP fixture tests, dataset/schema/split/leakage tests,
deterministic replay tests, native suite, Ruff, docs check, repository build,
and real pilot artifacts. No mocked local learning implementation.

## Ezra Humanization Review

N/A: standalone developer experiment with no staff-facing workflow.

## Shortcut Audit

Keep per-action counterfactual outcomes in the evaluator only. Do not turn
benchmark replay into full-information learning. No direct application model
calls; native transport belongs to the gateway's protocol provider bridge.
No endpoint activation, scaling, model deploys, or customer secret mutations.

## Progress Log

- 2026-10-13: Baseline tests pass; pinned public dataset download started.
- 2026-10-13: Real Rune probe returned valid probabilities; GLM-5.2
  returned two proposals with 248 reported reasoning tokens. Used the operator's
  existing Baseten credential for both endpoints; no secret mutations.
- 2026-10-13: Launched 1,000-prompt run after 45 tests passed. Added transport,
  cache independence and cost-report checks; 47 tests now pass. All 800 learning
  baselines cached successfully and first 200-feedback discovery is running.
- 2026-10-13: Initial full build encountered an unrelated agent-service health
  test returning 401 instead of 200. Its focused rerun passed 9/9; full build
  retry passed. No unrelated application code changes. Source/wheel builds and
  an isolated wheel CLI smoke check also passed.
- 2026-10-13: Completed 800 partial-feedback training decisions and all four
  learning updates. Retained code generation and multiple choice; six proposed
  additions were rejected for insufficient prediction-error improvement. No
  retained question was removed.
- 2026-10-14 UTC: First run stopped after 194/200 frozen test prompts with
  HTTP 502; provider transport reported Rune network/timeout failure. Replayed
  deterministically from the shared cache into `pilot-retry`; the same uncached
  inference failed with HTTP 502 again. No examples were skipped or substituted.
  Full pilot remains incomplete. Partial test quality: Rune 0.58627519, Nudge
  0.59015312; historical mean cost 0.00048494 vs 0.00049594 USD. Details and
  recovery instructions are in the package RouterBench runbook.
- 2026-10-14 UTC: User requested diagnosis. Added a read-only offline ablation
  tool with a real saved-policy regression test. It reproduces the stored
  critic and compares no-question, posterior-mean, full-training-feedback, and
  explicitly labeled oracle variants without inference. On the 194 completed
  test prompts, questions add 0.028 pp in the mean comparison; full feedback
  adds only 0.14 pp with representation/scale/tilt fixed. Detailed results and
  limitations are recorded in the package runbook. The six live requests remain
  blocked; diagnosis does not mark the original pilot complete.
- Diagnostic validation: all 48 Python tests, Ruff, diff whitespace checks,
  and the full repository build passed. Offline analysis changed no saved
  policy, feedback, or model configuration.

## Decisions

- Pilot evaluation is an adaptation of RouterBench with online partial feedback,
  not a reproduction of the paper's supervised router training protocol.

## Follow-Ups

- Finish the six remaining test prompts once the configured Rune endpoint
  accepts inference. Restart the local bridge/gateway and use a new run directory
  with the same prepared data, seed, configuration, and shared inference cache.
  Do not deploy/activate/scale the endpoint as part of this experiment.
- Multi-seed, cost-penalty sweeps and explicit drift experiments after the pilot.

## Expanded discovery follow-up

User: "We need to look at more training examples. We need more questions."
The question representation replaces input embeddings. Raised library and
pilot defaults to 256 newest training examples, 32 retained questions, three
rounds, and 16 proposals per round. Updated both proposer adapters and the
gateway JSON schema, exposed benchmark limits as CLI flags, and recorded them
in manifest/cache identity. Increased explicit learning/gateway call budgets
so old small caps do not silently prevent the expanded search. No live rerun
or endpoint mutation is included. Original measurements remain labeled with
their original settings; expanded experiments require new run directories.

Validation covers 256 actual training examples reaching discovery, 32 retained
question slots, and 16 proposals through both real HTTP gateway clients, with
an explicit lower-cap rejection. All 49 native tests, Ruff, CLI help smoke,
diff whitespace checks and the full repository build passed.

Expanded run attempt: local provider bridge and TensorZero restarted with the
expanded schema. The first live probe failed (gateway 502, native Rune 400).
Read-only Baseten management queries confirmed `rune-dst-v1` points to a single
deployment, status `INACTIVE`. No training or GLM proposal requests started.
Requested approval to reactivate this shared deployment, run the expanded
benchmark, and return it to INACTIVE afterward. No remote lifecycle mutation
has been made. Probe evidence: `benchmark-results/routerbench/expanded-probe.log`
and `expanded-cache/inferences.jsonl`; resolved settings: `expanded-probe/manifest.json`.

User approved activation. Baseten accepted POST activate for that model and
deployment (`success: true`, `no_op: false`); read-only verification
shows DEPLOYING. Scope is activation of this existing deployment, the expanded
benchmark, then deactivation back to its initial state. No hardware, scaling,
schedule, weights, or customer traffic configuration changes are authorized.

User subsequently authorized additional H100 capacity to accelerate the run.
Saved original scaling: min_replica=1, max_replica=2, autoscaling_window=60,
scale_down_delay=60, concurrency_target=32, target_utilization_percentage=30.
PATCH changed only min_replica/max_replica to 4/4. Verified ACTIVE with four
active H100:2 replicas (eight H100s). Restore min/max to 1/2, then deactivate
the deployment when the benchmark finishes or stops. No schedule/weights changes.

Stopped the two-worker expanded attempt and retained its cache. Added explicit
client/provider concurrency flags and per-learning-block feature prefetch;
decisions/feedback remain sequential. All 51 Python tests and Ruff passed,
including a real four-request parallel HTTP test and exact seeded-decision
equivalence with/without prefetch. Restarted as `expanded-fast` with 32 workers,
the same expanded cache, split, seed, model inputs and learning settings.

Further removed serial paraphrase inference by prefetching exact alternate
question definitions on the same newest training examples the core uses.
External HTTP/cache tests verify exact inputs and reuse; 52 Python tests and
Ruff pass. Restarted through the same cache as `expanded-parallel`, 32 workers.
This is the active run; `expanded` and `expanded-fast` are interrupted attempts.
The full build including paraphrase prefetch passed.

Recovery: `expanded-parallel-retry` stopped at 600 training decisions on a
TensorZero outbound timeout to the local native bridge, surfaced as HTTP 502.
Baseten remained ACTIVE with four replicas; this was not evidence of an inactive
deployment or a GLM failure. Restarted the bridge and replayed the shared cache
as `expanded-recovery` with eight inference workers. A further timeout occurred
during candidate evaluation; restarted the local TensorZero container and
replayed as `expanded-recovery2`, which passed both prior failure points.
No examples or learning settings were changed. The precise timeout root cause
is not established. The native bridge is API translation only; model inference
uses the remote H100 deployment. A process-scoped `caffeinate` prevents
idle sleep during this recovery run.

Completion: another timeout interrupted `expanded-recovery2` during the final
800-example update. Restarted TensorZero and replayed as `expanded-recovery3`;
it exited successfully with all 800 training and 200 test decisions. Combined
report: `benchmark-results/routerbench/expanded-report.json`. Test quality:
Rune 58.455%, Nudge 59.103%, gain 0.648 pp (conditional paired 95% interval
0.410–0.887 pp). Five questions retained. Historical downstream cost per 1,000:
USD 0.49049 / 0.50447. This is a modest improvement, not the oracle gap closed.
Published aggregate evidence and limitations in `docs/routerbench.md`.

Cleanup completed: restored Baseten min/max replicas to 1/2, verified the other
autoscaling settings unchanged, then deactivated the deployment. Independently verified
INACTIVE with zero active replicas. Stopped the owned local TensorZero container
and native bridge. No model/code deployment or question-batching change was made
during recovery. The underlying intermittent timeout cause remains unresolved.

Offline predictor follow-up: added `nudge.routerbench_predictors`, comparing
tuned linear, quadratic, shared quadratic, and shared RBF reward predictors using
only recorded chosen-action feedback and cached feature answers. Three-fold
training CV selected shared RBF, but its posterior-mean test quality was 59.066%
versus 59.028% for the original linear predictor (+0.038 pp) on the already
inspected 200-prompt split. This is evidence that predictor capacity alone is not
the substantial-improvement fix. Predictor and RouterBench tests pass; Ruff
passes. Report: `benchmark-results/routerbench/predictor-comparison.json`.

Categorical question support is now implemented across the public `Question`
contract, JSON proposal schema, core feature vector, TensorZero/TypeSafe/log-score
adapters, native Rune answerer, persistence round trips, and documentation. A
categorical answer is a validated soft probability mapping and expands to a
soft one-hot vector; it is not coerced into an ordinal score. Focused adapter and
math tests pass, Ruff passes, and the repository build passed before the final
additional categorical test was added.

Categorical smoke pilot: activated the existing deployment at original 1/2
scaling for a 100-prompt run (80 train, 20 test), then restored INACTIVE with
zero replicas. GLM proposed and Nudge retained `prompt_type`,
`output_format_type`, and `domain_knowledge` categorical heads. Test quality was
Rune 59.988% versus Nudge 61.256% (+1.268 pp), utility gain +0.012691, with
3,064 successful Rune calls and no errors. This is an exploratory n=20 smoke
result, not a confirmation benchmark. The run exposed and fixed categorical
SQLite serialization, regression dimension, and report-label bugs before the
successful retry. Artifacts live under
`benchmark-results/routerbench/categorical-pilot/`.

Fresh 1,000-prompt categorical run completed under
`benchmark-results/routerbench/categorical-1000-fresh2/` after adding five
bounded retries for transient inference failures. It used 800 training and 200
test prompts, 32 inference workers, one discovery round, and four proposals per
learning block. Two categorical questions were retained (`task_type` and
`output_format`); all 1,000 prompts completed with zero inference errors. Test
quality was Rune 58.438% versus Nudge 59.016% (+0.578 pp; paired bootstrap
95% interval +0.371–0.795 pp). Utility gain was +0.005650 and recorded cost
increased by USD 0.0000131 per prompt. Supervised all-outcomes test quality was
63.246%, so this remains an exploratory improvement with substantial headroom.
Baseten was restored to min/max 1/2 and deactivated with zero replicas; the
local bridge and gateway were stopped.

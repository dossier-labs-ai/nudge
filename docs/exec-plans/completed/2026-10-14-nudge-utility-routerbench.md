# Plan: Measure utility discovery on RouterBench

Status: Complete
Owner: Agent
Last updated: 2026-10-14

User requested measurement of the general decision-utility discovery change.
Reference: categorical-1000-fresh2/run, fixed 800/200 split, seed 20261013,
quality minus 10 * cost, learn every 200, one round and four proposals per update.
Keep that proposal/feedback/routing budget; compare the whole current discovery
loop (replay/history/v4 proposer/utility selector) with the archived v2/MSE run.
This is not a selector-only causal ablation. Freeze archived baseline vectors
to remove baseline inference variation; new question inference remains live.

Read runbook, pilot plan, runner/adapters, reference manifest and current local
gateway state. Add a tested reference-baseline CLI path and correct new-run
provenance. Run through the isolated local TensorZero gateway/native bridge,
using configured Secret Manager refs, no credentials in output. Do not modify
shared model lifecycle/scaling; reuse existing active inference if available.
Use new run/cache directories, preserve source and prepared-data fingerprints,
and compare paired final-test utilities and discovered/rejected questions.
Stop owned local processes on completion/failure. No customer operation.

Validation: baseline alignment/partition/fingerprint tests; package tests; live
complete 1000-row artifact and comparison, or exact inference blocker if unavailable.
Independent reads batched; shared adapter/runner edits serial. No agents needed.
Ezra review N/A: standalone experiment. No claims of new independent test data.

## Completion

Completed live run with 800/200 prompts and no inference errors; matched settings
and baseline verified (normalization roundoff maximum 1.11e-16). Utility 0.584653
versus 0.585129 previously; paired delta -0.000476, interval [-0.001806,+0.000822].
Zero retained questions: 12 coverage failures, 4 unstable. None of the scored
candidates clear the utility threshold without coverage. Report and source archive
are in utility-discovery-1000; detailed result is in runbook/experiment log.
99 package tests, full build and Ruff passed. Local owned gateway/bridge stopped;
shared deployment unchanged. No claimed uplift or selector-only attribution.

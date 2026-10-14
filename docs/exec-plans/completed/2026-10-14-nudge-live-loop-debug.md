# Plan: Debug live question discovery

Status: Complete
Owner: Agent
Last updated: 2026-10-14

## Goal

Run fresh v6 question discovery from an empty bank, obtain real Rune answers,
and identify where useful distinctions are lost. Compare generated questions,
the reference questions and no questions under identical training conditions.

## Done Criteria

- Real proposer calls, generated questions, revisions and answer checks saved.
- Matched chosen-feedback comparisons with greedy and posterior routing reported
  separately; retain the 800/200 boundary and do not tune on final-test outcomes.
- Explain the observed failure stage or measured improvement with artifacts.

## Context

Read ARCHITECTURE.md, docs/README.md, docs/AGENTIC_HARNESS.md,
docs/product-specs/index.md, specs/nudge-v0-spec.md and the package RouterBench
runbook, experiment log, live runner, discovery and saved-question evaluators.
Existing uncommitted changes and previous experiment artifacts are preserved.

## Policy / Mechanism Boundary

### Policy contract

User requests actual loop debugging and fresh discovery, not another substituted
reference retention run. Reference definitions never enter discovery. Only chosen
training outcomes enter discovery and fitting. Retention is a hypothesis, not proof.

### Mechanism contract

Existing isolated Python package, SQLite evidence and local TensorZero gateway.
No application, customer, deployment or database boundaries change. Real Rune
answers and GLM proposals use the documented gateway and Secret Manager flags.
Save source provenance, run configuration, inference audit and comparison output.

### Boundary tests

Baseline Python suite; any new evaluation tooling must verify aligned feedback,
identities, and invariance of fitted predictions to final-test rewards.

## Plan

1. Batch read-only discovery and baseline tests; inspect/start local transports.
2. Run empty-bank live discovery with real paraphrase checks in a fresh directory.
3. During inference, prepare matched evaluation tooling, then merge serially.
4. Inspect training evidence, make targeted fixes only when justified, and compare
   frozen questions on the final evaluation set without tuning against that set.
5. Record results and stop owned local services.

## Validation

- Baseline: 112 Python tests passed.
- Focused tests for handwritten changes, Ruff, docs check; full build if code changes.
- Strongest evidence: completed fresh inference and matched comparison artifacts.

## Ezra Humanization Review

N/A: isolated developer experiment with no staff-facing behavior.

## Shortcut Audit

- Saved reference answers are comparators only, never fresh-discovery evidence.
- Test rewards excluded from discovery, fitting and question selection.
- Parallelism pass: discovery reads and baseline checks batched; live inference
  can run alongside read-only diagnosis/tool implementation. No subagents needed.
- Live run and matched comparison completed; no improvement/equivalence claim.

## Progress Log

- 2026-10-14: Verified prior local gateway/bridge stopped. Restarted the documented
  local gateway and bridge, with 16 bridge workers. No shared scaling mutation.
- 2026-10-14: v6 generated and retained a four-question bundle with real Rune
  paraphrase checks. Two live proposer calls reported 254,496/297,466 input tokens.
  Whole decision records bloated the prompt. Preserved partial v6 artifacts and
  interrupted the owned runner; no final v6 result claimed.
- 2026-10-14: v7 projects sampled examples onto observed evidence without changing
  sampling or truncating prompts; adds proposer input audit on failed calls too.
  Started a fresh empty-bank v7 run. 114 Python tests and Ruff pass.
- 2026-10-14: Added matched evaluator with real-store alignment, synthetic useful
  feature recovery and test-reward invariance checks. Original feedback baseline
  gives greedy 0.676555 without questions versus 0.669325 with reference questions.
- 2026-10-14: v7 first call fell to 55,450 tokens on the same first 200 decisions.
  First two bundles each failed because one member was unstable. Subsequent
  calls twice attempted invalid revisions of absent rejected questions. v8
  clarifies atomic rejection/re-add semantics and reports the actual bank after
  every edit; started fresh v8 alongside v7, preserving both runs separately.
- 2026-10-14: Full repository build passed (557 files, 4,412 tests; 2 files/3 tests
  skipped). Targeted tests and Ruff pass for subsequent v8 feedback clarification.
  Separate full-outcome reference diagnostic: greedy improves 0.742267 to 0.778440;
  posterior declines 0.725929 to 0.697653. This is not discovery feedback.
- 2026-10-14: Stopped optional v7 diagnostic after its completed 400-observation
  update (five retained questions; repeated atomic-bundle failures). Concurrent
  runs were sharing the 16-worker bridge and slowing corrected v8. Preserve v7
  as partial training evidence, not a completed evaluation. v8 continues through
  the required full split and matched reference/no-question controls.
- 2026-10-14: v8 failed during the 400-prompt update: 16,384 reasoning/output
  tokens exhausted without JSON. Added three-attempt bounded parsed-output retry
  and explicit checked resume. A real-store regression verifies identical
  subsequent decisions after interrupted learning; HTTP fixture verifies bounded
  retries/audit and cache reuse. Resumed real v8 after verifying all 400 decisions,
  feedback and RNG draws, from the last activated 200-prompt bank.
- 2026-10-14: v7 had collected 600 decisions by SIGINT, last activated update 400.
  Its partial report correctly counts the interrupted proposer request separately.
  No new reference questions, feedback changes or test feedback entered recovery.

## Decisions

- Use current v6 defaults with three rounds, four proposals per round, 256 sampled
  examples, 32 maximum questions, 16 inference workers; all observed training rows
  remain eligible. Freeze original baseline probabilities for comparison.

## Follow-Ups

- Fresh v8 completed 800/200 with 15 generated questions. Matched chosen-feedback
  greedy utilities: none 0.671517, fresh 0.676220, reference 0.693846; posterior:
  none 0.655931, fresh 0.637639, reference 0.612183. Fresh greedy improvement is
  inconclusive (95% interval -0.043174 to 0.052747). The requested experiment is
  complete; finding an equally discriminative set remains unproven.
- Training evidence prompted a real drop and revisions; answer checks rejected ten
  edits in activated updates. Three proposer JSON failures are recorded, including
  two recovered by the bounded retry at the final update. 24,896 Rune calls, zero
  logical-call errors; HTTP retries are not separately counted.
- Investigate context growth, answer stability, sparse-feedback fitting and
  posterior routing using training-only studies or a fresh evaluation split. Do
  not optimize against the reported final 200 outcomes.
- Final artifacts and limitations are in the experiment log and
  `benchmark-results/routerbench/live-bank-debug-v8/`. Local services stopped;
  shared model lifecycle unchanged. 116 Python tests, full build, Ruff and package
  builds passed; final docs check and diff whitespace check passed.

# Plan: Online Nudge RouterBench 10k

Status: Complete — user ended the experiment at 5,000 prompts
Owner: Agent
Last updated: 2026-10-14

## Goal and Done Criteria

User approved running a continuous 10,000-prompt bandit stream. Score Rune and
Nudge before feedback on every prompt, reveal only the chosen outcome, and learn
every 200 prompts. Preserve rolling utility and cumulative advantage artifacts.

## Context and Boundaries

Read repository architecture, harness, Nudge spec/runbook and sampler, replay,
gateway, discovery and report implementations. Parallel read-only inspection;
serial edits to shared replay/report contracts. No customer operations, shared
provider scaling, deployment or production data. Use local TensorZero and existing
Rune/GLM endpoints. Credentials come from explicit Secret Manager references.

## Policy and Mechanism

Proportional sampling without replacement; seed 20261014; penalty 10; zero
holdout; default 1,024-row replay cap; 200-prompt combined learning/discovery.
Three discovery rounds, 32-question cap, 256 proposer examples, 16 proposals.
Use prior proven provider concurrency 16. Keep 200,000 logical inference-call
ceiling and checked resume; failures stop rather than skip examples. One seed
does not establish general improvement or monotonicity.

## Validation

Tests cover no-test completion, all chosen feedback, policy versions across
updates, rolling metrics and prefix invariance. Run package tests, Ruff, docs
checks and build before launch. Probe real providers; then record launch source,
manifest, sample fingerprint, logs and live progress.

## Ezra Humanization Review

N/A: isolated benchmark.

## Shortcut Audit

No train/test demonstration substituted for online learning. Historical outcomes
stay evaluator-only except the chosen action. Existing batching and GLM retries
do not imply bounded context or complete discovery batching.

## Progress

- Oct 15 06:20 UTC: User chose the 5,000-prompt result and explicitly requested
  stopping the benchmark and deactivating Rune. Supervisor stopped at exactly
  5,000 scored prompts, archived `run-v2/artifacts.tar.gz` (140,830,315 bytes,
  uploaded 06:20:51 UTC), deactivated the deployment, and shut down the VM.
  Independently verified VM `TERMINATED` and provider `INACTIVE`. The stop is
  recorded as `KeyboardInterrupt`; the final learning update was interrupted,
  which cannot change the already-recorded pre-feedback evaluation scores.
  Final mean utility: Nudge 0.6671034989734627, Rune 0.5809910636182534,
  relative gain 14.8216%. Website assets use exactly this 5,000-prompt prefix,
  plotting cumulative average score with axes from 0 to 5,000 prompts.
  Website renderer tests, full Python suite (131 passed, two local-artifact
  checks skipped), Ruff and wheel/sdist builds passed. Website deployment is
  separate from preparing these assets.

- Resumed successfully at 04:29 UTC Oct 15 under `rune-bench-v4`, using
  `input/source-v5.tar.gz` and the original `run-v2` output/root. Resume verified
  2,591 committed decisions and restored the final decision's missing feedback
  and export. Learning resumed at 2,600. All 131 Python tests, Ruff and full
  repository build passed (4,593 repository tests passed, three skipped).
  Live utility mirror resumed; 2,600-prompt cumulative mean Nudge/Rune utility
  is 0.671321/0.576553. Live speed verification passed: first resumed update
  completed in approximately 7.25 minutes; full 1,024-call replay passes measured
  53–59 seconds with concurrency 16, compared with about 208 seconds at the
  earlier measured serial rate. Whole-update workloads differ, so their durations
  are not a controlled speed ratio. Timing evidence is retained locally in
  `rune-bench-vm-20261014/speed-fix-live-hydration.log`. At 04:53 UTC the run
  was active at 3,150 prompts; cumulative mean utility Nudge/Rune 0.668102/0.577348.

- Resume exposed a shutdown interruption between decision and feedback/export:
  2,591 committed decisions, 2,590 feedback rows and exports. Added narrowly
  checked recovery for exactly one trailing committed decision, preserving its
  sampled action/probabilities. Tests inject real SQLite feedback failure and
  filesystem export failure, compare the complete recovered trajectory with an
  uninterrupted run, and reject corrupt prefixes before recovery writes. The
  pre-fix snapshot is preserved at `run-v2/before-speed-fix.tar.gz`; the first
  failed resume archive is also retained locally. Systemd now signals only the
  supervisor first (`KillMode=mixed`) so it can stop children and archive cleanly.

- Speed fix: replay hydration now opts into the live adapter's concurrency (16),
  with at most that many outstanding prompt requests. Each request still groups
  all missing heads for its prompt. Cache writes, budget reservation and result
  application stay on the calling thread in replay order. Proposal order, replay
  membership, full-sample variance checks and paraphrase checks are unchanged;
  early rejection is deferred because it could change diagnostic feedback to GLM.
  Tests compare serial/concurrent learned policies and verify batching, overlap,
  cache reuse and strict call budgets. Hydration logs report calls and wall time.
  VM resume preserves the existing database/cache and records execution provenance.
  User-visible proof still required: a completed live update with measured timing.

- Oct 15 03:09 UTC: Investigated apparent stall at 2,000 prompts directly on VM.
  Update ran 02:33:19–03:08:26 UTC, then routing resumed (2,100 exported).
  Audited update: 8,320 successful single-head Rune calls, zero Rune errors,
  1,687 seconds summed Rune latency; four GLM attempts, one error/retry,
  371 seconds summed GLM latency. Core reports 8,323 logical calls (the GLM
  retry is internal to its proposer invocation). Eight candidate additions each
  hydrated 1,024 rows before 16 paraphrase checks; six were rejected as unstable.
  `discovery.hydrate` runs serially despite routing-prefetch concurrency 16.
  This is a measured systems bottleneck, not a deadlock. Next implementation
  should screen candidates before full hydration and bound/parallelize hydration
  without altering replay membership or adaptive proposal order. No runtime
  settings or learning semantics changed during this diagnostic.

- User stopped the local run and authorized a temporary VM in the operator's GCP project,
  switching Rune to a second Baseten model's `eval` environment and turning it down afterward. Local worker,
  bridge and gateway stopped; interrupted evidence retained. New provider needs
  a fresh run/cache with the same sampled prompts.
- User requires live utility exports. Added atomic CSV/JSON/PNG/status exports,
  GCS publishing and local sync, plus tested provider cleanup on startup failure.
  GLM remains Baseten serverless through TensorZero; Rune is dedicated H100.
- User requires all GCP artifacts to use `rune-bench` branding. Replaced the
  just-created unused resources before workload launch. Dedicated VM
  `rune-bench-10k-20261014` (e2-standard-4, us-central1-a),
  a dedicated service account, private bucket
  `gs://<bucket>`: authorized benchmark infrastructure.
  Account access is limited to this bucket and the named Baseten secret.
  VM supervisor requests deactivation and powers off after success/failure;
  artifacts and disk are retained. Run limit 24 hours, readiness allowance 20 minutes.
- VM attempt `run/` failed before inference because the management scaling path
  used a custom environment name as a deployment ID. Its logs/archive are preserved;
  VM shutdown completed. Fixed lifecycle calls to the inspected deployment ID,
  with a contract test, and made deactivation independent of minimum-restore failure.
- Restarted as systemd `rune-bench-v2`, output
  `gs://<bucket>/run-v2/`, source `input/source-v2.tar.gz`.
  Confirmed the deployment is waking with min/max replicas 1/1. Gateway and
  bridge bind only 127.0.0.1. Local live mirror is
  `benchmark-results/routerbench/rune-bench-vm-20261014/live/`.
  Rune reached ACTIVE with one replica; both real provider probes passed.
  At 00:04 UTC Oct 15, 200 prompts were scored and the first learning update began.
  GCS contains live utility CSV/JSON/PNG and status/lifecycle artifacts. First-block
  Nudge/Rune expected utility: 0.501158/0.556330; cumulative gain -11.034276.
  This is prior exploration before the first learned policy, not a final result.
- All 126 Python tests pass. Full repository build passed after export/supervisor
  addition; targeted lifecycle tests and Ruff passed after the management fix.

- Continuous evaluation and live reporting implemented. All 123 Python tests,
  Ruff and docs checks passed. Full repository build passed (4,593 tests passed,
  three skipped), including package builds.
- Live Rune and GLM probes succeeded. Proposals from the standalone empty-context
  probe are not installed in the learner. Started seed 20261014 under
  `benchmark-results/routerbench/online-10k-20261014/`.
  Sample fingerprint: `815745fb86baeaa326b3001ed2a14cd0669c7dc79a235132b70ca43164ec07d9`.
  Archived source, git head and working diff; prepared sample, probe, cache,
  run manifest and logs preserved. Completion and utility results remain pending.

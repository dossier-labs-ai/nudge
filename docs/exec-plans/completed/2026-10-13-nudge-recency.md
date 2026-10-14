# Plan: Time-decayed Nudge learning

Status: Complete
Owner: Codex
Last updated: 2026-10-13

## Goal

Implement the user's instruction: "when learning, look at the latest data point
first, and use a time based decay factor to underweight the impact of training
data from the past."

## Done Criteria

- Learning visits labeled decisions newest first, uses explicit time-based
  exponential weights, and applies them to critic fitting and selection.
- Weight age uses decision timestamps rather than feedback receipt/correction.
- Half-life and reference timestamp are recorded in fitted snapshots; existing
  SQLite files and immutable historical decisions/snapshots remain readable.
- Paired stationary/reversal benchmarks use an explicit simulated clock so
  decay is independent of benchmark execution speed.

## Context

Read AGENTS.md, docs/AGENTIC_HARNESS.md, the existing Nudge spec, package math,
store, core, discovery, benchmark, and tests. Architecture/docs conventions were
read during the preceding implementation. Baseline has 29 Python tests.
All prior uncommitted files belong to the ongoing Nudge implementation.

## Policy / Mechanism Boundary

User owns recency weighting. Implementation choices: newest labeled decision
is reference time; default half-life seven days, adjustable per learn() call.
No new data deletion, live inference, app integration, or infrastructure.

## Plan

1. Implement newest-first time weighting and weighted Gaussian likelihood/MSE.
2. Thread weights through discovery, snapshots, evidence counts and reporting.
3. Add deterministic clocks and recency benchmark settings; test late feedback,
   chronological order, uncertainty, correction, and learning after reversal.
4. Run paired benchmarks, document the tradeoff, and complete package/repo checks.

Parallelism pass: batch independent code reads and start baseline tests before
edits. Math/persistence changes merge serially with discovery and benchmark
consumers because they share row contracts. Final validation follows integration.

## Validation

Analytic weighted posterior tests, timestamp/clock tests with real SQLite,
weighted selection/proposer ordering, deterministic paired benchmarks,
Ruff, native suite, repository build, wheel build and docs checks.

## Ezra Humanization Review

N/A: only the standalone developer library changes.

## Shortcut Audit

No alternate legacy fitting path: uniform weighting is simply the same weighted
fit with unit weights, available for controlled comparisons. Existing timestamp
columns suffice; no schema migration or historical snapshot rewrites needed.
Full historical reward range remains the clipping/scale definition; recency
weights govern the likelihood and selection statistics.

## Progress Log

- 2026-10-13: Read current implementation and started baseline package suite.
- 2026-10-13: Baseline 29 tests pass. Implemented decision-time decay,
  newest-first proposer/backfill order, weighted fitting/selection/feature checks,
  evidence-aware posterior sampling, snapshot metadata and report evidence mass.
- 2026-10-13: 38 tests and Ruff pass, including analytic posterior weights,
  uncertainty, late/corrected feedback, old-store preservation and reward reversal.
  Source/wheel builds pass.
- 2026-10-13: Ten paired seeds, 1,000 steps, 200-step fits, one hour/step:
  reversal second-half regret 216.36 -> 202.09 and test gain -0.09 -> +3.38 pp;
  stationary test gain 6.69 -> 4.71 pp. Both outcomes documented in package docs.
- 2026-10-13: Full repository build passes (557 test files passed, two skipped;
  4,412 tests passed, three skipped). Standalone wheel benchmark smoke passes
  outside the repository with explicit half-life and simulated-clock flags.
  Used an uncached tool environment to avoid reusing an earlier 0.1.0 wheel.

## Decisions

- Configure half-life on learn(), making each fitted snapshot self-describing
  without changing the persisted immutable choice contract.
- Decision timestamp, newest labeled point as reference; no decay merely due
  to idle time when no new labeled evidence arrives.

## Follow-Ups

- Real-model evaluations and formal policy promotion remain out of scope.

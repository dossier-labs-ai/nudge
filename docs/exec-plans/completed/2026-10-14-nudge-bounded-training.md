# Plan: Bounded training replay

Status: Complete
Owner: Agent
Last updated: 2026-10-14

## Goal

Bound each learning update independently of decision history while retaining
recent, representative historical and informative chosen-action observations.
Commit the implementation; no push or production operations requested.

## Context and boundaries

Read architecture, harness, Nudge spec, core, storage, replay, discovery,
question evidence, live proposer and their tests. Python experiment only; no
staff workflow, application schema, deployment or provider changes.

## Design and done criteria

- Expert.max_training_rows defaults to 1024; learning uses at most that many
  distinct labeled decisions for hydration, discovery, evidence and fitting.
- Feedback maintains bounded recent/uniform/informative candidate pools;
  indexed lookups load their union, never the entire audit log.
- Preserve complete decisions/feedback, corrections, decision-time recency,
  atomic snapshots, new-observation evidence boundaries and total feedback counts.
- Bound proposal-history and pending-state reads; remove proposer-side unbounded
  inference prefetch. All feature hydration uses the learning call budget.
- Explicit fresh-store migration boundary for old schema/contracts.

## Parallelism and validation

Batch independent source reads and start Python baseline checks early. Storage
and core edits overlap contracts, so integrate serially. Test real SQLite replay
retention, corrections, restart, bounded reads and inference; full Python suite,
Ruff, docs check and root build before commit.

## Policy / mechanism boundary

Replay is an intentionally biased training mixture, not a claim of unbiased
population evidence. Uniform history remains independently retained; informative
reservoirs balance actions and favor prediction surprise at feedback time.
Fixed configuration bounds row work; token lengths and database index lookup
costs are not constant wall-clock guarantees. Audit reporting can scan history.

## Ezra Humanization Review

N/A: developer-only experiment.

## Shortcut Audit

Verified learn/discovery read bounded replay via indexed payload lookups, not
Store.rows; proposer inference prefetch and all-history state accumulation are
removed. Proposal history has a choice/sequence index; pending results are
incrementally materialized and capped at 64. Audit reporting remains intentionally
unbounded and is outside the training operation. No provider calls, production
mutations, temporary scripts, or fabricated live-performance claims.

## Validation and outcome

- Existing 116-test Python suite passed after storage/core integration.
- Full `pnpm run build` passed with the initial four new replay tests, including
  docs/lint/typecheck, full unit suite and all package builds.
- Added a pending-state cap/objective regression; all five focused replay tests
  passed afterward, along with Ruff, docs and whitespace checks.
- Real SQLite tests cover full-log preservation, bounded feature requests,
  uniform historical membership, recent retention, action balancing, corrections,
  restart, version rejection without mutation, and constant SQLite instruction
  work for replay reads at 100 versus 1,000 historical decisions.
- Defaults to 1024 rows with CLI and Expert overrides. Schema v1 requires a fresh
  store/refit; historical stores are preserved. No live routing-utility benchmark
  was run, so the effect of replay sampling on model quality remains unmeasured.

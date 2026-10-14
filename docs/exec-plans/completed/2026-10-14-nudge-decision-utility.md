# Plan: General decision-utility discovery

Status: Complete
Owner: Agent
Last updated: 2026-10-14

## Goal and done criteria

User approved a general mechanism: the LLM designs/refines questions using replay
and measured historical question utility. Implement in Nudge's Choice contract,
without RouterBench actions, labels or metrics. Replace MSE acceptance with paired
held-out decision-utility estimates; give the proposer results, uncertainty,
coverage and current-question removal diagnostics. Test the actual learn loop.

## Mechanism choices

Use a paired doubly robust estimate with logged propensities and a common current
critic fitted only on training data. Evaluate the bounded policy, approximating
its posterior sampling by deterministic common Monte Carlo draws. Report sampling
uncertainty, importance-weight effective sample sizes and unsupported actions.
Accept only positive normalized gain above min_gain after a 1.96-standard-error
margin, with sufficient coverage. This is an exploratory guard, not statistical
certification after adaptive search. LLM receives aggregate selection metrics,
never selection examples; independent final tests remain external to discovery.

Keep MSE diagnostic only. Preserve historical objective labels, decision logs,
cache, runtime action policy and explicit learn triggering. Revise categorical
answer diagnostics to use vectors rather than arbitrary category ordering. Allow
another bounded round after rejection so feedback can lead to a revision.

## Plan and evidence

Read core discovery, math, types, store, adapters, benchmark/tests and Nudge spec.
Prior full build and 85 tests passed. Independent read-only discovery batched;
edits serial because evaluator, context and discovery share interfaces.
Add evaluator tests for known outcomes, changed preferences, identical policies,
partial coverage and reward scaling, plus learn/history/refinement integration.
Run Python suite, Ruff, docs check and full build. Document formulas and boundaries.
Ezra review N/A: standalone developer experiment, no staff workflow. No live
inference, deployment or benchmark uplift claim. No persistence schema migration;
old proposal objective labels retained, new snapshots record the new objective.

## Completion evidence

Implemented generic evaluator, utility acceptance, per-question removal diagnostics,
vector answer checks, v4 proposer feedback and continued bounded rejection rounds.
Old history objective labels remain intact; snapshots label the new selector.
96 Python tests pass, including real SQLite discovery/restart and rejection-driven
refinement. The known-allocation fixture recovers utility 4 -> 5 from only chosen
feedback with a deliberately wrong reward model. A lower-MSE feature cannot pass
when it leaves the policy unchanged. Ruff and full build pass; repository Vitest
reports 557 passed files / 4,412 passed tests, with 2 files / 3 tests skipped.
No live inference, deployment, benchmark uplift claim or unfinished runtime path.
Finite Monte Carlo and adaptive confidence limitations are documented in design.md.

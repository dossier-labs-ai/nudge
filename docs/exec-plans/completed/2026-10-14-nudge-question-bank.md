# Decouple question discovery from routing training

Status: Completed implementation and saved-question measurement

User invariant: determine question usefulness from historical action outcomes;
use incoming data to refine questions independently of the routing classifier.

Implement a retained question bank and explicit discovery-only API. LLM-proposed,
answerable questions are hypotheses, not certified utility improvements. Give the
proposer general action-by-answer outcome comparisons using logged propensities,
and prospective comparisons from decisions arriving after the question was kept.
No policy-value or reward-MSE gate. Explicit learn fits the bank; discovery alone
preserves serving questions/critics. Use all available feedback for discovery.

Read-only discovery batched; shared core/discovery/storage edits are serial.
Measure saved Rune heads with the same general evidence mechanism; separate
retention counts, subsequent directional evidence, and final routing utility.
Preserve old reports. No live inference or invented paraphrase checks. Tests:
common difficulty vs action discrimination, propensities, prospective boundary,
restart, unchanged serving after discovery, test leakage, package/build/docs.

## Outcome

Separate bank/evidence and serving questions/critic persist in snapshots.
`discover_questions()` does not fit the policy; default `learn()` does not invoke
the proposer. An explicit combined operation remains available. All labeled rows
inform the question loop. Per-question profiles freeze the initial evidence and
decision insertion boundary, including pending unlabeled decisions. Later rows
provide separate comparisons; old corrections cannot masquerade as validation.
The v6 proposer receives bounded evidence summaries. A new choice/store is required
by the versioned discovery contract; old reports/stores remain historical evidence.

Saved original 16-question diagnostic retains all 16 as hypotheses. Eight show
positive cross-products and majority direction agreement in a retrospective
400/400 exploration replay. This does not prove all retained questions useful,
nor represent new live LLM discovery or independent prospective validation.
All-16 routing utility remains 0.611841 vs 0.653526 without questions. Retention
no longer depends on that classifier result. Evidence and reproduction:
`benchmark-results/routerbench/question-bank-retention.json`
and the package RouterBench runbook. No test outcomes select questions.

112 Python tests pass, including real SQLite separation/restart, action-relative
signal vs common difficulty, inverse propensities, subsequent-evidence boundaries,
and final-test isolation. Full repository build passed (557 test files, 4,412
tests), plus Ruff/docs checks and Python source/wheel build. The last added default
fit-without-discovery assertion was checked with the focused discovery suite.
No deployment, inference service mutation, commit or push.

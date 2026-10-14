# Rune semantic answerability

Status: complete. All 1,000 prompts / 16,000 labels measured. All 77 package
tests, Ruff, docs check and diff whitespace check pass. Final management read
confirms ACTIVE, two replicas and original 2/2 settings with all other scaling
fields unchanged. Local gateway and bridge stopped.

User: "let's check if Rune can answer these questions" and "you can spin up
more Rune replicas." Compare native Rune Choice probabilities for the same 16
semantic questions against saved GLM labels on all 1,000 existing prompts.

Operation boundary: the shared experimental Baseten model, environment
rune-dst-v1, and its deployment. Read original state, activate and scale to four
replicas, run the diagnostic, restore captured min/max and initial activation
state. No model/code deployment, customer runtime, scheduler or dataset writes.
Local TensorZero and native bridge remain the inference path; credentials from
Secret Manager stay in memory/pipes and never appear in files or output.

Read: architecture/harness/Nudge spec from this session, Rune provider bridge,
Gateway/Choice validation, semantic rubric and saved analysis, previous run log,
and the operator's Baseten management endpoint contracts. Direct curl calls
use those verified management APIs; no temporary scripts or control-code edits.

Plan: save baseline deployment state; implement/test a retained answerability
audit; probe native multi-head inference; complete 16,000 semantic measurements;
report per-factor agreement, majority baseline, macro recall, confusion and
coding/numeric subset scores. Evaluate frozen GLM-derived supported routing
trees with Rune labels, plus refitted Rune-label trees, on recorded outcomes.
GLM agreement is agreement with a reference annotator, not proven ground truth.

Parallelism: batch read-only discovery and tests; prepare local audit while
replicas initialize. Runtime mutations and final cleanup remain sequential.
Default to 16 native Choice heads per prompt; verify complete label sets and
probabilities before a full run. Cache all successful responses for recovery.

Validation: native payload and probability validation, metrics fixtures,
real probe, full sample coverage/replay, package suite, Ruff/docs checks.
Stop boundary: result/failure followed by restoring owned runtime changes.

Observed initial state differs from old logs: ACTIVE, two H100:2 replicas,
min/max 2/2. Restore 2/2 and leave ACTIVE. Captured full management response in
`rune-semantic-before.json`. Changed only min/max to 4/4 using the verified API.

Probe: eight prompts, 16 heads each; all 128 top categories agree between
batched and single-head requests. Maximum probability difference 0.0926666.
Use batching for this categorical diagnostic; do not assume identical soft
features in Nudge's current single-head learning loop. Both probes retained.

Results: overall GLM agreement 70.84375%; reference tracking 90.6%; option
distinction 27.1%, including only 6/334 no-options prompts recognized. Refitted
supported Rune-label tree utility 0.79101845 vs GLM 0.791792478. Frozen GLM
rules with Rune labels fall to 0.764927076. Full combination fit is 0.89522375804
but uses 773 cells/647 singletons, explicitly documented as sample fitting.
Artifacts contain distributions, confusion matrices, applicable subsets, and
policy transfer/refitting comparisons. No learner policy changes.

Operational evidence: 1,128 real logical inference requests including probes;
zero failed calls. Requested scaling 4/4; observed three active replicas during
and after inference. Restored min/max 2/2; an initial read still showed three
replicas during scale-down, then `rune-semantic-final-state.json` confirmed two.

# Frozen 800/200 semantic regressor measurement

User requested the missing experiment: semantic questions, Rune answers, and a
learned regressor evaluated on the existing 200-prompt test partition.

Use the saved original 16-question rubric and Rune probability vectors; the
revised 17-question rubric has no live measurements. Train the existing Nudge
per-action Gaussian linear critic on only 800 training prompts. Preserve its
prior/regularization. Report both bounded posterior-mean tilt and greedy
predicted-utility routing. Separate original chosen-action feedback from the
stronger full-training-outcomes experiment so information settings stay clear.

No inference, activation, scaling, learner mutation or deployment. Read prepared
data, fresh-run decisions/manifest, archived rubric and Rune measurements;
validate fingerprints, prompt identities, partition membership and probability
vectors. No test labels enter fitting or reward scaling. Reuse core math and
evaluation. Tests must verify changing test outcomes cannot change predictions.

Parallelism: independent discovery batched where useful; small analysis module
owns all edits. Validate tests/Ruff/docs and real complete artifact execution.
No temporary scripts. Done means reporting actual 200-prompt metrics with
training information and policy rule explicitly identified.

Completed: evaluator and artifact produced all four comparisons on 200 test
prompts. Utility: chosen/bounded 0.585693, chosen/greedy 0.669325,
full/bounded 0.587040, full/greedy 0.778440. All 80 unit tests and evaluator
Ruff pass; leakage test confirms test outcomes do not affect predictions.
Results and reproduction documented in the experiment log and RouterBench guide.

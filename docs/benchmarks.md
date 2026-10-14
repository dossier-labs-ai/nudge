# Measuring Nudge

## Included synthetic suite

```sh
uv run nudge-benchmark --steps 1000 --learn-every 200 --seeds 0,1,2 \
  --output benchmark-results/v0.json
```

The default methods are baseline sampling, Nudge without questions, and Nudge
with discovery. Also available through `--methods`: uniform, fixed_features,
with posterior winner selection in every Nudge method.
Each decision advances a simulated clock by one hour (`--step-seconds 3600`).
Training defaults to a seven-day half-life (`--half-life-seconds 604800`);
`--half-life-seconds none` supplies unit weights for controlled comparisons.
Time decay never depends on benchmark runtime. Output records both settings.

| Scenario | What it measures |
| --- | --- |
| calibration | Intercept/confidence correction when the baseline favors the worse arm |
| context | Discovering a feature that reverses which action is best |
| no_signal | Chance feature selection when all actions have identical expected reward |
| weak_support | An optimal action starting at 0.5% baseline probability |
| noisy_features | A useful feature with a deterministic 25% measurement flip rate |
| interaction | XOR reward structure requiring a composite question |
| nonstationary | A halfway reversal of the reward function; compare adaptation with and without decay |

Features and proposals here are scripted. These experiments test the learning
mechanism, not an LLM's feature discovery or any particular HF model. The learner
receives only the selected action's outcome. The evaluator owns full reward
means and uses common contexts/potential-outcome randomness across methods.
Decision IDs and all sampling are deterministic for a given seed. Each fitted
policy is finally tested on 256 new contexts without further feedback/learning.

### Metrics

- **Cumulative pseudo-regret:** sum of best expected action reward minus expected
  reward under the actual distribution used at each step. This integrates the
  final action draw and includes the fixed posterior ensemble policy.
- **Tilt-oracle regret:** compare with the best clipped-score exponential tilt
  using the same baseline, drift limit, and mixture. The oracle knows true reward
  means. It is not the larger class of all distributions satisfying a per-action
  ratio bound. This is an archived comparator; current Nudge and uniform policies
  can beat it, giving negative tilt-oracle regret.
- **Reward:** expected online reward, realized sampled reward, and frozen-policy
  test reward/gain over the baseline on identical test contexts.
- **Diagnostics:** action counts, largest baseline probability ratio, learning
  model calls, final questions, weighted evidence, and cumulative learning curves.
- **Second-half regret:** regret after the midpoint, which is the reward reversal
  in the nonstationary case.
- **Across-seed uncertainty:** descriptive mean and standard error across seeds;
  a small seed count does not establish statistical significance.

Use unrestricted oracle regret for the current policy. The historical tilt
comparator remains solely to compare with earlier bounded-policy experiments.

### Initial smoke evidence

Run on 2026-10-13: 1,000 decisions, updates every 200, seeds 0/1/2, default
parameters before time decay was implemented. This historical table used unit
weights, bounded tilt, and residual-ranked proposer examples; current code mixes recent,
representative and informative training replay. These are small synthetic smoke
results, not tuned headline claims.

| Scenario | Baseline cumulative regret | Discovery cumulative regret | Discovery test reward gain |
| --- | ---: | ---: | ---: |
| calibration | 450.00 | 412.80 | +4.83 percentage points |
| context | 404.27 | 359.35 | +6.72 pp |
| no_signal | 0.00 | 0.00 | 0.00 pp |
| weak_support | 597.00 | 596.27 | +0.12 pp |
| noisy_features | 404.27 | 394.52 | +1.66 pp |
| interaction | 397.73 | 352.51 | +6.72 pp |
| nonstationary | 399.73 | 403.03 | -0.01 pp |

The weak-support and nonstationary cases are material limitations, not excluded
outliers. That initial implementation had no forgetting; time decay was added
subsequently, as measured below. Uniform exploration is still absent. In the
context case, no-feature Nudge has approximately zero test gain; the improvement
depends on extracting a useful feature. A zero-regret no-signal environment
cannot establish that discovery avoids spurious questions: inspect selected
question counts separately.

### Time-decay comparison

Run on 2026-10-13 with seeds 0 through 9, 1,000 decisions, updates every 200,
one simulated hour per decision, and 256 frozen-policy test contexts. Both
configurations use the same newest-first implementation and streams; only
the likelihood/selection decay differs. The seven-day default was chosen before
running these comparisons, not tuned on their results.

```sh
uv run nudge-benchmark --steps 1000 --learn-every 200 \
  --seeds 0,1,2,3,4,5,6,7,8,9 --scenarios context,nonstationary \
  --methods discovery --half-life-seconds none --output benchmark-results/recency-unit.json
uv run nudge-benchmark --steps 1000 --learn-every 200 \
  --seeds 0,1,2,3,4,5,6,7,8,9 --scenarios context,nonstationary \
  --methods discovery --half-life-seconds 604800 --output benchmark-results/recency-decay.json
```

| Scenario | Weights | Total regret | Second-half regret | Final test gain over baseline |
| --- | --- | ---: | ---: | ---: |
| stationary context | unit | 356.19 | 170.12 | +6.69 pp |
| stationary context | seven-day decay | 366.73 | 177.99 | +4.71 pp |
| reward reversal | unit | 402.43 | 216.36 | -0.09 pp |
| reward reversal | seven-day decay | 390.84 | 202.09 | +3.38 pp |

Decay reduced post-reversal regret by about 6.6% and improved final adaptation,
at the cost of weaker stationary learning because it retains less evidence.
These are synthetic measurements with scripted questions/proposals, not a
universal half-life recommendation. Tune the half-life to the actual rate of
change and decision volume. All raw outcomes remain in the log for refitting.

## Live RouterBench pilot

The [RouterBench pilot runbook](routerbench.md) defines a pinned 1,000-prompt
real Rune / thinking-enabled GLM-5.2 evaluation, its partial-feedback protocol,
held-out split, metrics and runtime evidence. This is separate from the synthetic
measurements above.

## Proposed next benchmarks

1. **Longer regret curves and ablations.** Run at least 20 paired seeds at
   horizons 1k/5k/20k. Compare baseline, uniform, no_features, fixed_features,
   discovery and mean_only. Add standard disjoint LinUCB and linear Thompson
   sampling on the identical fixed features as separate comparison algorithms.
   Report mean curves, paired differences with uncertainty, area under regret,
   late-horizon slope, and fractions of seeds worse than baseline. Do not give
   algorithms different feature access without labeling the comparison.

2. **Exploration/drift frontier.** Sweep baseline optimal-arm probability
   (0, 0.001, 0.01, 0.1), max_drift (1, 2, 4, 8), and mixture share. Measure time
   to twenty observations, arm starvation, excess loss relative to baseline,
   constrained regret, and achieved probability ratios. Include a zero-support
   case to explicitly show the irrecoverable boundary.

3. **Representation and real HF backend comparison.** Hold the environment and
   proposer budget fixed; compare oracle feature answers, noisy answers, actual
   model-extracted questions, and a fixed embedding linear bandit. Use exact
   checkpoint/tokenizer revisions and the same reward stream. Measure learning
   curves, question stability, feature count, inference calls/tokens, latency,
   and dollars per reward improvement. An HF checkpoint requires a verified
   closed-set scoring backend; the included LogScoreAnswerer supplies its seam.

4. **Classification-to-bandit benchmark.** Use public labeled text tasks with
   2-10 actions; expose only chosen-label correctness to each learner. Keep labels
   out of all state/prompt/proposer inputs except the observed chosen reward.
   Split final test examples before discovery. This yields known counterfactual
   rewards for exact regret comparisons, following the supervised-data reduction
   used in [A Contextual Bandit Bake-off](https://www.jmlr.org/papers/v22/18-863.html).
   Add a declared per-action cost to test reward-weight sensitivity.

5. **Delayed/missing feedback and distribution shift.** Sweep random delays,
   action-dependent missingness, outcome-dependent missingness, reward reversals,
   and gradual drift. Measure regret versus feedback received, adaptation delay,
   and observed-vs-true reward bias. Distinguish failure under biased missingness
   from a promise of bias correction; v0 makes no such promise.

6. **OPE validation later.** Once an evaluator is implemented, test IPS/SNIPS/DR
   against exact synthetic truth and the [Open Bandit Dataset](https://arxiv.org/abs/2008.07146).
   Measure bias, RMSE, effective sample size, interval coverage, and wrong policy
   rankings under overlap and missingness stress. Logged real-world outcomes
   alone cannot reveal each policy's true online regret. This is independent of
   whether v0 requires a promotion gate.

Prioritize 1 and 2, then 3 and 4. They directly test the conservative learning
mechanism and whether language-derived features earn their inference cost.

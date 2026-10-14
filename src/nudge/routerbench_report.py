"""Read-only live-pilot status, overhead accounting, and comparison plots."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from .routerbench import write_json
from .routerbench_data import MODELS, aggregate


def read_records(path):
    if not path.exists():
        return []
    # Concurrent writers can leave the last line incomplete until their flush.
    return [
        json.loads(line)
        for line in path.read_text().splitlines(keepends=True)
        if line.endswith("\n")
    ]


def overhead(records, input_price=1.40, cache_price=0.14, output_price=4.40):
    result = {}
    for function in sorted({r["function"] for r in records}):
        calls = [r for r in records if r["function"] == function]
        successful = [r for r in calls if "error" not in r
                      and ('result' in r or 'usage' in r)]
        latency = [r["latency_seconds"] for r in successful]
        result[function] = {
            "calls_including_probe": len(calls),
            "errors": sum("error" in r for r in calls),
            "interrupted_or_incomplete": sum('error' not in r and 'result' not in r
                                             and 'usage' not in r for r in calls),
            "input_tokens": sum(r.get("usage", {}).get("input_tokens", 0) for r in successful),
            "output_tokens": sum(r.get("usage", {}).get("output_tokens", 0) for r in successful),
            "latency_seconds_p50_p95": np.quantile(latency, [0.5, 0.95]).tolist()
            if latency
            else None,
        }
        if function == "nudge_propose":
            cached = sum(
                r.get("provider_usage", {}).get("prompt_tokens_details", {}).get("cached_tokens", 0)
                for r in successful
            )
            tokens = result[function]
            tokens["cached_input_tokens"] = cached
            tokens["reasoning_tokens"] = sum(
                r.get("provider_usage", {})
                .get("completion_tokens_details", {})
                .get("reasoning_tokens", 0)
                for r in successful
            )
            tokens["estimated_usd"] = (
                (tokens["input_tokens"] - cached) * input_price
                + cached * cache_price
                + tokens["output_tokens"] * output_price
            ) / 1_000_000
        else:
            result[function]["estimated_usd"] = None
            result[function]["cost_note"] = (
                "Shared dedicated Rune GPU cost is unattributed, not zero."
            )
    return result


def supervised_zero_router(data, target_cost):
    """Train a context-free mixture using the full training outcome table.

    This is an explicitly full-information comparator, never learner feedback.
    Optimize mean quality under the supplied mean training-cost budget.
    """
    q = np.mean([r["quality"] for r in data["train"]], axis=0)
    c = np.mean([r["cost"] for r in data["train"]], axis=0)
    candidates = []
    for i in range(len(MODELS)):
        if c[i] <= target_cost:
            p = np.zeros(len(MODELS))
            p[i] = 1
            candidates.append(p)
        for j in range(i):
            if c[i] == c[j]:
                continue
            weight = (target_cost - c[j]) / (c[i] - c[j])
            if 0 <= weight <= 1:
                p = np.zeros(len(MODELS))
                p[i], p[j] = weight, 1 - weight
                candidates.append(p)
    if not candidates:
        return {"status": "budget_below_cheapest_training_model"}
    best = max(candidates, key=lambda p: p @ q)
    return {
        "training_information": "all recorded outcomes, unlike Nudge's chosen feedback",
        "budget_training_usd": target_cost,
        "probabilities": {m: float(p) for m, p in zip(MODELS, best) if p > 0},
        "test_quality": float(np.mean([r["quality"] for r in data["test"]], axis=0) @ best),
        "test_cost": float(np.mean([r["cost"] for r in data["test"]], axis=0) @ best),
    }


def online_curve(records, window=200):
    """Pre-feedback expected utility; trailing windows never use future rows."""
    if window < 1:
        raise ValueError('window must be positive')
    rows = [r for r in records if r['partition'] == 'train']
    cumulative = {p: np.cumsum([r[p]['utility'] for r in rows]) for p in ('rune', 'nudge')}
    return [{
        'prompts': i + 1,
        'window_count': min(i + 1, window),
        **{p: float((values[i] - (values[i - window] if i >= window else 0)) /
                    min(i + 1, window)) for p, values in cumulative.items()},
        'cumulative_gain': float(cumulative['nudge'][i] - cumulative['rune'][i]),
    } for i in range(len(rows))]


def plot(records, summary, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not summary.get('test'):
        curve = online_curve(records)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        x = [r['prompts'] for r in curve]
        for policy in ('rune', 'nudge'):
            axes[0].plot(x, [r[policy] for r in curve], label=policy)
        axes[0].set(xlabel='Prompts with feedback', ylabel='Trailing 200-prompt utility')
        axes[0].legend()
        axes[1].plot(x, [r['cumulative_gain'] for r in curve])
        axes[1].axhline(0, color='gray', linewidth=1)
        axes[1].set(xlabel='Prompts with feedback', ylabel='Cumulative Nudge − Rune utility')
        fig.suptitle('Online RouterBench · scored before feedback · historical downstream utility')
        fig.savefig(output, dpi=170)
        plt.close(fig)
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    train = [r for r in records if r["partition"] == "train"]
    for policy, color in (("rune", "#65758b"), ("nudge", "#147d64")):
        axes[0].plot(
            np.arange(1, len(train) + 1),
            np.cumsum([r[policy]["hindsight_regret"] for r in train]),
            label=policy,
            color=color,
        )
    axes[0].set(
        xlabel="Training prompts (chosen-action feedback)",
        ylabel="Cumulative hindsight utility regret",
        title="Learning on 800 prompts",
    )
    axes[0].legend()
    families = list(summary["test_by_family"])
    metrics = [summary["test_by_family"][f]["paired_nudge_minus_rune"]["quality"] for f in families]
    means = np.array([m["mean"] for m in metrics]) * 100
    intervals = np.array([m["prompt_bootstrap_95"] for m in metrics]) * 100
    axes[1].errorbar(
        means,
        np.arange(len(families)),
        xerr=np.maximum(0, np.stack([means - intervals[:, 0], intervals[:, 1] - means])),
        fmt="o",
        color="#147d64",
        capsize=3,
    )
    axes[1].set_yticks(np.arange(len(families)), families)
    axes[1].axvline(0, color="#65758b", linewidth=1)
    axes[1].set(
        xlabel="Nudge − Rune quality (percentage points)",
        title="Frozen 200-prompt test; paired 95% intervals",
    )
    fig.suptitle(
        "RouterBench pilot · one seed · utility = quality − 10 × historical USD cost", fontsize=11
    )
    fig.savefig(output, dpi=170)
    plt.close(fig)


def compare_runs(reference, current):
    """Paired final-test comparison; fail rather than compare different baselines/splits."""
    before = json.loads((reference / 'summary.json').read_text())
    after = json.loads((current / 'summary.json').read_text())
    old_manifest = json.loads((reference / 'manifest.json').read_text())
    new_manifest = json.loads((current / 'manifest.json').read_text())
    if old_manifest['data_sha256'] != new_manifest['data_sha256']:
        raise ValueError('comparison sample mismatch')
    if old_manifest['max_proposals'] != new_manifest['max_proposals']:
        raise ValueError('comparison proposal budget mismatch')
    for key in ('seed', 'penalty_per_usd', 'learn_every', 'expert'):
        if before[key] != after[key]:
            raise ValueError(f'comparison setting mismatch: {key}')
    old = [r for r in read_records(reference / 'decisions.jsonl') if r['partition'] == 'test']
    new = [r for r in read_records(current / 'decisions.jsonl') if r['partition'] == 'test']
    if not old or [r['id'] for r in old] != [r['id'] for r in new]:
        raise ValueError('comparison test alignment mismatch')
    # Core validation renormalizes roundoff; do not confuse that with new inference.
    baseline_difference = max(abs(a['rune_probabilities'][m] - b['rune_probabilities'][m])
                              for a, b in zip(old, new) for m in MODELS)
    if baseline_difference > 1e-12:
        raise ValueError('comparison baseline changed')
    paired = aggregate([{'family': b['family'], 'rune': a['nudge'], 'nudge': b['nudge']}
                        for a, b in zip(old, new)])
    events = [e for fit in after['learning'] for e in fit['discovery']]
    return {
        'reference_run': str(reference), 'current_run': str(current), 'test_count': len(new),
        'maximum_baseline_probability_difference': baseline_difference,
        'rune_baseline': after['test']['rune'], 'previous_nudge': paired['rune'],
        'current_nudge': paired['nudge'],
        'current_minus_previous': paired['paired_nudge_minus_rune'],
        'previous_questions': before['report']['questions'],
        'current_questions': after['report']['questions'],
        'proposal_status_counts': dict(Counter(e['status'] for e in events)),
        'utility_threshold_passes_ignoring_coverage': sum(
            e.get('gain_lower_bound', -float('inf')) > after['expert']['min_gain']
            for e in events),
        'coverage_rejections': [
            {'after': fit['after'], 'question': e['proposal']['question'],
             'underobserved_changed_actions': e.get('underobserved_changed_actions', []),
             'action_effective_sample_sizes': e.get('action_effective_sample_sizes', {})}
            for fit in after['learning'] for e in fit['discovery']
            if e['status'] == 'insufficient_coverage'],
        'scope': 'Whole discovery-loop comparison at matched settings; reused exploratory test split.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument('--reference-run', type=Path)
    parser.add_argument("--glm-input-per-million", type=float, default=1.40)
    parser.add_argument("--glm-cache-per-million", type=float, default=0.14)
    parser.add_argument("--glm-output-per-million", type=float, default=4.40)
    args = parser.parse_args()
    records = read_records(args.run_dir / "decisions.jsonl")
    summary_file = args.run_dir / "summary.json"
    result = {
        "complete": summary_file.exists(),
        "inference_counting_scope": (
            "Audited logical cache-miss calls. Internal bounded HTTP retries are not separately counted."
        ),
        "decisions": dict(Counter(r["partition"] for r in records)),
        "online_curve": online_curve(records),
        "inference_overhead": overhead(
            read_records(args.cache_dir / "inferences.jsonl"),
            args.glm_input_per_million,
            args.glm_cache_per_million,
            args.glm_output_per_million,
        ),
        "glm_usd_per_million": {
            "input": args.glm_input_per_million,
            "cached_input": args.glm_cache_per_million,
            "output": args.glm_output_per_million,
        },
    }
    if summary_file.exists():
        summary = json.loads(summary_file.read_text())
        result.update(
            train=summary["train"],
            test=summary["test"],
            test_by_family=summary["test_by_family"],
            fixed_models=summary["fixed_model_test"],
            retained_questions=summary["report"]["questions"],
        )
        if args.prepared and summary['test']:
            result["supervised_zero_router"] = supervised_zero_router(
                json.loads(args.prepared.read_text()), summary["train"]["rune"]["cost"]
            )
        if args.plot:
            plot(records, summary, args.plot)
        if args.reference_run:
            result['comparison'] = compare_runs(args.reference_run, args.run_dir)
    elif args.plot:
        plot(records, {}, args.plot)
    if args.out:
        write_json(args.out, result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

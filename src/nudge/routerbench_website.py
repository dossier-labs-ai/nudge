"""Render a fixed RouterBench prefix as website SVG/PNG assets and public metrics."""

import argparse
import csv
import json
import math
from pathlib import Path


def cumulative_scores(snapshot, count):
    """Recover prefix sums from the exact trailing-window means, without resampling."""
    curve = snapshot['curve'][:count]
    window = snapshot['status']['window']
    if count < 1 or len(curve) != count or window < 1:
        raise ValueError('snapshot does not contain the requested prompt count')
    sums = {policy: [0.0] for policy in ('nudge', 'rune')}
    result = []
    for i, point in enumerate(curve, 1):
        width = min(i, window)
        if point['prompts'] != i or point['window_count'] != width:
            raise ValueError('snapshot must contain contiguous, correctly sized windows')
        row = {'prompts': i}
        for policy in sums:
            total = float(point[policy]) * width + sums[policy][i - width]
            if not math.isfinite(total):
                raise ValueError('scores must be finite')
            sums[policy].append(total)
            row[policy] = total / i
        if not math.isclose(sums['nudge'][-1] - sums['rune'][-1],
                            point['cumulative_gain'], abs_tol=1e-7):
            raise ValueError('snapshot cumulative gain does not match its window means')
        result.append(row)
    return result


def render(snapshot, output, count=5000):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    rows = cumulative_scores(snapshot, count)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    final = rows[-1]
    summary = {
        'prompts': count, 'averaging': 'all prompts processed so far',
        'nudge': final['nudge'], 'rune': final['rune'],
        'relative_gain': final['nudge'] / final['rune'] - 1 if final['rune'] else None,
        'penalty_per_usd': snapshot['status']['penalty_per_usd'],
        'source_exported_at': snapshot['status']['updated_at'],
        'metric': 'Expected answer quality minus cost penalty; scored before feedback',
        'limitations': 'One seed. Routing overhead excluded. Fixed prefix of the original 10k protocol.',
    }
    with (output / 'scores.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=['prompts', 'nudge', 'rune'], lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    # Keep only aggregate metrics, not prompts, decisions, credentials or provider logs.
    (output / 'source.json').write_text(json.dumps({
        'status': {key: snapshot['status'][key]
                   for key in ('window', 'penalty_per_usd', 'updated_at')},
        'curve': snapshot['curve'][:count],
    }, separators=(',', ':')) + '\n')
    colors = {'nudge': '#15755e', 'rune': '#7c8190'}
    names = {'nudge': 'Nudge + Rune', 'rune': 'Rune alone'}
    values = [r[p] for r in rows for p in colors]
    plt.rcParams.update({'svg.fonttype': 'none', 'font.family': 'DejaVu Sans'})
    for mobile in (False, True):
        fig, ax = plt.subplots(figsize=(4.8, 5.5) if mobile else (9, 5.5))
        fig.patch.set_facecolor('#ffffff')
        fig.subplots_adjust(left=.21 if mobile else .14, right=.91 if mobile else .96,
                            top=.75, bottom=.19)
        fig.text(.06 if mobile else .08, .94, 'Nudge vs. Rune', fontsize=20, weight='bold', color='#182523')
        fig.text(.06 if mobile else .08, .88,
                 f'{count:,} prompts · Average score so far · Higher is better',
                 fontsize=9 if mobile else 12, color='#59645f')
        for policy in ('nudge', 'rune'):
            ax.plot([r['prompts'] for r in rows], [r[policy] for r in rows],
                    color=colors[policy], linewidth=2.1, label=names[policy],
                    linestyle='-' if policy == 'nudge' else '--')
        ax.set_xlim(0, count)
        ax.set_ylim(min(0, math.floor(min(values) * 10) / 10),
                    max(1, math.ceil(max(values) * 10) / 10))
        ax.set_xlabel('Prompts processed', fontsize=12, labelpad=12)
        ax.set_ylabel('Average answer quality\nadjusted for cost', fontsize=12, labelpad=12)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f'{value:,.0f}'))
        ax.tick_params(axis='both', length=0, labelsize=10, pad=8, colors='#59645f')
        ax.grid(axis='y', color='#e6ebe8', linewidth=.8)
        ax.set_axisbelow(True)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.legend(loc='lower left', bbox_to_anchor=(0, 1.015), frameon=False,
                  ncol=2, fontsize=10, borderaxespad=0)
        fig.text(.06 if mobile else .08, .045,
                 f"Average across all {count:,}:  Nudge {final['nudge']:.3f}  ·  Rune {final['rune']:.3f}",
                 fontsize=10, color='#182523')
        stem = 'comparison-mobile' if mobile else 'comparison'
        for extension in ('svg', 'png'):
            fig.savefig(output / f'{stem}.{extension}', dpi=180,
                        metadata={'Title': f'Nudge and Rune on {count:,} RouterBench prompts'})
            if extension == 'svg':
                path = output / f'{stem}.{extension}'
                path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines()) + '\n')
        plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--out-dir', required=True, type=Path)
    parser.add_argument('--count', type=int, default=5000)
    args = parser.parse_args()
    print(json.dumps(render(json.loads(args.input.read_text()), args.out_dir, args.count), indent=2))


if __name__ == '__main__':
    main()

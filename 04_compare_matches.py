"""
Step 4: Compare gait metrics across multiple matches for the same player.
This answers: does asymmetry increase with fixture congestion or over a season?

Usage:
    python 04_compare_matches.py output/match_001_timeline.json output/match_002_timeline.json output/match_003_timeline.json
"""

import argparse
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats


def load_timeline(path):
    with open(path) as f:
        return json.load(f)


def summarize_match(timeline, label):
    """Compute per-match summary metrics."""
    if len(timeline) < 10:
        return None

    knee = [t['knee_asymmetry'] for t in timeline]
    hip = [t['hip_asymmetry'] for t in timeline]
    valgus = [t['valgus_asymmetry'] for t in timeline]

    # Split into thirds: early, mid, late
    n = len(timeline)
    third = n // 3

    return {
        'label': label,
        'n_points': n,
        'minutes': timeline[-1]['minute'] - timeline[0]['minute'],
        # Means
        'knee_mean': np.mean(knee),
        'hip_mean': np.mean(hip),
        'valgus_mean': np.mean(valgus),
        # Late-match values (last third)
        'knee_late': np.mean(knee[2*third:]),
        'hip_late': np.mean(hip[2*third:]),
        'valgus_late': np.mean(valgus[2*third:]),
        # Early-match values (first third)
        'knee_early': np.mean(knee[:third]),
        'hip_early': np.mean(hip[:third]),
        'valgus_early': np.mean(valgus[:third]),
        # Within-match degradation
        'knee_degradation': (np.mean(knee[2*third:]) - np.mean(knee[:third])) / (np.mean(knee[:third]) + 1e-8) * 100,
        'hip_degradation': (np.mean(hip[2*third:]) - np.mean(hip[:third])) / (np.mean(hip[:third]) + 1e-8) * 100,
    }


def plot_comparison(summaries, output_path):
    """Plot cross-match comparison."""
    labels = [s['label'] for s in summaries]
    x = range(len(labels))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Cross-Match Biomechanical Comparison', fontsize=14)

    # 1. Mean asymmetry per match
    ax = axes[0][0]
    width = 0.25
    ax.bar([i - width for i in x], [s['knee_mean'] for s in summaries], width, label='Knee', color='steelblue')
    ax.bar([i for i in x], [s['hip_mean'] for s in summaries], width, label='Hip', color='coral')
    ax.bar([i + width for i in x], [s['valgus_mean'] for s in summaries], width, label='Valgus', color='mediumpurple')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Mean Asymmetry Index')
    ax.set_title('Overall Asymmetry Per Match')
    ax.legend()

    # 2. Late-match asymmetry (fatigue signal)
    ax = axes[0][1]
    ax.bar([i - width for i in x], [s['knee_late'] for s in summaries], width, label='Knee', color='steelblue')
    ax.bar([i for i in x], [s['hip_late'] for s in summaries], width, label='Hip', color='coral')
    ax.bar([i + width for i in x], [s['valgus_late'] for s in summaries], width, label='Valgus', color='mediumpurple')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Late-Match Asymmetry')
    ax.set_title('Last 30 Min Asymmetry (Fatigue Window)')
    ax.legend()

    # 3. Within-match degradation
    ax = axes[1][0]
    ax.bar([i - 0.15 for i in x], [s['knee_degradation'] for s in summaries], 0.3, label='Knee', color='steelblue')
    ax.bar([i + 0.15 for i in x], [s['hip_degradation'] for s in summaries], 0.3, label='Hip', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Degradation (%)')
    ax.set_title('Within-Match Degradation (Early vs Late)')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()

    # 4. Data quality
    ax = axes[1][1]
    ax.bar(x, [s['n_points'] for s in summaries], color='green', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel('Valid Measurements')
    ax.set_title('Data Points Per Match (Quality Indicator)')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare gait metrics across matches')
    parser.add_argument('timelines', nargs='+', help='Timeline JSON files from step 2')
    parser.add_argument('--output', default='cross_match_comparison.png', help='Output plot path')
    args = parser.parse_args()

    summaries = []
    for i, path in enumerate(args.timelines):
        timeline = load_timeline(path)
        label = f"Match {i+1}"
        summary = summarize_match(timeline, label)
        if summary:
            summaries.append(summary)
            print(f"\n{label}: {summary['n_points']} measurements, {summary['minutes']:.0f} min range")
        else:
            print(f"\n{label}: Too few data points, skipping")

    if len(summaries) < 2:
        print("Need at least 2 valid matches to compare.")
        sys.exit(1)

    # Print comparison table
    print(f"\n{'='*70}")
    print("CROSS-MATCH COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<25} ", end="")
    for s in summaries:
        print(f"{s['label']:<15}", end="")
    print()
    print("-" * 70)

    rows = [
        ('Knee asym (mean)', 'knee_mean'),
        ('Knee asym (late)', 'knee_late'),
        ('Knee degradation %', 'knee_degradation'),
        ('Hip asym (mean)', 'hip_mean'),
        ('Hip asym (late)', 'hip_late'),
        ('Hip degradation %', 'hip_degradation'),
        ('Valgus asym (mean)', 'valgus_mean'),
        ('Valgus asym (late)', 'valgus_late'),
    ]

    for label, key in rows:
        print(f"{label:<25} ", end="")
        for s in summaries:
            val = s[key]
            print(f"{val:<15.4f}", end="")
        print()

    # Cross-match trend
    if len(summaries) >= 3:
        print(f"\n{'='*70}")
        print("CROSS-MATCH TREND (does asymmetry increase match-over-match?)")
        print(f"{'='*70}")
        match_idx = list(range(len(summaries)))
        for name, key in [('Knee late', 'knee_late'), ('Hip late', 'hip_late'), ('Valgus late', 'valgus_late')]:
            vals = [s[key] for s in summaries]
            rho, p = stats.spearmanr(match_idx, vals)
            print(f"  {name}: rho={rho:.3f}, p={p:.3f}", end="")
            if p < 0.1:
                print("  <- TREND DETECTED")
            else:
                print()

    plot_comparison(summaries, args.output)

    print(f"\nNext: If signals look real, talk to 5 people (see action plan).")
    print(f"If noise dominates, try club tactical cameras or higher fps.")


if __name__ == '__main__':
    main()

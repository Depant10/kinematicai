#!/usr/bin/env python3
"""
Phase 1: Aggregation analysis script

Takes one or more match JSON files and produces a per-bucket statistical analysis.
"""
import json
import glob
import os
import argparse
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    if nx < 2 or ny < 2:
        return 0.0
    dof = nx + ny - 2
    pool_var = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof
    if pool_var == 0:
        return 0.0
    return (np.mean(x) - np.mean(y)) / np.sqrt(pool_var)

def interpret_d(d):
    d = abs(d)
    if d < 0.2: return "negligible"
    elif d < 0.5: return "small"
    elif d < 0.8: return "medium"
    else: return "large"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs='+', default=["output/match_*_players.json"], help="Input JSON files")
    parser.add_argument("--bucket-size", type=int, default=15, help="Bucket size in minutes")
    parser.add_argument("--threshold", type=float, default=0.65, help="Confidence threshold")
    args = parser.parse_args()

    files = []
    for pattern in args.inputs:
        files.extend(glob.glob(pattern))

    if not files:
        print("No input files found.")
        return

    print(f"Processing {len(files)} files...")
    
    # Flatten all clips
    all_frames = []
    match_coverage = set()
    
    for file in files:
        match_id = os.path.basename(file).split('_')[1]
        with open(file, 'r') as f:
            data = json.load(f)
            for track_key, track_data in data.items():
                track_id = track_data.get('track_id', track_key)
                team = track_data.get('team', 'Unknown')
                for frame in track_data.get('timeline', []):
                    if frame.get('lower_confidence', 0) >= args.threshold:
                        minute = float(frame.get('minute', 0))
                        # bucket idx
                        bucket_idx = min(90, int(minute // args.bucket_size) * args.bucket_size)
                        bucket_label = f"{bucket_idx}-{bucket_idx + args.bucket_size}" if bucket_idx < 90 else "90+"
                        
                        frame['bucket_label'] = bucket_label
                        frame['bucket_idx'] = bucket_idx
                        frame['track_id_global'] = f"{match_id}_{track_id}"
                        frame['match_id'] = match_id
                        
                        all_frames.append(frame)
                        match_coverage.add((bucket_label, match_id))

    if not all_frames:
        print("No frames passed confidence threshold.")
        return

    # Extract available buckets
    buckets = sorted(list(set(f['bucket_label'] for f in all_frames)), key=lambda x: int(x.split('-')[0]) if '-' in x else 90)
    
    # Check coverage
    print("\n--- Match Coverage per Bucket ---")
    all_matches = sorted(list(set(f['match_id'] for f in all_frames)))
    for b in buckets:
        covered = [m for m in all_matches if (b, m) in match_coverage]
        missing = [m for m in all_matches if (b, m) not in match_coverage]
        if missing:
            print(f"Bucket {b} has no data for: {', '.join(['match_'+m for m in missing])}")
        else:
            print(f"Bucket {b} has data for all {len(all_matches)} matches.")

    metrics = ['knee_asymmetry', 'hip_asymmetry', 'hip_drop', 'valgus_asymmetry']
    
    # Output structure
    bucket_stats = {m: {} for m in metrics}
    
    print("\n--- Summary Statistics ---")
    for metric in metrics:
        print(f"\nMetric: {metric}")
        
        # Prepare for plotting
        plot_data = []
        plot_labels = []
        
        for b in buckets:
            b_frames = [f for f in all_frames if f['bucket_label'] == b]
            # Some clips might not have all metrics, hip_drop is derived or present?
            # Let's check if metric is in frame. If hip_drop isn't, calculate from left/right hip angles if available
            b_values = []
            track_ids = set()
            for f in b_frames:
                val = f.get(metric)
                if val is None and metric == 'hip_drop':
                    if 'left_hip_angle' in f and 'right_hip_angle' in f:
                        val = abs(f['left_hip_angle'] - f['right_hip_angle'])
                if val is not None:
                    b_values.append(val)
                    track_ids.add(f['track_id_global'])
            
            n = len(b_values)
            n_clips = len(track_ids)
            
            if n < 30 or n_clips < 3:
                print(f"  WARNING: Bucket {b} has insufficient sample (n={n}, clips={n_clips})")
                
            if n > 0:
                mean_val = np.mean(b_values)
                median_val = np.median(b_values)
                std_val = np.std(b_values, ddof=1) if n > 1 else 0
                p25 = np.percentile(b_values, 25)
                p75 = np.percentile(b_values, 75)
                
                bucket_stats[metric][b] = {
                    'n': n,
                    'n_clips': n_clips,
                    'mean': mean_val,
                    'median': median_val,
                    'std': std_val,
                    'p25': p25,
                    'p75': p75,
                    'raw': b_values # Not saved to JSON, just for stats
                }
                
                print(f"  Bucket {b:>6}: n={n:>4}, clips={n_clips:>2} | mean={mean_val:.4f}, median={median_val:.4f}, std={std_val:.4f}")
                
                plot_data.append(b_values)
                plot_labels.append(b)

        # Plotting
        if plot_data:
            plt.figure(figsize=(10, 6))
            plt.boxplot(plot_data, tick_labels=plot_labels, showfliers=False)
            plt.title(f'Distribution of {metric.replace("_", " ").title()} across Match Minutes')
            plt.xlabel('Match Minute Buckets')
            plt.ylabel(metric.replace("_", " ").title())
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'output/{metric}_boxplot.png', dpi=150)
            plt.close()
            
            # Scatter with error bars
            means = [np.mean(d) for d in plot_data]
            stds = [np.std(d, ddof=1) if len(d) > 1 else 0 for d in plot_data]
            plt.figure(figsize=(10, 6))
            plt.errorbar(range(len(plot_labels)), means, yerr=stds, fmt='o-', capsize=5)
            plt.xticks(range(len(plot_labels)), plot_labels)
            plt.title(f'{metric.replace("_", " ").title()} Means with Standard Deviation')
            plt.xlabel('Match Minute Buckets')
            plt.ylabel(metric.replace("_", " ").title())
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(f'output/{metric}_scatter.png', dpi=150)
            plt.close()

        # Statistical comparison (First vs Last)
        if len(buckets) >= 2:
            first_b = buckets[0]
            last_b = buckets[-1]
            if first_b in bucket_stats[metric] and last_b in bucket_stats[metric]:
                b1_vals = bucket_stats[metric][first_b]['raw']
                b2_vals = bucket_stats[metric][last_b]['raw']
                
                if len(b1_vals) >= 2 and len(b2_vals) >= 2:
                    u_stat, p_val = stats.mannwhitneyu(b1_vals, b2_vals, alternative='two-sided')
                    d_val = cohen_d(b1_vals, b2_vals)
                    d_interp = interpret_d(d_val)
                    
                    bucket_stats[metric]['comparison'] = {
                        'bucket_A': first_b,
                        'bucket_B': last_b,
                        'mann_whitney_u': u_stat,
                        'p_value': p_val,
                        'cohens_d': d_val,
                        'effect_size': d_interp
                    }
                    
                    print(f"  Comparison ({first_b} vs {last_b}):")
                    print(f"    Mann-Whitney U: {u_stat:.1f}, p={p_val:.4e}")
                    print(f"    Cohen's d: {d_val:.4f} ({d_interp} effect)")

    # Clean up 'raw' from JSON
    for m in bucket_stats:
        for b in bucket_stats[m]:
            if isinstance(bucket_stats[m][b], dict) and 'raw' in bucket_stats[m][b]:
                del bucket_stats[m][b]['raw']

    # Also output full data structure for the React dashboard
    dashboard_data = {
        "buckets": buckets,
        "metrics": bucket_stats,
        "all_frames_subsampled": [] # We can pass a subset of frames to frontend for scatter plots
    }
    
    # Subsample for frontend to prevent massive files
    for m in metrics:
        for b in buckets:
            b_frames = [f for f in all_frames if f['bucket_label'] == b]
            # just need values for the scatter
            vals = []
            for f in b_frames:
                val = f.get(m)
                if val is None and m == 'hip_drop':
                    if 'left_hip_angle' in f and 'right_hip_angle' in f:
                        val = abs(f['left_hip_angle'] - f['right_hip_angle'])
                if val is not None:
                    vals.append(val)
            
            # Subsample up to 200 points per bucket per metric
            if len(vals) > 200:
                vals = np.random.choice(vals, 200, replace=False).tolist()
            
            if m not in dashboard_data:
                dashboard_data[m] = {}
            dashboard_data[m][b] = vals

    with open('output/bucket_analysis.json', 'w') as f:
        json.dump(bucket_stats, f, indent=2)
        
    # Overwrite data.js with the new aggregated data
    # We still need the original player data for the "Clip Inspector" mode
    with open('dashboard-app/src/data.js', 'r') as f:
        data_js = f.read()
        
    new_data_js = data_js + "\n\nexport const AGGREGATE_DATA = " + json.dumps(dashboard_data) + ";\n"
    with open('dashboard-app/src/data.js', 'w') as f:
        f.write(new_data_js)
    # we'll wait and do this cleaner if needed. Wait, data.js is simple. Let's just output aggregate_data.json and import it.
    with open('output/aggregate_data.json', 'w') as f:
        json.dump(dashboard_data, f)
        
    print("\nWrote output/bucket_analysis.json")
    print("Wrote output/aggregate_data.json")

if __name__ == '__main__':
    main()

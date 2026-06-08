#!/usr/bin/env python3
"""
Phase 3: Per-player temporal analysis
Analyzes an individual player's gait over the course of a match using SoccerNet tracks.

License Constraint: NON-COMMERCIAL RESEARCH USE ONLY (NDA-BOUND).
"""

import os
import json
import argparse
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

METRICS = ['knee_asymmetry', 'hip_asymmetry', 'hip_drop', 'valgus_asymmetry']
BUCKET_SIZE_MINS = 15
MIN_TRACK_MINUTES = 30

def cohens_d(x, y):
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2: return 0.0
    dof = nx + ny - 2
    pool_var = ((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof
    if pool_var == 0: return 0.0
    return (np.mean(x) - np.mean(y)) / np.sqrt(pool_var)

def analyze_player_timeline(processed_json):
    with open(processed_json, 'r') as f:
        match_data = json.load(f)
        
    match_id = os.path.basename(processed_json).replace('_players.json', '')
    
    out_dir = os.path.join('soccernet_processed', 'analysis', match_id)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"--- Phase 3: Player Timeline Analysis ---")
    print(f"Match: {match_id}")
    
    results = {}
    summary_counts = {m: 0 for m in METRICS}
    valid_players = 0
    insufficient_players = 0
    
    for p_key, p_data in match_data.items():
        timeline = p_data.get("timeline", [])
        if not timeline:
            continue
            
        minutes = [f["minute"] for f in timeline]
        min_min = min(minutes)
        max_min = max(minutes)
        duration = max_min - min_min
        
        if duration < MIN_TRACK_MINUTES:
            insufficient_players += 1
            # print(f"  Track {p_data['track_id']} excluded: only {duration:.1f} mins tracked.")
            continue
            
        valid_players += 1
        print(f"\nAnalyzing Track {p_data['track_id']} ({duration:.1f} mins):")
        
        # Bucket data
        max_bucket_idx = int(max_min // BUCKET_SIZE_MINS)
        buckets = [f"{i*BUCKET_SIZE_MINS}-{(i+1)*BUCKET_SIZE_MINS}'" for i in range(max_bucket_idx + 1)]
        
        player_res = {"track_id": p_data["track_id"], "duration_mins": round(duration, 1), "metrics": {}}
        
        for metric in METRICS:
            bucket_vals = {b: [] for b in buckets}
            for f in timeline:
                b_idx = int(f["minute"] // BUCKET_SIZE_MINS)
                bucket_vals[buckets[b_idx]].append(f[metric])
                
            # Compute stats
            stats_by_bucket = {}
            for b in buckets:
                vals = bucket_vals[b]
                if len(vals) > 0:
                    stats_by_bucket[b] = {
                        "n": len(vals),
                        "mean": np.mean(vals),
                        "std": np.std(vals)
                    }
                else:
                    stats_by_bucket[b] = {"n": 0, "mean": 0, "std": 0}
                    
            # Stats test (first non-empty vs last non-empty)
            valid_buckets = [b for b in buckets if stats_by_bucket[b]["n"] > 5]
            if len(valid_buckets) >= 2:
                first_b = valid_buckets[0]
                last_b = valid_buckets[-1]
                
                vals_first = bucket_vals[first_b]
                vals_last = bucket_vals[last_b]
                
                u_stat, p_val = stats.mannwhitneyu(vals_first, vals_last, alternative='two-sided')
                d = cohens_d(vals_last, vals_first) # Last vs First
                
                if p_val < 0.05:
                    summary_counts[metric] += 1
                    
                player_res["metrics"][metric] = {
                    "buckets": stats_by_bucket,
                    "test": {
                        "first_bucket": first_b,
                        "last_bucket": last_b,
                        "p_value": p_val,
                        "cohens_d": d
                    }
                }
                print(f"  {metric}: p={p_val:.4f}, d={d:.2f} (n={len(vals_first)} vs n={len(vals_last)})")
            else:
                player_res["metrics"][metric] = {"buckets": stats_by_bucket, "test": None}
                
        results[p_key] = player_res
        
        # Plot timeline
        fig, axs = plt.subplots(len(METRICS), 1, figsize=(10, 2*len(METRICS)), sharex=True)
        if len(METRICS) == 1: axs = [axs]
        
        for i, metric in enumerate(METRICS):
            means = [player_res["metrics"][metric]["buckets"][b]["mean"] for b in buckets]
            stds = [player_res["metrics"][metric]["buckets"][b]["std"] for b in buckets]
            ns = [player_res["metrics"][metric]["buckets"][b]["n"] for b in buckets]
            
            x_pos = np.arange(len(buckets))
            
            # Plot only buckets with data
            valid_x = [x for x, n in zip(x_pos, ns) if n > 0]
            valid_means = [means[x] for x in valid_x]
            valid_stds = [stds[x] for x in valid_x]
            
            axs[i].errorbar(valid_x, valid_means, yerr=valid_stds, fmt='-o', color='#34d399', ecolor='#1e293b', capsize=4)
            axs[i].set_ylabel(metric.replace('_', ' ').title(), color='#94a3b8')
            axs[i].grid(True, linestyle='--', alpha=0.3, color='#1e293b')
            axs[i].set_facecolor('#0f172a')
            
        axs[-1].set_xticks(np.arange(len(buckets)))
        axs[-1].set_xticklabels(buckets)
        axs[-1].set_xlabel("Match Minute Bucket", color='#94a3b8')
        
        fig.patch.set_facecolor('#0f172a')
        fig.suptitle(f"Track {p_data['track_id']} Gait Evolution", color='#f8fafc')
        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"track_{p_data['track_id']}_timeline.png")
        plt.savefig(plot_path, dpi=120)
        plt.close()
        
    print("\n=== Match Summary ===")
    print(f"Players with >= {MIN_TRACK_MINUTES}m tracked: {valid_players}")
    print(f"Players excluded (insufficient data): {insufficient_players}")
    print("Players showing significant change (p < 0.05):")
    for m in METRICS:
        print(f"  {m}: {summary_counts[m]}/{valid_players}")
        
    out_json = os.path.join(out_dir, "timeline_analysis.json")
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_json}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=str, default="soccernet_processed/SNMOT-060_players.json")
    args = parser.parse_args()
    
    analyze_player_timeline(args.json)

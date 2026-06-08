"""
Step 5: Injury Validation — Track a specific player and analyze biomechanical
signals leading up to a known injury moment.

This is the KEY experiment: can your pipeline detect anomalies BEFORE
the injury actually occurs?

Usage:
    python 05_injury_validation.py output/match_keypoints.json --injury-minute 63 --player-pos 520,380

    --injury-minute: The match minute when the injury occurred
    --player-pos: Starting bbox center of the target player (x,y from first frame)
    --pre-window: Minutes before injury to flag as "pre-injury zone" (default: 15)
"""

import argparse
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import stats


def load_results(path):
    with open(path) as f:
        return json.load(f)


def compute_angle(p1, p2, p3):
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def asymmetry_index(left, right):
    mean_val = (left + right) / 2
    if mean_val < 1e-6:
        return 0.0
    return abs(left - right) / mean_val


def extract_biomechanics(person):
    kp = person['keypoints']
    scores = person['keypoint_scores']

    lower_indices = [11, 12, 13, 14, 15, 16]
    lower_confidence = np.mean([scores[i] for i in lower_indices])
    if lower_confidence < 0.3:
        return None

    left_knee = compute_angle(kp[11], kp[13], kp[15])
    right_knee = compute_angle(kp[12], kp[14], kp[16])
    left_hip = compute_angle(kp[5], kp[11], kp[13])
    right_hip = compute_angle(kp[6], kp[12], kp[14])

    def lateral_displacement(hip, knee, ankle):
        hip, knee, ankle = np.array(hip), np.array(knee), np.array(ankle)
        line = ankle - hip
        line_len = np.linalg.norm(line) + 1e-8
        line_unit = line / line_len
        proj = hip + np.dot(knee - hip, line_unit) * line_unit
        return np.linalg.norm(knee - proj)

    left_valgus = lateral_displacement(kp[11], kp[13], kp[15])
    right_valgus = lateral_displacement(kp[12], kp[14], kp[16])

    stride_width = np.linalg.norm(np.array(kp[15]) - np.array(kp[16]))
    hip_drop = abs(kp[11][1] - kp[12][1])

    # Stride length proxy (horizontal distance between ankles)
    stride_length = abs(kp[15][0] - kp[16][0])

    # Torso lean (angle of shoulder midpoint to hip midpoint vs vertical)
    shoulder_mid = [(kp[5][0] + kp[6][0]) / 2, (kp[5][1] + kp[6][1]) / 2]
    hip_mid = [(kp[11][0] + kp[12][0]) / 2, (kp[11][1] + kp[12][1]) / 2]
    torso_angle = np.degrees(np.arctan2(
        abs(shoulder_mid[0] - hip_mid[0]),
        abs(shoulder_mid[1] - hip_mid[1]) + 1e-8
    ))

    return {
        'left_knee_angle': left_knee,
        'right_knee_angle': right_knee,
        'left_hip_angle': left_hip,
        'right_hip_angle': right_hip,
        'knee_asymmetry': asymmetry_index(left_knee, right_knee),
        'hip_asymmetry': asymmetry_index(left_hip, right_hip),
        'left_valgus': left_valgus,
        'right_valgus': right_valgus,
        'valgus_asymmetry': asymmetry_index(left_valgus, right_valgus),
        'stride_width': stride_width,
        'stride_length': stride_length,
        'hip_drop': hip_drop,
        'torso_lean': torso_angle,
        'lower_confidence': lower_confidence
    }


def track_player(results, start_pos, tolerance=80):
    timeline = []
    current_pos = list(start_pos)

    for frame in results:
        best_match = None
        best_dist = tolerance

        for person in frame['persons']:
            bbox = person['bbox']
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            dist = np.sqrt((cx - current_pos[0])**2 + (cy - current_pos[1])**2)

            if dist < best_dist:
                best_dist = dist
                best_match = person

        if best_match is not None:
            bio = extract_biomechanics(best_match)
            if bio is not None:
                bio['minute'] = frame['match_minute']
                bio['frame_idx'] = frame['frame_idx']
                timeline.append(bio)

                bbox = best_match['bbox']
                current_pos = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]

    return timeline


def auto_select_player(results):
    sample_indices = np.linspace(0, len(results)-1, min(100, len(results)), dtype=int)
    first_frame = results[sample_indices[0]]
    best_score = 0
    best_pos = None

    for person in first_frame['persons']:
        bbox = person['bbox']
        pos = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]
        current_pos = list(pos)
        hits = 0
        for idx in sample_indices[1:]:
            frame = results[idx]
            for p in frame['persons']:
                b = p['bbox']
                cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
                if np.sqrt((cx-current_pos[0])**2 + (cy-current_pos[1])**2) < 100:
                    hits += 1
                    current_pos = [cx, cy]
                    break
        if hits > best_score:
            best_score = hits
            best_pos = pos

    if best_pos is None and first_frame['persons']:
        bbox = first_frame['persons'][0]['bbox']
        best_pos = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]

    return best_pos


def compute_rolling_zscore(values, window=50):
    """Compute rolling z-score: how many std devs from the rolling mean."""
    zscores = np.zeros(len(values))
    for i in range(window, len(values)):
        window_vals = values[i-window:i]
        mean = np.mean(window_vals)
        std = np.std(window_vals)
        if std > 1e-8:
            zscores[i] = (values[i] - mean) / std
    return zscores


def plot_injury_analysis(timeline, injury_minute, pre_window, output_prefix, player_name):
    if len(timeline) < 20:
        print(f"ERROR: Only {len(timeline)} data points.")
        return

    minutes = np.array([t['minute'] for t in timeline])
    knee_asym = np.array([t['knee_asymmetry'] for t in timeline])
    hip_asym = np.array([t['hip_asymmetry'] for t in timeline])
    valgus_asym = np.array([t['valgus_asymmetry'] for t in timeline])
    hip_drop = np.array([t['hip_drop'] for t in timeline])
    stride_length = np.array([t['stride_length'] for t in timeline])
    torso_lean = np.array([t['torso_lean'] for t in timeline])
    confidence = np.array([t['lower_confidence'] for t in timeline])

    # Smooth
    win = min(51, len(knee_asym) // 4 * 2 + 1)
    if win < 5:
        win = 5
    if len(knee_asym) <= win:
        print("Not enough data points for smoothing")
        return

    knee_smooth = savgol_filter(knee_asym, win, 3)
    hip_smooth = savgol_filter(hip_asym, win, 3)
    valgus_smooth = savgol_filter(valgus_asym, win, 3)
    stride_smooth = savgol_filter(stride_length, win, 3)

    # Rolling z-scores (anomaly detection)
    knee_zscore = compute_rolling_zscore(knee_asym, window=min(50, len(knee_asym)//4))
    hip_zscore = compute_rolling_zscore(hip_asym, window=min(50, len(hip_asym)//4))
    valgus_zscore = compute_rolling_zscore(valgus_asym, window=min(50, len(valgus_asym)//4))

    # Combined anomaly score
    anomaly_score = np.sqrt(knee_zscore**2 + hip_zscore**2 + valgus_zscore**2)
    anomaly_smooth = savgol_filter(anomaly_score, win, 3) if len(anomaly_score) > win else anomaly_score

    # Pre-injury zone
    pre_injury_start = injury_minute - pre_window

    # ========== MAIN PLOT ==========
    fig, axes = plt.subplots(6, 1, figsize=(16, 28), sharex=True)
    fig.suptitle(f'INJURY VALIDATION: {player_name}\n'
                 f'Injury at minute {injury_minute} | Pre-injury window: {pre_window} min',
                 fontsize=16, fontweight='bold')

    for ax in axes:
        # Injury moment
        ax.axvline(x=injury_minute, color='red', linewidth=2, linestyle='-', alpha=0.8)
        # Pre-injury zone
        ax.axvspan(pre_injury_start, injury_minute, alpha=0.15, color='red',
                   label=f'Pre-injury zone ({pre_window} min)')
        # Half-time
        ax.axvline(x=45, color='gray', linewidth=1, linestyle='--', alpha=0.4)

    # 1. Knee asymmetry
    axes[0].scatter(minutes, knee_asym, alpha=0.1, s=3, color='steelblue')
    axes[0].plot(minutes, knee_smooth, color='darkblue', linewidth=2)
    axes[0].set_ylabel('Knee Asymmetry')
    axes[0].set_title('Knee Gait Asymmetry')
    axes[0].legend(loc='upper left')

    # 2. Hip asymmetry
    axes[1].scatter(minutes, hip_asym, alpha=0.1, s=3, color='coral')
    axes[1].plot(minutes, hip_smooth, color='darkred', linewidth=2)
    axes[1].set_ylabel('Hip Asymmetry')
    axes[1].set_title('Hip Gait Asymmetry')

    # 3. Valgus asymmetry
    axes[2].scatter(minutes, valgus_asym, alpha=0.1, s=3, color='mediumpurple')
    axes[2].plot(minutes, valgus_smooth, color='indigo', linewidth=2)
    axes[2].set_ylabel('Valgus Asymmetry')
    axes[2].set_title('Knee Valgus Asymmetry')

    # 4. Stride length (drops with fatigue/guarding)
    axes[3].scatter(minutes, stride_length, alpha=0.1, s=3, color='teal')
    axes[3].plot(minutes, stride_smooth, color='darkgreen', linewidth=2)
    axes[3].set_ylabel('Stride Length (px)')
    axes[3].set_title('Stride Length (decreases with fatigue/injury guarding)')

    # 5. ANOMALY SCORE — the money plot
    axes[4].plot(minutes, anomaly_smooth, color='black', linewidth=2.5, label='Combined anomaly score')
    axes[4].fill_between(minutes, 0, anomaly_smooth, alpha=0.3, color='orange')
    # Threshold line
    anomaly_threshold = np.mean(anomaly_score) + 2 * np.std(anomaly_score)
    axes[4].axhline(y=anomaly_threshold, color='red', linestyle=':', alpha=0.7,
                    label=f'Anomaly threshold (mean + 2 std = {anomaly_threshold:.2f})')
    axes[4].set_ylabel('Anomaly Score')
    axes[4].set_title('COMBINED ANOMALY SCORE (z-score magnitude across all metrics)')
    axes[4].legend(loc='upper left')

    # 6. Confidence
    axes[5].scatter(minutes, confidence, alpha=0.2, s=3, color='green')
    axes[5].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    axes[5].set_ylabel('Pose Confidence')
    axes[5].set_xlabel('Match Minute')
    axes[5].set_title('Data Quality')

    plt.tight_layout()
    plot_path = f'{output_prefix}_injury_validation.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot saved: {plot_path}")

    return anomaly_score, anomaly_threshold, minutes


def compute_injury_statistics(timeline, injury_minute, pre_window):
    """The critical statistical analysis: was the pre-injury window anomalous?"""

    # Split into zones
    baseline = [t for t in timeline if t['minute'] < injury_minute - pre_window]
    pre_injury = [t for t in timeline if injury_minute - pre_window <= t['minute'] < injury_minute]
    post_injury = [t for t in timeline if t['minute'] >= injury_minute]

    if len(baseline) < 10 or len(pre_injury) < 5:
        print("ERROR: Not enough data in baseline or pre-injury window.")
        print(f"  Baseline: {len(baseline)} points, Pre-injury: {len(pre_injury)} points")
        return

    print(f"\n{'='*70}")
    print(f"INJURY VALIDATION RESULTS")
    print(f"{'='*70}")
    print(f"Injury minute:        {injury_minute}")
    print(f"Pre-injury window:    {pre_window} min (minute {injury_minute - pre_window} to {injury_minute})")
    print(f"Baseline points:      {len(baseline)} (minute 0 to {injury_minute - pre_window})")
    print(f"Pre-injury points:    {len(pre_injury)}")
    print(f"Post-injury points:   {len(post_injury)}")

    metrics = [
        ('Knee Asymmetry', 'knee_asymmetry'),
        ('Hip Asymmetry', 'hip_asymmetry'),
        ('Valgus Asymmetry', 'valgus_asymmetry'),
        ('Hip Drop', 'hip_drop'),
        ('Stride Length', 'stride_length'),
        ('Torso Lean', 'torso_lean'),
    ]

    print(f"\n{'Metric':<22} {'Baseline':>10} {'Pre-Injury':>12} {'Change':>10} {'p-value':>10} {'Signal?':>10}")
    print("-" * 75)

    signals_detected = 0
    total_metrics = 0

    for name, key in metrics:
        baseline_vals = [t[key] for t in baseline]
        pre_vals = [t[key] for t in pre_injury]

        if len(baseline_vals) < 5 or len(pre_vals) < 5:
            continue

        total_metrics += 1
        b_mean = np.mean(baseline_vals)
        p_mean = np.mean(pre_vals)

        if b_mean > 0:
            change_pct = ((p_mean / b_mean) - 1) * 100
        else:
            change_pct = 0

        t_stat, p_value = stats.ttest_ind(baseline_vals, pre_vals, equal_var=False)

        pooled_std = np.sqrt((np.std(baseline_vals)**2 + np.std(pre_vals)**2) / 2)
        cohens_d = (p_mean - b_mean) / pooled_std if pooled_std > 0 else 0

        is_signal = p_value < 0.05 and abs(cohens_d) > 0.2
        if is_signal:
            signals_detected += 1

        flag = "** YES **" if is_signal else "no"
        print(f"{name:<22} {b_mean:>10.4f} {p_mean:>12.4f} {change_pct:>+9.1f}% {p_value:>10.4f} {flag:>10}")

    # Overall pre-injury trend (did anomaly increase leading into injury?)
    pre_minutes = [t['minute'] for t in pre_injury]
    pre_knee = [t['knee_asymmetry'] for t in pre_injury]
    pre_hip = [t['hip_asymmetry'] for t in pre_injury]

    print(f"\n{'='*70}")
    print("PRE-INJURY TREND (within the pre-injury window)")
    print(f"{'='*70}")

    for name, vals in [('Knee asym', pre_knee), ('Hip asym', pre_hip)]:
        if len(vals) > 5:
            rho, p = stats.spearmanr(pre_minutes, vals)
            trend = "INCREASING" if rho > 0.1 and p < 0.1 else "FLAT" if abs(rho) < 0.1 else "DECREASING"
            print(f"  {name}: rho={rho:.3f}, p={p:.4f} -> {trend}")

    # Final verdict
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    if signals_detected >= 3:
        print(f"  STRONG SIGNAL: {signals_detected}/{total_metrics} metrics showed significant")
        print(f"  pre-injury anomalies compared to baseline.")
        print(f"  >> The model detected biomechanical degradation before the injury.")
        print(f"  >> THIS VALIDATES THE CORE THESIS.")
    elif signals_detected >= 1:
        print(f"  WEAK SIGNAL: {signals_detected}/{total_metrics} metrics showed anomalies.")
        print(f"  >> Some degradation detected but not conclusive.")
        print(f"  >> Test on more injury events before drawing conclusions.")
    else:
        print(f"  NO SIGNAL: {signals_detected}/{total_metrics} metrics were anomalous.")
        print(f"  >> Pre-injury window looked similar to baseline.")
        print(f"  >> Possible reasons:")
        print(f"     - Injury was truly sudden (no biomechanical precursor)")
        print(f"     - Signal too noisy at broadcast resolution")
        print(f"     - Wrong pre-injury window (try longer/shorter)")
        print(f"     - Player tracking lost the target during key moments")

    # Comparison with post-injury (if player continued/limped)
    if len(post_injury) > 5:
        print(f"\n{'='*70}")
        print("POST-INJURY CHECK (did metrics change after the injury moment?)")
        print(f"{'='*70}")
        for name, key in [('Knee Asymmetry', 'knee_asymmetry'), ('Hip Asymmetry', 'hip_asymmetry')]:
            pre_vals = [t[key] for t in pre_injury]
            post_vals = [t[key] for t in post_injury]
            if len(post_vals) >= 3:
                t_stat, p_value = stats.ttest_ind(pre_vals, post_vals, equal_var=False)
                print(f"  {name}: pre={np.mean(pre_vals):.4f}, post={np.mean(post_vals):.4f}, p={p_value:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Validate injury prediction on a known injury event')
    parser.add_argument('keypoints', help='Path to keypoints JSON from step 1')
    parser.add_argument('--injury-minute', type=float, required=True,
                        help='Match minute when the injury occurred')
    parser.add_argument('--player-pos', type=str, default=None,
                        help='Starting player bbox center as x,y (e.g., 640,400)')
    parser.add_argument('--player-name', type=str, default='Target Player',
                        help='Player name for plot labels')
    parser.add_argument('--pre-window', type=float, default=15,
                        help='Minutes before injury to analyze (default: 15)')
    parser.add_argument('--tolerance', type=int, default=80,
                        help='Tracking tolerance in pixels (default: 80)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output prefix for plots and data')
    args = parser.parse_args()

    print(f"Loading keypoint data from {args.keypoints}...")
    results = load_results(args.keypoints)
    print(f"Loaded {len(results)} frames")

    if args.player_pos:
        x, y = args.player_pos.split(',')
        start_pos = [float(x), float(y)]
        print(f"Targeting player at position ({start_pos[0]}, {start_pos[1]})")
    else:
        print("Auto-selecting most visible player...")
        start_pos = auto_select_player(results)
        print(f"Selected position: ({start_pos[0]:.0f}, {start_pos[1]:.0f})")

    print(f"Tracking player across {len(results)} frames...")
    timeline = track_player(results, start_pos, tolerance=args.tolerance)
    print(f"Got {len(timeline)} valid biomechanical measurements")

    if len(timeline) < 20:
        print("ERROR: Too few measurements. Try different --player-pos or larger --tolerance")
        sys.exit(1)

    # Check we have data around the injury minute
    minutes = [t['minute'] for t in timeline]
    if max(minutes) < args.injury_minute:
        print(f"WARNING: Data only goes to minute {max(minutes):.1f}, "
              f"but injury is at minute {args.injury_minute}")

    prefix = args.output or args.keypoints.replace('.json', '')

    # Save timeline
    timeline_path = f'{prefix}_injury_timeline.json'
    with open(timeline_path, 'w') as f:
        json.dump(timeline, f, indent=2)
    print(f"Timeline saved: {timeline_path}")

    # Plot
    anomaly_score, threshold, mins = plot_injury_analysis(
        timeline, args.injury_minute, args.pre_window,
        prefix, args.player_name
    )

    # Statistics
    compute_injury_statistics(timeline, args.injury_minute, args.pre_window)

    # Summary of anomaly score near injury
    if anomaly_score is not None:
        pre_injury_mask = (mins >= args.injury_minute - args.pre_window) & (mins < args.injury_minute)
        pre_injury_anomaly = anomaly_score[pre_injury_mask]
        if len(pre_injury_anomaly) > 0:
            breaches = np.sum(pre_injury_anomaly > threshold)
            print(f"\n  Anomaly threshold breaches in pre-injury window: "
                  f"{breaches}/{len(pre_injury_anomaly)} frames "
                  f"({breaches/len(pre_injury_anomaly)*100:.0f}%)")


if __name__ == '__main__':
    main()

"""
Step 2: Analyze extracted keypoints for gait asymmetry and fatigue signals.
This is THE experiment — does the biomechanical signal survive broadcast video?

Usage:
    python 02_analyze_gait.py match_001_keypoints.json [--player-pos 640,400]

If --player-pos is not given, it will auto-select the most consistently
tracked player (most detections across the match).
"""

import argparse
import json
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy import stats


def load_results(path):
    with open(path) as f:
        return json.load(f)


def compute_angle(p1, p2, p3):
    """Angle at p2 formed by p1-p2-p3, in degrees."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))


def asymmetry_index(left, right):
    """0 = symmetric, higher = more asymmetric."""
    mean_val = (left + right) / 2
    if mean_val < 1e-6:
        return 0.0
    return abs(left - right) / mean_val


def extract_biomechanics(person):
    """Extract all relevant joint angles and metrics from one person detection."""
    kp = person['keypoints']
    scores = person['keypoint_scores']

    # COCO keypoint indices
    # 5:left_shoulder, 6:right_shoulder, 11:left_hip, 12:right_hip,
    # 13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle

    # Check lower body confidence
    lower_indices = [11, 12, 13, 14, 15, 16]
    lower_confidence = np.mean([scores[i] for i in lower_indices])
    if lower_confidence < 0.3:
        return None

    # Knee angles
    left_knee = compute_angle(kp[11], kp[13], kp[15])
    right_knee = compute_angle(kp[12], kp[14], kp[16])

    # Hip angles
    left_hip = compute_angle(kp[5], kp[11], kp[13])
    right_hip = compute_angle(kp[6], kp[12], kp[14])

    # Ankle-knee-hip alignment (rough valgus/varus proxy)
    # Lateral displacement of knee relative to hip-ankle line
    def lateral_displacement(hip, knee, ankle):
        hip, knee, ankle = np.array(hip), np.array(knee), np.array(ankle)
        line = ankle - hip
        line_len = np.linalg.norm(line) + 1e-8
        line_unit = line / line_len
        proj = hip + np.dot(knee - hip, line_unit) * line_unit
        return np.linalg.norm(knee - proj)

    left_valgus = lateral_displacement(kp[11], kp[13], kp[15])
    right_valgus = lateral_displacement(kp[12], kp[14], kp[16])

    # Stride width proxy (distance between ankles)
    stride_width = np.linalg.norm(np.array(kp[15]) - np.array(kp[16]))

    # Hip drop proxy (vertical difference between hips)
    hip_drop = abs(kp[11][1] - kp[12][1])

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
        'hip_drop': hip_drop,
        'lower_confidence': lower_confidence
    }


def track_player_by_position(results, start_pos, tolerance=80):
    """Simple positional tracking — follow a player by bbox center proximity."""
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
                timeline.append(bio)

                bbox = best_match['bbox']
                current_pos = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]

    return timeline


def auto_select_player(results):
    """Find the most consistently visible player across the match."""
    # Sample frames evenly across the match
    sample_indices = np.linspace(0, len(results)-1, min(100, len(results)), dtype=int)

    # For each detection in first sampled frame, count how many subsequent
    # frames have a detection nearby
    first_frame = results[sample_indices[0]]
    best_score = 0
    best_pos = None

    for person in first_frame['persons']:
        bbox = person['bbox']
        pos = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]

        # Track this person across sampled frames
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

    print(f"Auto-selected player at position ({best_pos[0]:.0f}, {best_pos[1]:.0f})")
    print(f"Tracking confidence: {best_score}/{len(sample_indices)} sampled frames")
    return best_pos


def plot_analysis(timeline, output_prefix):
    """Generate the key fatigue analysis plots."""
    if len(timeline) < 20:
        print(f"ERROR: Only {len(timeline)} data points. Need at least 20. "
              "Try adjusting player position or tolerance.")
        return

    minutes = [t['minute'] for t in timeline]
    knee_asym = [t['knee_asymmetry'] for t in timeline]
    hip_asym = [t['hip_asymmetry'] for t in timeline]
    valgus_asym = [t['valgus_asymmetry'] for t in timeline]
    confidence = [t['lower_confidence'] for t in timeline]

    # Smooth
    win = min(51, len(knee_asym) // 4 * 2 + 1)
    if win >= 5:
        knee_smooth = savgol_filter(knee_asym, win, 3)
        hip_smooth = savgol_filter(hip_asym, win, 3)
        valgus_smooth = savgol_filter(valgus_asym, win, 3)
    else:
        knee_smooth = knee_asym
        hip_smooth = hip_asym
        valgus_smooth = valgus_asym

    fig, axes = plt.subplots(5, 1, figsize=(14, 20), sharex=True)
    fig.suptitle('Biomechanical Fatigue Analysis — Signal Validation Experiment', fontsize=14)

    # 1. Knee asymmetry
    axes[0].scatter(minutes, knee_asym, alpha=0.12, s=3, color='steelblue')
    axes[0].plot(minutes, knee_smooth, color='darkblue', linewidth=2)
    axes[0].set_ylabel('Knee Asymmetry')
    axes[0].set_title('Knee Gait Asymmetry')

    # 2. Hip asymmetry
    axes[1].scatter(minutes, hip_asym, alpha=0.12, s=3, color='coral')
    axes[1].plot(minutes, hip_smooth, color='darkred', linewidth=2)
    axes[1].set_ylabel('Hip Asymmetry')
    axes[1].set_title('Hip Gait Asymmetry')

    # 3. Valgus asymmetry
    axes[2].scatter(minutes, valgus_asym, alpha=0.12, s=3, color='mediumpurple')
    axes[2].plot(minutes, valgus_smooth, color='indigo', linewidth=2)
    axes[2].set_ylabel('Valgus Asymmetry')
    axes[2].set_title('Knee Valgus Asymmetry (Lateral Displacement)')

    # 4. Combined overlay
    axes[3].plot(minutes, knee_smooth, color='darkblue', linewidth=2, label='Knee')
    axes[3].plot(minutes, hip_smooth, color='darkred', linewidth=2, label='Hip')
    axes[3].plot(minutes, valgus_smooth, color='indigo', linewidth=2, label='Valgus')
    axes[3].axvline(x=45, color='gray', linestyle='--', alpha=0.5, label='Half-time')
    axes[3].set_ylabel('Asymmetry Index')
    axes[3].legend()
    axes[3].set_title('Combined Trends')

    # 5. Confidence
    axes[4].scatter(minutes, confidence, alpha=0.2, s=3, color='green')
    axes[4].axhline(y=0.5, color='red', linestyle='--', alpha=0.5)
    axes[4].set_ylabel('Pose Confidence')
    axes[4].set_xlabel('Match Minute')
    axes[4].set_title('Lower Body Pose Confidence (Data Quality)')

    plt.tight_layout()
    plot_path = f'{output_prefix}_fatigue_plot.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Plot saved: {plot_path}")

    return knee_smooth, hip_smooth, valgus_smooth, minutes


def compute_statistics(timeline):
    """Compute the key statistical tests."""
    if len(timeline) < 20:
        return

    # Split into first half vs second half of data
    mid = len(timeline) // 2
    first = timeline[:mid]
    second = timeline[mid:]

    print(f"\n{'='*60}")
    print("SIGNAL VALIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Total data points: {len(timeline)}")
    print(f"Time range: {timeline[0]['minute']:.1f} - {timeline[-1]['minute']:.1f} min")

    metrics = [
        ('Knee Asymmetry', 'knee_asymmetry'),
        ('Hip Asymmetry', 'hip_asymmetry'),
        ('Valgus Asymmetry', 'valgus_asymmetry'),
    ]

    signal_found = False

    for name, key in metrics:
        first_vals = [t[key] for t in first]
        second_vals = [t[key] for t in second]

        first_mean = np.mean(first_vals)
        second_mean = np.mean(second_vals)
        change_pct = ((second_mean / first_mean) - 1) * 100 if first_mean > 0 else 0

        # Statistical test: are the distributions different?
        t_stat, p_value = stats.ttest_ind(first_vals, second_vals)

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((np.std(first_vals)**2 + np.std(second_vals)**2) / 2)
        cohens_d = (second_mean - first_mean) / pooled_std if pooled_std > 0 else 0

        print(f"\n{name}:")
        print(f"  First half mean:  {first_mean:.4f} (std: {np.std(first_vals):.4f})")
        print(f"  Second half mean: {second_mean:.4f} (std: {np.std(second_vals):.4f})")
        print(f"  Change: {change_pct:+.1f}%")
        print(f"  t-statistic: {t_stat:.3f}, p-value: {p_value:.4f}")
        print(f"  Cohen's d: {cohens_d:.3f}", end="")

        if abs(cohens_d) > 0.5:
            print(" (MEDIUM-LARGE EFFECT)")
            signal_found = True
        elif abs(cohens_d) > 0.2:
            print(" (SMALL-MEDIUM EFFECT)")
            signal_found = True
        else:
            print(" (NEGLIGIBLE EFFECT)")

        if p_value < 0.05:
            print(f"  ** STATISTICALLY SIGNIFICANT (p < 0.05) **")

    # Trend analysis: is there a monotonic trend over time?
    print(f"\n{'='*60}")
    print("TREND ANALYSIS (Spearman correlation with match minute)")
    print(f"{'='*60}")

    minutes = [t['minute'] for t in timeline]
    for name, key in metrics:
        values = [t[key] for t in timeline]
        rho, p_val = stats.spearmanr(minutes, values)
        print(f"  {name}: rho={rho:.3f}, p={p_val:.4f}", end="")
        if p_val < 0.05 and abs(rho) > 0.1:
            print(f" ** SIGNIFICANT TREND **")
            signal_found = True
        else:
            print()

    # Verdict
    print(f"\n{'='*60}")
    if signal_found:
        print("VERDICT: SIGNAL DETECTED")
        print("Meaningful asymmetry differences found between match halves")
        print("or significant trends over match duration.")
        print(">> PROCEED to multi-match validation (Step 3)")
    else:
        print("VERDICT: NO CLEAR SIGNAL")
        print("No statistically significant fatigue-driven asymmetry changes")
        print("detected at this resolution.")
        print(">> Consider: higher fps, different player, or club tactical cameras")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Analyze gait asymmetry from keypoint data')
    parser.add_argument('keypoints', help='Path to keypoints JSON from step 1')
    parser.add_argument('--player-pos', type=str, default=None,
                        help='Starting player bbox center as x,y (e.g., 640,400). Auto-detects if not given.')
    parser.add_argument('--tolerance', type=int, default=80,
                        help='Tracking tolerance in pixels (default: 80)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output prefix for plots and data')
    args = parser.parse_args()

    print("Loading keypoint data...")
    results = load_results(args.keypoints)
    print(f"Loaded {len(results)} frames")

    # Determine player position
    if args.player_pos:
        x, y = args.player_pos.split(',')
        start_pos = [float(x), float(y)]
        print(f"Using manual player position: ({start_pos[0]}, {start_pos[1]})")
    else:
        print("Auto-selecting most visible player...")
        start_pos = auto_select_player(results)

    # Track and extract biomechanics
    print(f"\nTracking player across {len(results)} frames...")
    timeline = track_player_by_position(results, start_pos, tolerance=args.tolerance)
    print(f"Got {len(timeline)} valid biomechanical measurements")

    if len(timeline) < 20:
        print("ERROR: Too few measurements. Try:")
        print("  1. Different --player-pos")
        print("  2. Larger --tolerance")
        print("  3. Different match video")
        sys.exit(1)

    # Output prefix
    prefix = args.output or args.keypoints.replace('.json', '')

    # Save timeline
    timeline_path = f'{prefix}_timeline.json'
    with open(timeline_path, 'w') as f:
        json.dump(timeline, f, indent=2)
    print(f"Timeline saved: {timeline_path}")

    # Plot
    plot_analysis(timeline, prefix)

    # Statistics
    compute_statistics(timeline)


if __name__ == '__main__':
    main()

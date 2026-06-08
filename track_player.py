#!/usr/bin/env python3
"""
Enhanced player tracker with jersey-color based team identification
and bbox-size + position heuristics for targeting a specific player.

Usage:
  python3 track_player.py videos/match_001.mp4 --team barca --jersey 10 --fps 5
"""
import cv2
import json
import numpy as np
import argparse
import os
import time
from collections import defaultdict
from ultralytics import YOLO

# COCO keypoint indices
KP = {
    'nose': 0, 'l_eye': 1, 'r_eye': 2, 'l_ear': 3, 'r_ear': 4,
    'l_shoulder': 5, 'r_shoulder': 6, 'l_elbow': 7, 'r_elbow': 8,
    'l_wrist': 9, 'r_wrist': 10, 'l_hip': 11, 'r_hip': 12,
    'l_knee': 13, 'r_knee': 14, 'l_ankle': 15, 'r_ankle': 16,
}

def angle_3pt(a, b, c):
    """Angle at vertex b formed by points a-b-c, in degrees."""
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos_a, -1, 1)))

def extract_jersey_color(frame, bbox):
    """Extract dominant jersey color from the torso region of a bounding box."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h = y2 - y1
    w = x2 - x1
    # Torso region: middle 60% height, center 40% width
    ty1 = y1 + int(h * 0.2)
    ty2 = y1 + int(h * 0.5)
    tx1 = x1 + int(w * 0.3)
    tx2 = x1 + int(w * 0.7)
    ty1 = max(0, ty1); ty2 = min(frame.shape[0], ty2)
    tx1 = max(0, tx1); tx2 = min(frame.shape[1], tx2)
    if ty2 <= ty1 or tx2 <= tx1:
        return None, None
    torso = frame[ty1:ty2, tx1:tx2]
    hsv = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)
    return hsv, (tx1, ty1, tx2, ty2)

def classify_team(hsv_torso):
    """Classify team based on jersey color in HSV space.
    Barcelona: blue/red (garnet) - Hue ~0-10 or ~160-180 (red) or ~100-130 (blue)
    Real Madrid: white - low saturation, high value
    Referee: yellow/green - Hue ~25-45
    Returns: 'barca', 'madrid', 'referee', or 'unknown'
    """
    if hsv_torso is None or hsv_torso.size == 0:
        return 'unknown'
    avg_h = np.mean(hsv_torso[:,:,0])
    avg_s = np.mean(hsv_torso[:,:,1])
    avg_v = np.mean(hsv_torso[:,:,2])

    # White jersey (Real Madrid): low saturation, high value
    if avg_s < 50 and avg_v > 150:
        return 'madrid'
    # Yellow (referee): hue 20-40, high saturation
    if 20 < avg_h < 45 and avg_s > 80:
        return 'referee'
    # Blue (Barca away or home): hue 100-130
    if 95 < avg_h < 135 and avg_s > 50:
        return 'barca'
    # Red/garnet (Barca): hue 0-10 or 160-180
    if (avg_h < 12 or avg_h > 155) and avg_s > 60:
        return 'barca'
    # Dark blue/purple
    if 130 < avg_h < 160 and avg_s > 40:
        return 'barca'
    return 'unknown'

def compute_biomechanics(kps, confs):
    """Compute biomechanical metrics from COCO keypoints."""
    def kp(name):
        idx = KP[name]
        return kps[idx], confs[idx]

    # Get all lower body keypoints
    lhip, lhip_c = kp('l_hip')
    rhip, rhip_c = kp('r_hip')
    lknee, lknee_c = kp('l_knee')
    rknee, rknee_c = kp('r_knee')
    lankle, lankle_c = kp('l_ankle')
    rankle, rankle_c = kp('r_ankle')
    lshoulder, lshoulder_c = kp('l_shoulder')
    rshoulder, rshoulder_c = kp('r_shoulder')

    lower_conf = min(lhip_c, rhip_c, lknee_c, rknee_c, lankle_c, rankle_c)
    if lower_conf < 0.3:
        return None

    # Joint angles
    lk_angle = angle_3pt(lhip, lknee, lankle)
    rk_angle = angle_3pt(rhip, rknee, rankle)
    lh_angle = angle_3pt(lshoulder, lhip, lknee)
    rh_angle = angle_3pt(rshoulder, rhip, rknee)

    # Asymmetry indices
    knee_asym = abs(lk_angle - rk_angle) / ((lk_angle + rk_angle) / 2 + 1e-9)
    hip_asym = abs(lh_angle - rh_angle) / ((lh_angle + rh_angle) / 2 + 1e-9)

    # Valgus proxy: lateral knee displacement relative to hip-ankle line
    l_valgus = abs(lknee[0] - (lhip[0] + lankle[0]) / 2)
    r_valgus = abs(rknee[0] - (rhip[0] + rankle[0]) / 2)
    valgus_asym = abs(l_valgus - r_valgus) / (max(l_valgus, r_valgus) + 1e-9)

    # Stride width and hip drop
    stride_w = np.linalg.norm(np.array(lankle) - np.array(rankle))
    hip_drop = abs(lhip[1] - rhip[1])

    return {
        'left_knee_angle': round(lk_angle, 2),
        'right_knee_angle': round(rk_angle, 2),
        'left_hip_angle': round(lh_angle, 2),
        'right_hip_angle': round(rh_angle, 2),
        'knee_asymmetry': round(knee_asym, 4),
        'hip_asymmetry': round(hip_asym, 4),
        'left_valgus': round(l_valgus, 2),
        'right_valgus': round(r_valgus, 2),
        'valgus_asymmetry': round(valgus_asym, 4),
        'stride_width': round(stride_w, 2),
        'hip_drop': round(hip_drop, 2),
        'lower_confidence': round(float(lower_conf), 4),
    }


class PlayerTracker:
    """Track a specific player across frames using appearance + position."""
    def __init__(self, target_team='barca', max_gap=30):
        self.target_team = target_team
        self.max_gap = max_gap  # max frames without detection before resetting
        self.tracks = {}  # track_id -> track info
        self.next_id = 0
        self.frame_count = 0

    def _bbox_center(self, bbox):
        return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)

    def _bbox_area(self, bbox):
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    def _iou(self, b1, b2):
        """Intersection over Union."""
        x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
        x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = self._bbox_area(b1); a2 = self._bbox_area(b2)
        return inter / (a1 + a2 - inter + 1e-9)

    def update(self, frame, detections, frame_idx):
        """Match detections to existing tracks and classify teams.
        Returns list of (track_id, team, bbox, keypoints, confidences)"""
        self.frame_count += 1
        results = []

        for det in detections:
            bbox = det['bbox']
            kps = det['keypoints']
            confs = det['confidences']
            score = det['score']

            # Classify team
            hsv, _ = extract_jersey_color(frame, bbox)
            team = classify_team(hsv)

            # Try to match to existing track
            best_track = None
            best_score = 0
            center = self._bbox_center(bbox)

            for tid, track in self.tracks.items():
                if self.frame_count - track['last_seen'] > self.max_gap:
                    continue
                # Spatial distance
                prev_center = track['center']
                dist = np.sqrt((center[0]-prev_center[0])**2 + (center[1]-prev_center[1])**2)
                max_dist = 200  # pixels
                if dist > max_dist:
                    continue
                # Team match bonus
                team_bonus = 0.3 if team == track['team'] and team != 'unknown' else 0
                # IoU
                iou = self._iou(bbox, track['bbox'])
                # Combined score
                match_score = iou * 0.4 + (1 - dist/max_dist) * 0.4 + team_bonus + 0.1 * (team != 'unknown')
                if match_score > best_score:
                    best_score = match_score
                    best_track = tid

            if best_track is not None and best_score > 0.3:
                tid = best_track
            else:
                tid = self.next_id
                self.next_id += 1

            self.tracks[tid] = {
                'center': center, 'bbox': bbox, 'team': team,
                'last_seen': self.frame_count, 'area': self._bbox_area(bbox)
            }
            results.append((tid, team, bbox, kps, confs, score))

        # Prune old tracks
        stale = [t for t, info in self.tracks.items()
                 if self.frame_count - info['last_seen'] > self.max_gap * 2]
        for t in stale:
            del self.tracks[t]

        return results


def process_match(video_path, target_team, fps_sample=5, output_dir='output'):
    """Process match and extract per-player biomechanics."""
    os.makedirs(output_dir, exist_ok=True)

    # Load YOLO
    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using {device}")
    model = YOLO('yolo11m-pose.pt')
    model.to(device)

    cap = cv2.VideoCapture(video_path)
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    skip = max(1, int(vid_fps / fps_sample))
    est_frames = total_frames // skip

    print(f"Video: {vid_fps}fps, {total_frames} frames, ~{total_frames/vid_fps/60:.0f} min")
    print(f"Sampling every {skip} frames ({fps_sample}fps), ~{est_frames} to process")
    print(f"Target team: {target_team}")

    tracker = PlayerTracker(target_team=target_team)

    # Per-player data storage
    player_data = defaultdict(lambda: {'team': 'unknown', 'readings': [], 'frame_count': 0})
    team_counts = defaultdict(int)

    frame_idx = 0
    processed = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        match_minute = round(frame_idx / vid_fps / 60, 2)

        # Run pose estimation
        results = model(frame, verbose=False)

        detections = []
        for r in results:
            if r.keypoints is None:
                continue
            boxes = r.boxes
            kps_data = r.keypoints

            for i in range(len(boxes)):
                bbox = boxes.xyxy[i].cpu().numpy().tolist()
                score = float(boxes.conf[i].cpu())
                kps = kps_data.xy[i].cpu().numpy().tolist()
                confs_arr = kps_data.conf[i].cpu().numpy().tolist() if kps_data.conf is not None else [0.5]*17

                if score < 0.4:
                    continue
                # Filter out very small detections (crowd, far away)
                area = (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])
                if area < 3000:
                    continue

                detections.append({
                    'bbox': bbox, 'score': score,
                    'keypoints': kps, 'confidences': confs_arr
                })

        # Track players
        tracked = tracker.update(frame, detections, frame_idx)

        for (tid, team, bbox, kps, confs, score) in tracked:
            team_counts[team] += 1
            player_data[tid]['team'] = team
            player_data[tid]['frame_count'] += 1

            # Only compute biomechanics for target team players
            if team == target_team or team == 'unknown':
                bio = compute_biomechanics(kps, confs)
                if bio is not None:
                    bio['minute'] = match_minute
                    bio['frame_idx'] = frame_idx
                    bio['track_id'] = tid
                    bio['bbox_center'] = list(tracker._bbox_center(bbox))
                    player_data[tid]['readings'].append(bio)

        processed += 1
        if processed % 50 == 0:
            elapsed = time.time() - start_time
            fps_rate = processed / elapsed
            remaining = (est_frames - processed) / fps_rate / 60
            # Find most tracked players
            top_players = sorted(player_data.items(), key=lambda x: len(x[1]['readings']), reverse=True)[:5]
            top_str = ', '.join([f"T{tid}({info['team']}:{len(info['readings'])})" for tid, info in top_players])
            print(f"  [{processed}/{est_frames}] Min {match_minute:.1f} | {fps_rate:.1f}fps | ~{remaining:.0f}min left | Teams: {dict(team_counts)} | Top: {top_str}")

        frame_idx += 1

    cap.release()
    elapsed = time.time() - start_time
    print(f"\nDone. {processed} frames in {elapsed/60:.1f} minutes.")

    # --- Post-process: identify the best candidate for each team ---
    print("\n=== PLAYER IDENTIFICATION SUMMARY ===")
    print(f"Team detection counts: {dict(team_counts)}")
    print(f"Total unique tracks: {len(player_data)}")

    # Group tracks by team and find the most consistently tracked player per team
    team_players = defaultdict(list)
    for tid, info in player_data.items():
        if len(info['readings']) >= 20:  # minimum threshold
            team_players[info['team']].append((tid, info))

    output = {}
    for team in ['barca', 'madrid']:
        players = team_players.get(team, [])
        # Sort by number of readings (most consistently tracked = likely most visible player)
        players.sort(key=lambda x: len(x[1]['readings']), reverse=True)

        print(f"\n--- {team.upper()} ---")
        for i, (tid, info) in enumerate(players[:8]):
            print(f"  Track {tid}: {len(info['readings'])} readings, {info['frame_count']} frames")

        # Save top players' data
        for rank, (tid, info) in enumerate(players[:5]):
            key = f"{team}_player_{rank+1}"
            output[key] = {
                'track_id': tid,
                'team': team,
                'total_readings': len(info['readings']),
                'timeline': info['readings']
            }

    # Save all player data
    out_file = os.path.join(output_dir, 'match_001_players.json')
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved player data to {out_file}")

    # Print recommendation
    print("\n=== RECOMMENDATION ===")
    barca_top = team_players.get('barca', [])
    if barca_top:
        best_tid, best_info = barca_top[0]
        print(f"Best Barcelona player candidate (most visible): Track {best_tid}")
        print(f"  Readings: {len(best_info['readings'])}")
        print(f"  Note: In broadcast footage, the most tracked player is often")
        print(f"  the one most involved in play — likely Messi in a Barça match.")
    else:
        print("No Barcelona players consistently tracked. Jersey colors may not match expected ranges.")

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Track specific player')
    parser.add_argument('video', help='Path to match video')
    parser.add_argument('--team', default='barca', choices=['barca', 'madrid', 'all'])
    parser.add_argument('--fps', type=int, default=5)
    parser.add_argument('--output', default='output')
    args = parser.parse_args()

    process_match(args.video, args.team, args.fps, args.output)

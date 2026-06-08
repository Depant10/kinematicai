"""
Step 1: Process a full match video — detect players and extract pose keypoints.
Uses YOLO11-pose (works out of the box with RTX 5070 + M4 Air).

Outputs a JSON file with per-frame, per-player keypoint data.
Keypoints follow COCO format (17 keypoints):
  0:nose, 1:left_eye, 2:right_eye, 3:left_ear, 4:right_ear,
  5:left_shoulder, 6:right_shoulder, 7:left_elbow, 8:right_elbow,
  9:left_wrist, 10:right_wrist, 11:left_hip, 12:right_hip,
  13:left_knee, 14:right_knee, 15:left_ankle, 16:right_ankle

Usage:
    python 01_process_match.py match_video.mp4 output_keypoints.json [--fps 5]
"""

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO


def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        print(f"Using GPU: {name}")
        return 'cuda:0'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        print("Using Apple MPS")
        return 'mps'
    print("Using CPU (will be slower)")
    return 'cpu'


def process_match(video_path, output_path, sample_fps=5):
    device = get_device()

    # yolo11m-pose: good balance of speed and accuracy
    # yolo11n-pose: faster but less accurate (use if m is too slow)
    model_name = 'yolo11m-pose.pt'
    print(f"Loading {model_name}...")
    model = YOLO(model_name)
    model.to(device)
    print(f"Model loaded on {device}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    native_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(native_fps / sample_fps))
    est_process_frames = total_frames // frame_interval

    print(f"Video: {native_fps:.0f}fps, {total_frames} frames, ~{total_frames/native_fps/60:.0f} min")
    print(f"Sampling every {frame_interval}th frame ({sample_fps}fps)")
    print(f"Estimated frames to process: {est_process_frames}")
    print()

    all_results = []
    frame_idx = 0
    processed = 0
    start_time = time.time()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            # Run YOLO pose estimation
            results = model(frame, verbose=False, conf=0.3)

            match_minute = (frame_idx / native_fps) / 60
            frame_data = {
                'frame_idx': frame_idx,
                'match_minute': round(match_minute, 2),
                'persons': []
            }

            for result in results:
                if result.keypoints is None or result.boxes is None:
                    continue

                boxes = result.boxes
                keypoints = result.keypoints

                for i in range(len(boxes)):
                    bbox_conf = float(boxes.conf[i])
                    if bbox_conf < 0.5:
                        continue

                    bbox = boxes.xyxy[i].cpu().numpy().tolist()
                    kp_xy = keypoints.xy[i].cpu().numpy().tolist()      # (17, 2)
                    kp_conf = keypoints.conf[i].cpu().numpy().tolist()   # (17,)

                    frame_data['persons'].append({
                        'bbox': [float(x) for x in bbox],
                        'bbox_score': bbox_conf,
                        'keypoints': [[float(x) for x in pt] for pt in kp_xy],
                        'keypoint_scores': [float(x) for x in kp_conf]
                    })

            all_results.append(frame_data)
            processed += 1

            if processed % 50 == 0:
                elapsed = time.time() - start_time
                fps_rate = processed / elapsed
                remaining = (est_process_frames - processed) / fps_rate if fps_rate > 0 else 0
                print(f"  [{processed}/{est_process_frames}] "
                      f"Minute {match_minute:.1f} | "
                      f"{fps_rate:.1f} frames/sec | "
                      f"~{remaining/60:.0f} min remaining | "
                      f"{len(frame_data['persons'])} players detected")

            # Save checkpoint every 200 frames
            if processed % 200 == 0:
                with open(output_path, 'w') as f:
                    json.dump(all_results, f)

        frame_idx += 1

    cap.release()

    with open(output_path, 'w') as f:
        json.dump(all_results, f)

    elapsed = time.time() - start_time
    print(f"\nDone. {processed} frames in {elapsed/60:.1f} minutes.")
    print(f"Saved to {output_path}")
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process match video for pose estimation')
    parser.add_argument('video', help='Path to match video file')
    parser.add_argument('output', help='Path for output JSON file')
    parser.add_argument('--fps', type=int, default=5, help='Sample rate in fps (default: 5)')
    args = parser.parse_args()

    process_match(args.video, args.output, sample_fps=args.fps)

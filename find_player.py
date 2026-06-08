"""
Helper: Extract the first frame of a match and annotate all detected players
with their bounding box center coordinates. This helps you identify the
target player's starting position for tracking.

Usage:
    python find_player.py videos/match_001.mp4

Opens/saves an annotated image showing all detected players with their
bbox center coordinates. Find your target player and note their (x, y).
"""

import sys
import cv2
import numpy as np
from ultralytics import YOLO


def annotate_first_frame(video_path, output_path=None, frame_number=0):
    model = YOLO('yolo11m-pose.pt')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    # Skip to requested frame
    if frame_number > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("ERROR: Could not read frame")
        sys.exit(1)

    # Run detection
    results = model(frame, verbose=False, conf=0.3)

    # Draw annotations
    annotated = frame.copy()

    for result in results:
        if result.boxes is None:
            continue

        for i in range(len(result.boxes)):
            bbox = result.boxes.xyxy[i].cpu().numpy()
            conf = float(result.boxes.conf[i])

            if conf < 0.5:
                continue

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)

            # Draw bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw center point
            cv2.circle(annotated, (cx, cy), 5, (0, 0, 255), -1)

            # Label with coordinates
            label = f"({cx},{cy}) conf:{conf:.2f}"
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw keypoints if available
            if result.keypoints is not None:
                kp_xy = result.keypoints.xy[i].cpu().numpy()
                kp_conf = result.keypoints.conf[i].cpu().numpy()
                for j, (pt, c) in enumerate(zip(kp_xy, kp_conf)):
                    if c > 0.3:
                        px, py = int(pt[0]), int(pt[1])
                        color = (255, 0, 0) if j in [11,12,13,14,15,16] else (255, 255, 0)
                        cv2.circle(annotated, (px, py), 3, color, -1)

    out_path = output_path or video_path.replace('.mp4', '_players.jpg')
    cv2.imwrite(out_path, annotated)

    # Print all detected positions
    print(f"\nDetected players in frame {frame_number}:")
    print(f"{'#':<5} {'Center (x,y)':<20} {'BBox':<35} {'Confidence':<12}")
    print("-" * 72)

    player_idx = 0
    for result in results:
        if result.boxes is None:
            continue
        for i in range(len(result.boxes)):
            bbox = result.boxes.xyxy[i].cpu().numpy()
            conf = float(result.boxes.conf[i])
            if conf < 0.5:
                cx = int((bbox[0] + bbox[2]) / 2)
                cy = int((bbox[1] + bbox[3]) / 2)
                continue

            cx = int((bbox[0] + bbox[2]) / 2)
            cy = int((bbox[1] + bbox[3]) / 2)
            bbox_str = f"[{int(bbox[0])},{int(bbox[1])},{int(bbox[2])},{int(bbox[3])}]"
            player_idx += 1
            print(f"{player_idx:<5} ({cx},{cy}){'':<12} {bbox_str:<35} {conf:.3f}")

    print(f"\nAnnotated image saved: {out_path}")
    print(f"\nTo track a specific player, use their (x,y) coordinates:")
    print(f"  python 05_injury_validation.py output/keypoints.json --player-pos X,Y --injury-minute M")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python find_player.py VIDEO_PATH [FRAME_NUMBER]")
        sys.exit(1)

    video_path = sys.argv[1]
    frame_num = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    annotate_first_frame(video_path, frame_number=frame_num)

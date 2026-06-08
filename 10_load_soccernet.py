#!/usr/bin/env python3
"""
Phase 1: SoccerNet reader
Loads one SoccerNet match and returns a clean per-player data structure.

License Constraint: NON-COMMERCIAL RESEARCH USE ONLY (NDA-BOUND).
"""

import os
import json
import configparser
import argparse
from collections import defaultdict

def load_soccernet_match(match_dir, min_detections=500):
    match_id = os.path.basename(os.path.normpath(match_dir))
    
    # Parse seqinfo.ini
    seqinfo_path = os.path.join(match_dir, 'seqinfo.ini')
    config = configparser.ConfigParser()
    config.read(seqinfo_path)
    
    try:
        fps = int(config['Sequence']['frameRate'])
        seq_length = int(config['Sequence']['seqLength'])
    except KeyError:
        fps = 25
        seq_length = 0
        print("Warning: Could not parse frameRate or seqLength from seqinfo.ini. Defaulting to 25 fps.")
        
    duration_seconds = seq_length / fps if fps > 0 else 0
    
    # Parse gt.txt
    # format: frame_id, track_id, bbox_x, bbox_y, bbox_w, bbox_h, conf, class, vis
    gt_path = os.path.join(match_dir, 'gt', 'gt.txt')
    
    raw_tracks = defaultdict(list)
    
    if not os.path.exists(gt_path):
        print(f"Error: {gt_path} not found.")
        return
        
    with open(gt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 9:
                continue
            
            frame_id = int(parts[0])
            track_id = int(parts[1])
            bbox_x = float(parts[2])
            bbox_y = float(parts[3])
            bbox_w = float(parts[4])
            bbox_h = float(parts[5])
            conf = float(parts[6])
            cls = int(parts[7])
            vis = float(parts[8])
            
            # class=1 is player in standard MOT. Sometimes other classes are refs/ball.
            if cls != 1:
                continue
                
            minute = (frame_id / fps) / 60.0
            
            raw_tracks[track_id].append({
                "frame_idx": frame_id,
                "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                "vis": vis,
                "minute": round(minute, 4)
            })

    # Filter tracks
    filtered_players = {}
    track_durations = []
    
    for tid, frames in raw_tracks.items():
        frames.sort(key=lambda x: x["frame_idx"])
        n_det = len(frames)
        
        first_min = frames[0]["minute"]
        last_min = frames[-1]["minute"]
        duration_mins = last_min - first_min
        
        if n_det >= min_detections:
            filtered_players[f"track_{tid}"] = {
                "track_id": tid,
                "frames": frames,
                "first_minute": first_min,
                "last_minute": last_min,
                "duration_minutes": round(duration_mins, 2),
                "n_detections": n_det
            }
            track_durations.append(duration_mins)
            
            # Honest reporting
            if duration_mins < 5.0:
                print(f"  Flag: Track {tid} has >= {min_detections} detections but spans only {duration_mins:.1f} minutes.")
        else:
            # print(f"  Filtered out track {tid} ({n_det} detections).")
            pass

    # Summary
    print("\n=== SoccerNet Load Summary ===")
    print(f"Match ID: {match_id}")
    print(f"Total Duration: {duration_seconds/60:.1f} minutes ({fps} fps)")
    print(f"Total raw tracks: {len(raw_tracks)}")
    print(f"Valid player tracks (>= {min_detections} det): {len(filtered_players)}")
    
    if track_durations:
        mean_dur = sum(track_durations) / len(track_durations)
        max_dur = max(track_durations)
        print(f"Mean track duration: {mean_dur:.1f} mins")
        print(f"Max track duration: {max_dur:.1f} mins")
        
        # Distribution
        bins = {"<10m": 0, "10-30m": 0, "30-60m": 0, ">60m": 0}
        for d in track_durations:
            if d < 10: bins["<10m"] += 1
            elif d < 30: bins["10-30m"] += 1
            elif d < 60: bins["30-60m"] += 1
            else: bins[">60m"] += 1
        print("Track duration distribution:", bins)
    else:
        print("No tracks passed the threshold.")

    output_data = {
        "match_id": match_id,
        "fps": fps,
        "duration_seconds": duration_seconds,
        "players": filtered_players
    }
    
    os.makedirs('soccernet_loaded', exist_ok=True)
    out_path = os.path.join('soccernet_loaded', f"{match_id}_tracks.json")
    with open(out_path, 'w') as f:
        json.dump(output_data, f)
    
    print(f"Saved to {out_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--match-dir", type=str, default="/Users/devangpant/datasets/SoccerNet/tracking/train/SNMOT-060", help="Path to SoccerNet match directory")
    parser.add_argument("--min-det", type=int, default=500, help="Minimum detections to keep a track")
    args = parser.parse_args()
    
    load_soccernet_match(args.match_dir, args.min_det)

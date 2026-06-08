import os
import json
import configparser
import argparse
from collections import defaultdict

def load_soccernet_clip(clip_dir):
    clip_id = os.path.basename(os.path.normpath(clip_dir))
    
    # Parse seqinfo.ini
    seqinfo_path = os.path.join(clip_dir, 'seqinfo.ini')
    config = configparser.ConfigParser()
    config.read(seqinfo_path)
    
    try:
        fps = int(config['Sequence']['frameRate'])
        n_frames = int(config['Sequence']['seqLength'])
    except KeyError:
        fps = 25
        n_frames = 750
        print("Warning: Could not parse frameRate or seqLength. Defaulting.")
        
    duration_seconds = n_frames / fps if fps > 0 else 0
    
    gt_path = os.path.join(clip_dir, 'gt', 'gt.txt')
    if not os.path.exists(gt_path):
        print(f"Error: {gt_path} not found.")
        return
        
    raw_tracks = defaultdict(list)
    
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
            
            # class 1 is player, but SoccerNet might use -1 or other depending on format.
            # Based on user prompt: 1,1,914,855,55,172,1,-1,-1,-1
            # Here conf is index 6, class is index 7. Let's just grab all valid track_ids
            # if they have valid bounding boxes (width > 0, height > 0)
            if bbox_w <= 0 or bbox_h <= 0:
                continue
                
            raw_tracks[track_id].append({
                "frame_idx": frame_id,
                "bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
                "vis": vis
            })

    output_tracks = {}
    track_durations = []
    full_clip_tracks = 0
    
    for tid, frames in raw_tracks.items():
        frames.sort(key=lambda x: x["frame_idx"])
        n_det = len(frames)
        
        output_tracks[str(tid)] = {
            "track_id": tid,
            "n_detections": n_det,
            "frames": frames
        }
        
        track_durations.append(n_det)
        if n_det >= (n_frames * 0.9): # spans 90%+ of the clip
            full_clip_tracks += 1

    print("\n=== SoccerNet Clip Load Summary ===")
    print(f"Clip ID: {clip_id}")
    print(f"Total Tracks: {len(output_tracks)}")
    print(f"Tracks spanning >90% of clip ({n_frames} frames): {full_clip_tracks}")
    
    if track_durations:
        bins = {"<100f": 0, "100-400f": 0, "400-700f": 0, ">700f": 0}
        for d in track_durations:
            if d < 100: bins["<100f"] += 1
            elif d < 400: bins["100-400f"] += 1
            elif d < 700: bins["400-700f"] += 1
            else: bins[">700f"] += 1
        print("Track duration distribution (in frames):", bins)

    output_data = {
        "clip_id": clip_id,
        "fps": fps,
        "n_frames": n_frames,
        "duration_seconds": duration_seconds,
        "tracks": output_tracks
    }
    
    os.makedirs('soccernet_loaded', exist_ok=True)
    out_path = os.path.join('soccernet_loaded', f"{clip_id}_tracks.json")
    with open(out_path, 'w') as f:
        json.dump(output_data, f)
        
    print(f"Saved to {out_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-dir", type=str, default="/Users/devangpant/datasets/SoccerNet/tracking/train/SNMOT-060", help="Path to SoccerNet clip directory")
    args = parser.parse_args()
    
    load_soccernet_clip(args.clip_dir)

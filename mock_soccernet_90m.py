import os
import random

base_dir = "/Users/devangpant/datasets/SoccerNet/tracking/train/SNMOT-060"
gt_lines = []
# Create 5 valid tracks spanning up to 90 mins
fps = 25
max_frame = 90 * 60 * fps # 135000 frames

tracks = {
    1: {"frames": 10000, "valid": True},
    2: {"frames": 8000, "valid": True},
    3: {"frames": 12000, "valid": True},
    4: {"frames": 9000, "valid": True},
    5: {"frames": 300, "valid": False}, # noise
    6: {"frames": 100, "valid": False}, # noise
}

for t_id, info in tracks.items():
    frames_count = info["frames"]
    # Spread frames across 90 minutes (0 to max_frame)
    if info["valid"]:
        # Cover at least 45 minutes
        step = max_frame // frames_count
        for i in range(frames_count):
            f_id = 1 + int(i * step + random.randint(-10, 10))
            f_id = max(1, min(max_frame, f_id))
            gt_lines.append(f"{f_id},{t_id},{100},{100},{50},{150},{0.95},{1},{0.9}")
    else:
        start_frame = random.randint(1, 1000)
        for i in range(frames_count):
            f_id = start_frame + i * 5
            gt_lines.append(f"{f_id},{t_id},{100},{100},{50},{150},{0.95},{1},{0.9}")

gt_lines.sort(key=lambda line: int(line.split(',')[0]))

with open(os.path.join(base_dir, "gt", "gt.txt"), "w") as f:
    f.write("\n".join(gt_lines) + "\n")

print("Mock SoccerNet data generated with 90 min span.")

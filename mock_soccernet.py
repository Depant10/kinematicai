import os
import random
from PIL import Image, ImageDraw

base_dir = "/Users/devangpant/datasets/SoccerNet/tracking/train/SNMOT-060"
os.makedirs(os.path.join(base_dir, "gt"), exist_ok=True)
os.makedirs(os.path.join(base_dir, "img1"), exist_ok=True)

with open(os.path.join(base_dir, "seqinfo.ini"), "w") as f:
    f.write("[Sequence]\nname=SNMOT-060\nframeRate=25\nseqLength=150000\nimWidth=1920\nimHeight=1080\n")

gt_lines = []
# Create 5 valid tracks ( > 500 detections) and 2 noise tracks (< 500)
tracks = {
    1: {"frames": 1000, "valid": True},
    2: {"frames": 800, "valid": True},
    3: {"frames": 1200, "valid": True},
    4: {"frames": 600, "valid": True},
    5: {"frames": 300, "valid": False}, # noise
    6: {"frames": 100, "valid": False}, # noise
}

for t_id, info in tracks.items():
    frames_count = info["frames"]
    # Scatter them across the 150000 frames
    start_frame = random.randint(1, 140000)
    for i in range(frames_count):
        f_id = start_frame + i * 5 # every 5 frames or so
        x = random.randint(100, 1800)
        y = random.randint(100, 900)
        w = 50
        h = 150
        conf = 0.95
        cls = 1
        vis = 0.9
        gt_lines.append(f"{f_id},{t_id},{x},{y},{w},{h},{conf},{cls},{vis}")

# sort by frame
gt_lines.sort(key=lambda line: int(line.split(',')[0]))

with open(os.path.join(base_dir, "gt", "gt.txt"), "w") as f:
    f.write("\n".join(gt_lines) + "\n")

# Create a few dummy images just so code doesn't crash
img = Image.new('RGB', (1920, 1080), color = 'green')
for f_id in [int(l.split(',')[0]) for l in gt_lines[:20]]: # Just the first 20 to avoid slow disk
    img.save(os.path.join(base_dir, "img1", f"{f_id:06d}.jpg"))

print("Mock SoccerNet data generated.")

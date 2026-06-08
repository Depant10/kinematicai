# KinematicAI — Gait-Based Injury Risk from Broadcast Football Video

> ⚠️ **Work in Progress** — core pipeline functional, dashboard and validation layer under active development.

> Detects biomechanical asymmetry in athletes' gait using only standard broadcast video — no wearables, no lab setup, no cooperation from the subject.

---

## The problem

Sports injury prediction systems either require expensive sensor suits (unrealistic in the field) or rely on outcome statistics after injuries have already occurred. Neither prevents injuries. KinematicAI works from footage that already exists: broadcast video of athletes in motion.

## What it does

Given a video clip of an athlete running, the system:

1. Detects and tracks each player across frames (multi-object tracking)
2. Estimates full-body pose at each frame for the target athlete
3. Extracts kinematic features: stride symmetry, knee flexion asymmetry, hip drop angle, cadence variance
4. Scores gait asymmetry on a per-athlete basis and flags elevated injury risk

## Architecture

```
Input video (broadcast / match footage)
    │
    ▼
YOLO11m-pose — person detection + pose estimation (17 keypoints)
    │
    ▼
BoT-SORT — multi-object tracking across frames
    │
    ▼
Kinematic feature extraction
  ├── Left/right stride length ratio
  ├── Knee flexion angle asymmetry
  ├── Hip drop (Trendelenburg) proxy
  └── Cadence variance over N strides
    │
    ▼
PyTorch classifier — asymmetry score → risk tier (low / moderate / high)
    │
    ▼
Per-athlete risk report with flagged frames
```

## Stack

| Layer | Technology |
|---|---|
| Pose estimation | MMPose + YOLO11m-pose |
| Tracking | BoT-SORT |
| Feature extraction | OpenCV, NumPy |
| Classification | PyTorch |
| Visualization | OpenCV overlays |

## What was hard

**Occlusion.** In broadcast football, players overlap constantly. BoT-SORT handles re-identification reasonably well, but short occlusions still break stride windows. Filtering incomplete stride cycles without discarding too much data required careful windowing logic.

**Defining "asymmetry" without ground truth.** There's no universally agreed biomechanical threshold for injury risk from gait alone. The classifier is trained on proxies derived from published sports medicine literature — treat the risk tiers as directional, not clinical. `[VERIFY: cite specific papers used for threshold calibration]`

## Usage

```bash
# [VERIFY: add actual run instructions once repo is finalized]
python run_analysis.py --video path/to/clip.mp4 --player_id 7
```

---

Built by [Devang Pant](https://linkedin.com/in/devangpant1) · [linkedin.com/in/devangpant1](https://linkedin.com/in/devangpant1)

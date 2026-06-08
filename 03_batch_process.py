"""
Step 3: Batch process multiple matches before you leave.
Downloads videos and runs pose estimation on all of them.

Usage:
    python 03_batch_process.py

Edit the MATCHES list below with your video URLs/paths.
"""

import os
import subprocess
import sys
import time


# === EDIT THIS LIST ===
# Add match video URLs (YouTube full match replays) or local file paths.
# Pick 3 matches from the SAME TEAM so you can track the same players.
MATCHES = [
    # Example — replace with real URLs:
    # "https://www.youtube.com/watch?v=EXAMPLE1",
    # "https://www.youtube.com/watch?v=EXAMPLE2",
    # "https://www.youtube.com/watch?v=EXAMPLE3",
    #
    # Or local files:
    # "C:/Users/Admin/injury-pred/videos/match_001.mp4",
]

SAMPLE_FPS = 5  # Frames per second to sample. 5 is good for RTX 5070.
VIDEO_DIR = "C:/Users/Admin/injury-pred/videos"
OUTPUT_DIR = "C:/Users/Admin/injury-pred/output"


def download_video(url, output_path):
    """Download video using yt-dlp."""
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height>=720]+bestaudio/best[height>=720]",
        "--merge-output-format", "mp4",
        "-o", output_path,
        url
    ]
    print(f"Downloading: {url}")
    print(f"  -> {output_path}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr[:200]}")
        return False
    print(f"  Done.")
    return True


def process_video(video_path, output_path, fps):
    """Run pose estimation on a video."""
    cmd = [
        sys.executable,
        "C:/Users/Admin/injury-pred/01_process_match.py",
        video_path,
        output_path,
        "--fps", str(fps)
    ]
    print(f"\nProcessing: {video_path}")
    print(f"  -> {output_path}")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    if not MATCHES:
        print("ERROR: No matches configured!")
        print("Edit the MATCHES list in this file with YouTube URLs or local file paths.")
        sys.exit(1)

    os.makedirs(VIDEO_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    video_paths = []
    total_start = time.time()

    # Step 1: Download any URLs
    for i, match in enumerate(MATCHES):
        if match.startswith("http"):
            video_path = os.path.join(VIDEO_DIR, f"match_{i+1:03d}.mp4")
            if os.path.exists(video_path):
                print(f"Already downloaded: {video_path}")
                video_paths.append(video_path)
            else:
                if download_video(match, video_path):
                    video_paths.append(video_path)
                else:
                    print(f"Skipping failed download: {match}")
        else:
            if os.path.exists(match):
                video_paths.append(match)
            else:
                print(f"File not found: {match}")

    if not video_paths:
        print("No valid videos to process.")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Processing {len(video_paths)} matches at {SAMPLE_FPS}fps")
    print(f"{'='*60}\n")

    # Step 2: Process each video
    for i, video_path in enumerate(video_paths):
        output_path = os.path.join(OUTPUT_DIR, f"match_{i+1:03d}_keypoints.json")
        if os.path.exists(output_path):
            print(f"Already processed: {output_path}")
            continue

        match_start = time.time()
        success = process_video(video_path, output_path, SAMPLE_FPS)
        match_elapsed = time.time() - match_start

        if success:
            print(f"  Completed in {match_elapsed/60:.1f} minutes")
        else:
            print(f"  FAILED after {match_elapsed/60:.1f} minutes")

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"All done. Total time: {total_elapsed/60:.1f} minutes")
    print(f"Output files in: {OUTPUT_DIR}")
    print(f"\nNext step: run analysis on each output:")
    print(f"  python 02_analyze_gait.py {OUTPUT_DIR}/match_001_keypoints.json")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

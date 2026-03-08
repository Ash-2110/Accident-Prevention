"""
=============================================================================
Dataset Creator — Extract Frames from YouTube Industrial Safety Videos
=============================================================================
This script:
1. Downloads YouTube videos using yt-dlp
2. Extracts frames at regular intervals
3. Saves them into a structured dataset folder
=============================================================================
Usage:
    python src/extract_frames.py

Requirements:
    pip install yt-dlp opencv-python
=============================================================================
"""

import cv2
import os
import subprocess
import sys
import time
from pathlib import Path

# ─── Dataset Configuration ──────────────────────────────────────────────────
YOUTUBE_SOURCES = [
    {
        "url"    : "https://youtu.be/bVHod9L73Pc",
        "name"   : "video1_stone_crusher",
        "output" : "dataset/frames_video1",
    },
    {
        "url"    : "https://youtube.com/shorts/hXrkYIzjDg4",
        "name"   : "video2_conveyor_belt",
        "output" : "dataset/frames_video2",
    },
]

FRAME_INTERVAL = 15      # Extract 1 frame every N frames
MAX_FRAMES     = 300     # Maximum frames to extract per video
IMG_WIDTH      = 720     # Resize width
IMG_HEIGHT     = 480     # Resize height
IMG_QUALITY    = 95      # JPEG quality (1-100)


# =============================================================================
#  DOWNLOADER
# =============================================================================
def check_ytdlp():
    """Check if yt-dlp is installed."""
    try:
        result = subprocess.run(["yt-dlp", "--version"],
                                capture_output=True, text=True)
        print(f"[OK] yt-dlp version: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("[ERROR] yt-dlp not found.")
        print("        Install it with: pip install yt-dlp")
        return False


def download_video(url: str, output_path: str) -> str:
    """
    Download a YouTube video using yt-dlp.

    Returns:
        Path to the downloaded .mp4 file, or None if failed.
    """
    os.makedirs("dataset/raw_videos", exist_ok=True)
    output_template = f"dataset/raw_videos/{output_path}.%(ext)s"

    print(f"\n[DOWNLOAD] Downloading: {url}")
    cmd = [
        "yt-dlp",
        "--format", "mp4/bestvideo[height<=480]+bestaudio/best",
        "--merge-output-format", "mp4",
        "--output", output_template,
        "--no-playlist",
        url
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            # Find the downloaded file
            for f in Path("dataset/raw_videos").glob(f"{output_path}.*"):
                print(f"[OK] Downloaded: {f}")
                return str(f)
        else:
            print(f"[ERROR] Download failed:\n{result.stderr}")
            return None
    except subprocess.TimeoutExpired:
        print("[ERROR] Download timed out.")
        return None
    except Exception as e:
        print(f"[ERROR] {e}")
        return None


# =============================================================================
#  FRAME EXTRACTOR
# =============================================================================
def extract_frames(video_path: str, output_dir: str, source_name: str) -> int:
    """
    Extract frames from a video file at regular intervals.

    Args:
        video_path  : Path to the video file
        output_dir  : Directory to save extracted frames
        source_name : Label prefix for filenames

    Returns:
        Number of frames extracted
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return 0

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    duration = total_video_frames / fps_video if fps_video > 0 else 0

    print(f"\n[INFO] Video: {video_path}")
    print(f"       Total frames : {total_video_frames}")
    print(f"       FPS          : {fps_video:.1f}")
    print(f"       Duration     : {duration:.1f}s")
    print(f"       Extracting every {FRAME_INTERVAL} frames → max {MAX_FRAMES} images")

    frame_count = 0
    saved_count = 0

    while saved_count < MAX_FRAMES:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % FRAME_INTERVAL == 0:
            # Resize
            frame_resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))

            # Save
            filename = f"{source_name}_frame_{saved_count:04d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame_resized,
                        [cv2.IMWRITE_JPEG_QUALITY, IMG_QUALITY])
            saved_count += 1

            if saved_count % 50 == 0:
                print(f"  ... {saved_count} frames saved")

    cap.release()
    print(f"[OK] Extracted {saved_count} frames → {output_dir}")
    return saved_count


# =============================================================================
#  DATASET SUMMARY
# =============================================================================
def print_dataset_summary():
    """Print a summary of the created dataset."""
    print("\n" + "="*55)
    print("  DATASET SUMMARY")
    print("="*55)

    total = 0
    for src in YOUTUBE_SOURCES:
        out_dir = src["output"]
        if os.path.exists(out_dir):
            frames = list(Path(out_dir).glob("*.jpg"))
            n = len(frames)
            total += n
            print(f"  {src['name']:<30} {n:>4} frames")
        else:
            print(f"  {src['name']:<30}    0 frames (dir missing)")

    print(f"  {'─'*45}")
    print(f"  {'TOTAL':<30} {total:>4} frames")
    print("="*55)

    print("\nDataset Folder Structure:")
    for src in YOUTUBE_SOURCES:
        print(f"  {src['output']}/")
        out_dir = src["output"]
        if os.path.exists(out_dir):
            files = sorted(Path(out_dir).glob("*.jpg"))[:3]
            for f in files:
                print(f"    ├── {f.name}")
            if len(list(Path(out_dir).glob("*.jpg"))) > 3:
                print(f"    └── ...")


# =============================================================================
#  CREATE SAMPLE ANNOTATIONS FILE
# =============================================================================
def create_annotation_template(output_dir: str):
    """
    Create a simple CSV template for manual annotation.
    (For training custom YOLO models)
    """
    import pandas as pd

    frames = sorted(Path(output_dir).glob("*.jpg"))
    if not frames:
        return

    data = []
    for f in frames:
        data.append({
            "image_file"  : str(f),
            "label"       : "person",   # or 'machine', 'danger_zone'
            "x_center"    : "",         # YOLO format (0–1)
            "y_center"    : "",
            "width"       : "",
            "height"      : "",
            "in_danger_zone": "",       # yes / no
        })

    df = pd.DataFrame(data)
    csv_path = os.path.join(output_dir, "annotations_template.csv")
    df.to_csv(csv_path, index=False)
    print(f"[OK] Annotation template saved: {csv_path}")


# =============================================================================
#  MAIN
# =============================================================================
def main():
    print("="*55)
    print("  Industrial Safety Dataset Creator")
    print("  Extracting frames from YouTube videos")
    print("="*55)

    if not check_ytdlp():
        print("\n[FALLBACK] yt-dlp not available.")
        print("  You can manually download the videos and place them in:")
        print("  dataset/raw_videos/")
        print("\n  Then run this script again with --local flag.")
        print("  OR provide a local video path in YOUTUBE_SOURCES.")
        return

    total_frames = 0

    for src in YOUTUBE_SOURCES:
        print(f"\n{'='*55}")
        print(f"  Processing: {src['name']}")
        print(f"{'='*55}")

        # Step 1: Download
        video_file = download_video(src["url"], src["name"])

        if video_file is None:
            print(f"[SKIP] Could not download {src['name']}")
            continue

        # Step 2: Extract frames
        n = extract_frames(video_file, src["output"], src["name"])
        total_frames += n

        # Step 3: Create annotation template
        try:
            import pandas
            create_annotation_template(src["output"])
        except ImportError:
            pass

        print(f"[DONE] {src['name']}: {n} frames extracted")

    print_dataset_summary()
    print(f"\n✅ Dataset creation complete! Total frames: {total_frames}")
    print("   Next step: Run safety_monitor.py to start detection.\n")


if __name__ == "__main__":
    # Allow passing local video files as arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--local":
        print("[LOCAL MODE] Using local video files...")
        # Override URLs with local paths
        if len(sys.argv) > 2:
            YOUTUBE_SOURCES[0]["url"] = sys.argv[2]
        if len(sys.argv) > 3:
            YOUTUBE_SOURCES[1]["url"] = sys.argv[3]

    main()

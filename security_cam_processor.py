#!/usr/bin/env python3
"""
Local Video Motion Zone Detector

Processes video files frame-by-frame to detect motion within configurable
rectangular zones. Logs motion events with timestamps and motion scores.

Usage:
    python3 security_cam_processor.py --make-demo  # Generate synthetic demo
    python3 security_cam_processor.py               # Process demo video
    python3 security_cam_processor.py --input path/to/video.mp4 --threshold 300000
"""

import argparse
import json
import os
from pathlib import Path
from datetime import datetime, timezone

import cv2
import numpy as np


DEFAULT_ZONES = {
    "zones": [
        {
            "name": "front_door",
            "x1": 80,
            "y1": 120,
            "x2": 280,
            "y2": 330
        },
        {
            "name": "driveway",
            "x1": 300,
            "y1": 140,
            "x2": 560,
            "y2": 330
        }
    ]
}


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Local video motion zone detector"
    )

    parser.add_argument(
        "--input",
        default="demo/synthetic_security_clip.mp4",
        help="Input video file"
    )

    parser.add_argument(
        "--zones",
        default="zones.json",
        help="Zone config JSON file"
    )

    parser.add_argument(
        "--out",
        default="results",
        help="Output directory"
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=250000,
        help="Motion score threshold per zone"
    )

    parser.add_argument(
        "--cooldown",
        type=float,
        default=2.0,
        help="Cooldown in seconds between events per zone"
    )

    parser.add_argument(
        "--frame-skip",
        type=int,
        default=2,
        help="Process every N frames"
    )

    parser.add_argument(
        "--make-demo",
        action="store_true",
        help="Generate a synthetic demo security clip"
    )

    return parser.parse_args()


def ensure_default_zones(path):
    """Create a default zones.json file if it doesn't exist."""
    path = Path(path)

    if path.exists():
        return

    with path.open("w", encoding="utf-8") as f:
        json.dump(DEFAULT_ZONES, f, indent=2)

    print(f"Created default zones file: {path}")


def load_zones(path):
    """
    Load zone definitions from JSON file.
    
    Args:
        path: Path to zones.json
        
    Returns:
        List of zone dictionaries with keys: name, x1, y1, x2, y2
        
    Raises:
        ValueError: If zones are missing or have invalid structure
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    zones = data.get("zones", [])

    if not zones:
        raise ValueError("No zones found in zones.json")

    # Validate that each zone has required coordinate fields
    required = {"name", "x1", "y1", "x2", "y2"}

    for zone in zones:
        missing = required - set(zone.keys())
        if missing:
            raise ValueError(f"Zone missing required fields: {missing}")

    return zones


def generate_demo_video(path):
    """
    Create a synthetic security camera video for testing.
    
    Generates a 12-second clip (640x360, 20fps) with:
    - Two labeled zones (front_door, driveway)
    - Moving white rectangle crossing front_door zone
    - Moving white circle crossing driveway zone
    
    Args:
        path: Output video file path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    width = 640
    height = 360
    fps = 20
    seconds = 12
    total_frames = fps * seconds

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError("Could not create demo video")

    for i in range(total_frames):
        # Start with dark background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (30, 30, 30)

        # Draw zone background rectangles
        cv2.rectangle(frame, (70, 110), (290, 340), (45, 45, 45), -1)
        cv2.rectangle(frame, (310, 130), (570, 340), (50, 50, 50), -1)

        # Label zones
        cv2.putText(
            frame,
            "front_door",
            (95, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (120, 120, 120),
            1
        )

        cv2.putText(
            frame,
            "driveway",
            (390, 125),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (120, 120, 120),
            1
        )

        # Moving object crosses front_door zone (frames 30-95)
        # Horizontal motion left-to-right across zone
        if 30 <= i <= 95:
            x = 40 + (i - 30) * 3  # Linear movement
            y = 200
            cv2.rectangle(frame, (x, y), (x + 45, y + 45), (230, 230, 230), -1)

        # Moving object crosses driveway zone (frames 120-205)
        # Horizontal motion right-to-left across zone
        if 120 <= i <= 205:
            x = 580 - (i - 120) * 3  # Linear movement in opposite direction
            y = 230
            cv2.circle(frame, (x, y), 25, (240, 240, 240), -1)

        # Frame counter for reference
        cv2.putText(
            frame,
            f"synthetic security clip | frame {i}",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (180, 180, 180),
            1
        )

        writer.write(frame)

    writer.release()
    print(f"Demo video created: {path}")


def crop_zone(frame, zone):
    """
    Extract the region of interest (ROI) for a zone from a frame.
    
    Safely handles boundary conditions to avoid out-of-bounds errors.
    
    Args:
        frame: Input frame (grayscale or BGR)
        zone: Zone dict with x1, y1, x2, y2 coordinates
        
    Returns:
        Cropped region as numpy array
    """
    h, w = frame.shape[:2]

    # Clamp coordinates to frame bounds
    x1 = max(0, min(w, int(zone["x1"])))
    y1 = max(0, min(h, int(zone["y1"])))
    x2 = max(0, min(w, int(zone["x2"])))
    y2 = max(0, min(h, int(zone["y2"])))

    # OpenCV uses [row, col] indexing (y before x)
    return frame[y1:y2, x1:x2]


def draw_zones(frame, zones, active_zone=None):
    """
    Draw zone rectangles and labels on a frame.
    
    Active zone is highlighted in red; others in green.
    
    Args:
        frame: Input frame (BGR)
        zones: List of zone dicts
        active_zone: Name of zone to highlight (optional)
        
    Returns:
        Annotated frame copy
    """
    output = frame.copy()

    for zone in zones:
        # Green for inactive, red for active zone
        color = (0, 255, 0)

        if active_zone == zone["name"]:
            color = (0, 0, 255)

        x1 = int(zone["x1"])
        y1 = int(zone["y1"])
        x2 = int(zone["x2"])
        y2 = int(zone["y2"])

        # Draw rectangle outline
        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
        
        # Draw zone name above rectangle
        cv2.putText(
            output,
            zone["name"],
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return output


def process_video(args):
    """
    Main video processing pipeline.
    
    Reads video frame-by-frame, detects motion in each zone, logs events
    when motion exceeds threshold, and saves snapshots with annotation.
    
    Motion detection method:
    1. Convert each frame to grayscale
    2. Apply Gaussian blur to reduce noise
    3. Compute absolute difference between consecutive frames
    4. Sum pixel differences within each zone (motion score)
    5. Trigger event if score >= threshold and cooldown has passed
    
    Args:
        args: Parsed command-line arguments
    """
    input_path = Path(args.input)
    out_dir = Path(args.out)
    snapshot_dir = out_dir / "snapshots"
    event_log_path = out_dir / "events.jsonl"
    summary_path = out_dir / "summary.txt"

    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    ensure_default_zones(args.zones)
    zones = load_zones(args.zones)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        fps = 30

    previous_gray = None
    frame_index = 0
    event_count = 0
    zone_counts = {zone["name"]: 0 for zone in zones}
    last_event_time_by_zone = {zone["name"]: -9999.0 for zone in zones}

    started_at = datetime.now(timezone.utc).isoformat()

    print("=" * 60)
    print(" Local Video Motion Zone Detector")
    print(" PromptHound")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Zones: {args.zones}")
    print(f"Threshold: {args.threshold}")
    print(f"Frame skip: {args.frame_skip}")
    print(f"Cooldown: {args.cooldown}s")
    print("=" * 60)

    with event_log_path.open("w", encoding="utf-8") as event_log:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame_index += 1

            # Process every Nth frame to reduce computation
            if frame_index % args.frame_skip != 0:
                continue

            # Convert to grayscale and blur to reduce noise
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            # First frame: store as baseline, move to next
            if previous_gray is None:
                previous_gray = gray
                continue

            timestamp_sec = frame_index / fps

            # Check motion in each zone
            for zone in zones:
                zone_name = zone["name"]

                # Extract zone regions from current and previous frames
                previous_crop = crop_zone(previous_gray, zone)
                current_crop = crop_zone(gray, zone)

                # Skip empty crops (zone outside frame bounds)
                if previous_crop.size == 0 or current_crop.size == 0:
                    continue

                # Compute pixel-wise absolute difference (motion detection)
                diff = cv2.absdiff(previous_crop, current_crop)
                
                # Motion score = sum of all pixel differences in zone
                # Higher values = more motion
                motion_score = int(np.sum(diff))

                # Check if enough time has passed since last event (cooldown)
                time_since_last = timestamp_sec - last_event_time_by_zone[zone_name]

                # Trigger event if motion exceeds threshold and cooldown satisfied
                if motion_score >= args.threshold and time_since_last >= args.cooldown:
                    event_count += 1
                    zone_counts[zone_name] += 1
                    last_event_time_by_zone[zone_name] = timestamp_sec

                    # Generate unique snapshot filename
                    snapshot_name = (
                        f"event_{event_count:04d}_"
                        f"{zone_name}_"
                        f"t{timestamp_sec:.2f}_"
                        f"s{motion_score}.jpg"
                    )

                    snapshot_path = snapshot_dir / snapshot_name
                    
                    # Annotate frame with zone boundaries (active zone in red)
                    annotated = draw_zones(frame, zones, active_zone=zone_name)
                    cv2.imwrite(str(snapshot_path), annotated)

                    # Log event metadata as JSONL (one JSON object per line)
                    event = {
                        "event_id": event_count,
                        "input_file": str(input_path),
                        "zone": zone_name,
                        "frame": frame_index,
                        "timestamp_sec": round(timestamp_sec, 2),
                        "motion_score": motion_score,
                        "threshold": args.threshold,
                        "snapshot": str(snapshot_path),
                        "processed_at": datetime.now(timezone.utc).isoformat()
                    }

                    event_log.write(json.dumps(event) + "\n")
                    event_log.flush()

                    print(
                        f"[EVENT] #{event_count} "
                        f"zone={zone_name} "
                        f"t={timestamp_sec:.2f}s "
                        f"score={motion_score}"
                    )

            # Current frame becomes previous for next iteration
            previous_gray = gray

    cap.release()

    finished_at = datetime.now(timezone.utc).isoformat()

    # Write human-readable summary
    with summary_path.open("w", encoding="utf-8") as f:
        f.write("Local Video Motion Zone Detector Summary\n")
        f.write("=" * 45 + "\n")
        f.write(f"Input: {input_path}\n")
        f.write(f"Started: {started_at}\n")
        f.write(f"Finished: {finished_at}\n")
        f.write(f"Total events: {event_count}\n")
        f.write(f"Threshold: {args.threshold}\n")
        f.write(f"Frame skip: {args.frame_skip}\n")
        f.write(f"Cooldown: {args.cooldown}s\n")
        f.write("\nEvents by zone:\n")

        for zone_name, count in zone_counts.items():
            f.write(f"- {zone_name}: {count}\n")

    print("=" * 60)
    print("DONE")
    print(f"Total events: {event_count}")
    print(f"Events log: {event_log_path}")
    print(f"Snapshots: {snapshot_dir}")
    print(f"Summary: {summary_path}")
    print("=" * 60)


def main():
    """Entry point."""
    args = parse_args()

    if args.make_demo:
        generate_demo_video(args.input)

    process_video(args)


if __name__ == "__main__":
    main()

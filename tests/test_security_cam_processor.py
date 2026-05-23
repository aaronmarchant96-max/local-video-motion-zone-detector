from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from security_cam_processor import (
    crop_zone,
    ensure_default_zones,
    generate_demo_video,
    load_zones,
    process_video,
)


def test_ensure_default_zones_creates_valid_config(tmp_path):
    zones_path = tmp_path / "zones.json"

    ensure_default_zones(zones_path)
    zones = load_zones(zones_path)

    assert zones_path.exists()
    assert len(zones) >= 1
    assert zones[0]["name"]
    assert {"x1", "y1", "x2", "y2"}.issubset(zones[0].keys())


def test_crop_zone_returns_expected_shape():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)

    zone = {
        "name": "test_zone",
        "x1": 10,
        "y1": 20,
        "x2": 60,
        "y2": 80,
    }

    cropped = crop_zone(frame, zone)

    assert cropped.shape == (60, 50, 3)


def test_generate_demo_video_creates_readable_video(tmp_path):
    video_path = tmp_path / "demo.mp4"

    generate_demo_video(video_path)

    assert video_path.exists()
    assert video_path.stat().st_size > 0

    cap = cv2.VideoCapture(str(video_path))
    assert cap.isOpened()

    ok, frame = cap.read()
    cap.release()

    assert ok
    assert frame is not None


def test_process_video_writes_events_and_summary(tmp_path):
    video_path = tmp_path / "demo.mp4"
    zones_path = tmp_path / "zones.json"
    out_dir = tmp_path / "results"

    generate_demo_video(video_path)
    ensure_default_zones(zones_path)

    args = SimpleNamespace(
        input=str(video_path),
        zones=str(zones_path),
        out=str(out_dir),
        threshold=100000,
        cooldown=2.0,
        frame_skip=2,
    )

    process_video(args)

    events_path = out_dir / "events.jsonl"
    summary_path = out_dir / "summary.txt"
    snapshots_dir = out_dir / "snapshots"

    assert events_path.exists()
    assert summary_path.exists()
    assert snapshots_dir.exists()

    event_lines = events_path.read_text().strip().splitlines()

    assert len(event_lines) >= 1
    assert any(snapshots_dir.iterdir())
    assert "Total events:" in summary_path.read_text()

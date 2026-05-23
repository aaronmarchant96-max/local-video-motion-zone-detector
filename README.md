# Local Video Motion Zone Detector

A lightweight OpenCV tool for detecting motion events inside defined zones from local video files.

This project is a file based security footage analysis prototype. It does not require live cameras, webcams, RTSP feeds, or home surveillance hardware. It can generate its own synthetic demo clip, process that clip, detect motion inside configured zones, save event snapshots, and log structured JSONL output.

## Overview

The detector reads a video file frame by frame, compares motion between frames, checks motion inside named zones, and saves candidate events when motion crosses a threshold.

The goal is not facial recognition, object identification, or surveillance deployment. The goal is to build a clear, reproducible local motion event detector.

## Current V1 Capability

The V1 processor can:

1. Generate a synthetic security camera style demo video
2. Load named motion zones from `zones.json`
3. Compare grayscale frame differences
4. Calculate a motion score per zone
5. Apply a configurable threshold
6. Apply cooldown timing to reduce duplicate alerts
7. Save annotated event snapshots
8. Write structured event logs to JSONL
9. Write a plain text summary

## Project Structure

    local-video-motion-zone-detector/
      security_cam_processor.py
      requirements.txt
      zones.json
      README.md
      .gitignore

Generated local outputs are not committed to GitHub:

    demo/
    results/
    venv/

## Method

1. Open a local video file
2. Convert frames to grayscale
3. Blur frames slightly to reduce noise
4. Compare each processed frame to the previous processed frame
5. Crop comparison regions using configured zones
6. Sum pixel differences inside each zone
7. Trigger an event when the zone score crosses the threshold
8. Save an annotated snapshot with the active zone highlighted
9. Log event metadata to JSONL

## Example Zones

Zones are defined in `zones.json`.

    {
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

Each zone is a rectangle on the video frame.

    x1, y1 = top left corner
    x2, y2 = bottom right corner

## Setup

Create and activate a virtual environment:

    python3 -m venv venv
    source venv/bin/activate

Install dependencies:

    python3 -m pip install -r requirements.txt

## Usage

Generate a synthetic demo video and process it:

    python3 security_cam_processor.py --make-demo

Run against the default demo video:

    python3 security_cam_processor.py

Run against a custom video file:

    python3 security_cam_processor.py --input path/to/video.mp4

Use a custom threshold:

    python3 security_cam_processor.py --threshold 300000

Use a custom cooldown:

    python3 security_cam_processor.py --cooldown 3

Use a custom zones file:

    python3 security_cam_processor.py --zones zones.json

## Outputs

The processor writes results locally.

    results/events.jsonl
    results/summary.txt
    results/snapshots/

Example JSONL event:

    {
      "event_id": 1,
      "input_file": "demo/synthetic_security_clip.mp4",
      "zone": "front_door",
      "frame": 97,
      "timestamp_sec": 4.9,
      "motion_score": 382405,
      "threshold": 250000,
      "snapshot": "results/snapshots/event_0001_front_door_t4.90_s382405.jpg",
      "processed_at": "2026-05-23T18:36:00Z"
    }

## Why This Exists

This project is a small builder focused prototype.

It shows how a basic computer vision pipeline can turn messy video input into reviewable events. It is intentionally simple and local first. The value is in the workflow:

    define zones
    process video
    detect motion
    save evidence
    log structured output
    review results

## Relationship To Other Projects

This project follows the same practical methodology as my other tools.

    UAP Footage Analyzer:
    public footage to motion event candidates

    GOES Anomaly Hunter:
    satellite thermal data to hotspot candidates

    Local Video Motion Zone Detector:
    local video files to zone motion candidates

Different data source, same core pattern.

## Limitations

This V1 detector is intentionally simple.

It does not identify people, vehicles, animals, or objects. It only detects motion inside configured zones. Lighting changes, compression artifacts, shadows, camera shake, rain, and synthetic video artifacts can all trigger motion events.

## Future Improvements

Possible next steps:

1. Add contact sheets for event review
2. Add per zone thresholds
3. Add minimum object area filtering
4. Add background subtraction
5. Add CSV summaries
6. Add a demo GIF
7. Add optional real webcam or RTSP input later
8. Add a manual review labeling tool

## Disclaimer

This tool is for local video analysis and portfolio demonstration. It is not a production security system. It does not make identity claims, threat claims, or object classification claims.

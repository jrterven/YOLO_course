#!/usr/bin/env python3
"""YOLO object detection on live video or a video file.

Runs YOLO on frames from a webcam, RTSP stream, or video file and displays
annotated results in an OpenCV window.

Example:
    python scripts/detect/detect_video.py --source 0
    python scripts/detect/detect_video.py --source path/to/video.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_NAME = "yolo26n.pt"
DEFAULT_MODEL = str(Path(__file__).resolve().parents[2] / "models" / MODEL_NAME)
DEFAULT_SOURCE = "0"
WINDOW_NAME = "YOLO Detection"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the detection video demo."""
    parser = argparse.ArgumentParser(description="YOLO object detection video demo")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="YOLO detection model checkpoint",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Video path/URL or camera index (default: 0)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size for inference",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for detections",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for inference (e.g., 'cpu', '0', 'cuda:0')",
    )
    return parser.parse_args()


def resolve_source(source: str) -> str | int:
    """Convert numeric camera indices to int values."""
    return int(source) if source.isdigit() else source


def main() -> None:
    """Run YOLO detection on a video stream."""
    args = parse_args()
    source = resolve_source(args.source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise SystemExit(f"Unable to open video source: {args.source}")

    model = YOLO(args.model)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model.predict(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )
        annotated = results[0].plot()

        cv2.imshow(WINDOW_NAME, annotated)
        if cv2.waitKey(1) in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

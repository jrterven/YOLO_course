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


def annotate_detection(result) -> "cv2.typing.MatLike":
    """Draw detection results using OpenCV."""
    image = result.orig_img.copy()
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None:
        return image

    # result.boxes.xyxy -> (N, 4) tensor with [x1, y1, x2, y2]
    # result.boxes.conf -> (N,) tensor with confidence per box
    # result.boxes.cls -> (N,) tensor with class indices
    for xyxy, conf, cls_id in zip(boxes.xyxy, boxes.conf, boxes.cls):
        x1, y1, x2, y2 = [int(v) for v in xyxy.tolist()]
        cls_id = int(cls_id)
        label = result.names[cls_id] if result.names else str(cls_id)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            f"{label}: {float(conf):.2f}",
            (x1, max(y1 - 6, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return image


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
        annotated = annotate_detection(results[0])

        cv2.imshow(WINDOW_NAME, annotated)
        if cv2.waitKey(1) in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""YOLO classification on live video or a video file.

Runs YOLO on frames from a webcam, RTSP stream, or video file and displays
annotated results in an OpenCV window.

Example:
    python scripts/classify/classify_video.py --source 0
    python scripts/classify/classify_video.py --source path/to/video.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_NAME = "yolo26n-cls.pt"
DEFAULT_MODEL = str(Path(__file__).resolve().parents[2] / "models" / MODEL_NAME)
DEFAULT_SOURCE = "0"
WINDOW_NAME = "YOLO Classification"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the classification video demo."""
    parser = argparse.ArgumentParser(description="YOLO classification video demo")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="YOLO classification model checkpoint",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Video path/URL or camera index (default: 0)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Image size for inference",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold for predictions",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for inference (e.g., 'cpu', '0', 'cuda:0')",
    )
    return parser.parse_args()


def annotate_classification(result) -> "cv2.typing.MatLike":
    """Draw classification results using OpenCV."""
    image = result.orig_img.copy()
    probs = result.probs
    if probs is None:
        return image

    # probs.top1 -> class index of the best prediction
    # probs.top1conf -> confidence score for that class
    # result.names maps class indices to label strings
    top1 = int(probs.top1)
    top1_conf = float(probs.top1conf)
    label = result.names[top1] if result.names else str(top1)
    cv2.putText(
        image,
        f"{label}: {top1_conf:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
    )

    # probs.top5 / probs.top5conf -> top-5 class indices and confidences
    if hasattr(probs, "top5") and hasattr(probs, "top5conf"):
        y_offset = 55
        for cls_id, conf in zip(probs.top5, probs.top5conf):
            cls_id = int(cls_id)
            conf = float(conf)
            label = result.names[cls_id] if result.names else str(cls_id)
            cv2.putText(
                image,
                f"{label}: {conf:.2f}",
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )
            y_offset += 18

    return image


def resolve_source(source: str) -> str | int:
    """Convert numeric camera indices to int values."""
    return int(source) if source.isdigit() else source


def main() -> None:
    """Run YOLO classification on a video stream."""
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
            device=args.device,
            verbose=False,
        )
        annotated = annotate_classification(results[0])

        cv2.imshow(WINDOW_NAME, annotated)
        if cv2.waitKey(1) in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

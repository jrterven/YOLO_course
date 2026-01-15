#!/usr/bin/env python3
"""YOLO instance segmentation on live video or a video file.

Runs YOLO segmentation on frames from a webcam, RTSP stream, or video file and
shows annotated results in an OpenCV window.

Example:
    python scripts/segment/segment_video.py --source 0
    python scripts/segment/segment_video.py --source path/to/video.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_NAME = "yolo26n-seg.pt"
DEFAULT_MODEL = str(Path(__file__).resolve().parents[2] / "models" / MODEL_NAME)
DEFAULT_SOURCE = "0"
WINDOW_NAME = "YOLO Segmentation"
SEGMENT_COLORS = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 128, 0),
    (128, 0, 255),
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the segmentation video demo."""
    parser = argparse.ArgumentParser(description="YOLO segmentation video demo")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="YOLO segmentation model checkpoint",
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


def annotate_segmentation(result) -> "cv2.typing.MatLike":
    """Draw segmentation masks and boxes using OpenCV."""
    image = result.orig_img.copy()
    masks = result.masks
    boxes = result.boxes

    # result.masks.xy -> list of (N_i, 2) polygons per instance
    # result.boxes.xyxy -> (N, 4) tensor for bounding boxes
    # result.boxes.cls/conf -> class indices and confidences
    if masks is not None and masks.xy is not None:
        for idx, polygon in enumerate(masks.xy):
            points = np.array(polygon, dtype=np.int32)
            cls_id = (
                int(boxes.cls[idx])
                if boxes is not None and boxes.cls is not None and idx < len(boxes.cls)
                else -1
            )
            color_index = cls_id if cls_id >= 0 else idx
            color = SEGMENT_COLORS[color_index % len(SEGMENT_COLORS)]
            overlay = image.copy()
            cv2.fillPoly(overlay, [points], color)
            image = cv2.addWeighted(overlay, 0.3, image, 0.7, 0)
            cv2.polylines(image, [points], True, color, 2)

            if boxes is not None and boxes.xyxy is not None and idx < len(boxes.xyxy):
                x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[idx].tolist()]
                cls_id = int(boxes.cls[idx]) if boxes.cls is not None else cls_id
                conf = float(boxes.conf[idx]) if boxes.conf is not None else 0.0
                label = result.names[cls_id] if result.names and cls_id >= 0 else str(cls_id)
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    image,
                    f"{label}: {conf:.2f}",
                    (x1, max(y1 - 6, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2,
                )

    return image


def resolve_source(source: str) -> str | int:
    """Convert numeric camera indices to int values."""
    return int(source) if source.isdigit() else source


def main() -> None:
    """Run YOLO segmentation on a video stream."""
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
        annotated = annotate_segmentation(results[0])

        cv2.imshow(WINDOW_NAME, annotated)
        if cv2.waitKey(1) in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

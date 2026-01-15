#!/usr/bin/env python3
"""YOLO pose estimation on live video or a video file.

Runs YOLO pose estimation on frames from a webcam, RTSP stream, or video file
and shows annotated results in an OpenCV window.

Example:
    python scripts/pose/pose_video.py --source 0
    python scripts/pose/pose_video.py --source path/to/video.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_NAME = "yolo26n-pose.pt"
DEFAULT_MODEL = str(Path(__file__).resolve().parents[2] / "models" / MODEL_NAME)
DEFAULT_SOURCE = "0"
WINDOW_NAME = "YOLO Pose"
POSE_CONF_THRESHOLD = 0.25

# COCO keypoint skeleton (17 keypoints). Indices follow the model output order.
COCO_SKELETON = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the pose video demo."""
    parser = argparse.ArgumentParser(description="YOLO pose estimation video demo")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="YOLO pose model checkpoint",
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


def annotate_pose(result) -> "cv2.typing.MatLike":
    """Draw pose keypoints and skeletons using OpenCV."""
    image = result.orig_img.copy()
    keypoints = result.keypoints
    boxes = result.boxes
    if keypoints is None or keypoints.xy is None:
        return image

    # result.keypoints.xy -> (N, K, 2) tensor of keypoint coordinates
    # result.keypoints.conf -> (N, K) tensor of per-keypoint confidence scores
    # result.boxes.xyxy -> (N, 4) tensor with [x1, y1, x2, y2] per person
    # result.boxes.conf/cls -> (N,) confidences and class indices (usually person)
    coords = keypoints.xy
    confs = keypoints.conf

    for person_idx, person_kpts in enumerate(coords):
        person_conf = confs[person_idx] if confs is not None else None
        if boxes is not None and boxes.xyxy is not None and person_idx < len(boxes.xyxy):
            x1, y1, x2, y2 = [int(v) for v in boxes.xyxy[person_idx].tolist()]
            box_conf = float(boxes.conf[person_idx]) if boxes.conf is not None else 0.0
            cls_id = int(boxes.cls[person_idx]) if boxes.cls is not None else -1
            label = result.names[cls_id] if result.names and cls_id >= 0 else str(cls_id)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(
                image,
                f"{label}: {box_conf:.2f}",
                (x1, max(y1 - 6, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        for kpt_idx, (x, y) in enumerate(person_kpts):
            conf = float(person_conf[kpt_idx]) if person_conf is not None else 1.0
            if conf < POSE_CONF_THRESHOLD:
                continue
            cv2.circle(image, (int(x), int(y)), 4, (0, 255, 0), -1)

        for start, end in COCO_SKELETON:
            if person_conf is not None:
                if (
                    float(person_conf[start]) < POSE_CONF_THRESHOLD
                    or float(person_conf[end]) < POSE_CONF_THRESHOLD
                ):
                    continue
            x1, y1 = person_kpts[start]
            x2, y2 = person_kpts[end]
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    return image


def resolve_source(source: str) -> str | int:
    """Convert numeric camera indices to int values."""
    return int(source) if source.isdigit() else source


def main() -> None:
    """Run YOLO pose estimation on a video stream."""
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
        annotated = annotate_pose(results[0])

        cv2.imshow(WINDOW_NAME, annotated)
        if cv2.waitKey(1) in (27, ord("q")):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

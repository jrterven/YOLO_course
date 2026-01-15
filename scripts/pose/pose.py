#!/usr/bin/env python3
"""YOLO26 Human Pose Estimation Script.

Run YOLO26 pose estimation on an image, folder, or URL. The default model is
`yolo26n-pose.pt`, trained to detect person keypoints.

Example:
    python scripts/pose/pose.py --source images/zidane.jpg
"""

from __future__ import annotations

import argparse

import cv2
from pathlib import Path

from ultralytics import YOLO

DEFAULT_SOURCE = str(Path(__file__).resolve().parents[2] / "images" / "zidane.jpg")
MODEL_NAME = "yolo26n-pose.pt"
DEFAULT_MODEL = str(Path(__file__).resolve().parents[2] / "models" / MODEL_NAME)
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
    """Parse CLI arguments for the pose demo."""
    parser = argparse.ArgumentParser(description="YOLO26 human pose estimation demo")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="YOLO26 pose model checkpoint",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Path/URL to an image, video, or folder",
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
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save outputs to the runs/ directory",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Display the annotated image in an OpenCV window",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving outputs",
    )
    parser.add_argument(
        "--project",
        default="runs/pose",
        help="Project directory for outputs",
    )
    parser.add_argument(
        "--name",
        default="predict",
        help="Name of the run inside the project directory",
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


def main() -> None:
    """Run YOLO26 pose estimation with the provided arguments."""
    args = parse_args()
    save_outputs = args.save and not args.no_save
    show_outputs = args.show

    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=save_outputs,
        project=args.project,
        name=args.name,
    )

    if show_outputs:
        for index, result in enumerate(results, start=1):
            annotated = annotate_pose(result)
            window_name = f"YOLO Pose ({index}/{len(results)})"
            cv2.imshow(window_name, annotated)
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)

    print("Pose estimation complete.")
    if save_outputs:
        print("Check the output in:", args.project)


if __name__ == "__main__":
    main()

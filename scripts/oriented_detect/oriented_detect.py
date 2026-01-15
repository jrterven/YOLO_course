#!/usr/bin/env python3
"""YOLO26 Oriented Object Detection (OBB) Script.

Run YOLO26 oriented bounding box detection on an image, folder, or URL. The
default model is `yolo26n-obb.pt`, trained for rotated object detection.

Example:
    python scripts/oriented_detect/oriented_detect.py --source images/boats.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO

DEFAULT_SOURCE = str(Path(__file__).resolve().parents[2] / "images" / "boats.jpg")
MODEL_NAME = "yolo26n-obb.pt"
DEFAULT_MODEL = str(Path(__file__).resolve().parents[2] / "models" / MODEL_NAME)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the oriented detection demo."""
    parser = argparse.ArgumentParser(description="YOLO26 oriented detection demo")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="YOLO26 oriented detection model checkpoint",
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
        default="runs/obb",
        help="Project directory for outputs",
    )
    parser.add_argument(
        "--name",
        default="predict",
        help="Name of the run inside the project directory",
    )
    return parser.parse_args()


def annotate_obb(result) -> "cv2.typing.MatLike":
    """Draw oriented bounding boxes using OpenCV."""
    image = result.orig_img.copy()
    obb = result.obb
    if obb is None or obb.xyxyxyxy is None:
        return image

    # result.obb.xyxyxyxy -> (N, 4, 2) polygon corners for each box
    # result.obb.conf -> (N,) confidence scores
    # result.obb.cls -> (N,) class indices
    for polygon, conf, cls_id in zip(obb.xyxyxyxy, obb.conf, obb.cls):
        points = np.array(polygon.tolist(), dtype=np.int32)
        cv2.polylines(image, [points], True, (0, 255, 0), 2)

        cls_id = int(cls_id)
        label = result.names[cls_id] if result.names else str(cls_id)
        text_origin = tuple(points[0])
        cv2.putText(
            image,
            f"{label}: {float(conf):.2f}",
            text_origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return image


def main() -> None:
    """Run YOLO26 oriented detection with the provided arguments."""
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
            annotated = annotate_obb(result)
            window_name = f"YOLO Oriented Detection ({index}/{len(results)})"
            cv2.imshow(window_name, annotated)
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)

    print("Oriented detection complete.")
    if save_outputs:
        print("Check the output in:", args.project)


if __name__ == "__main__":
    main()

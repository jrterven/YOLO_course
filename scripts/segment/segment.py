#!/usr/bin/env python3
"""YOLO26 Instance Segmentation Script.

Run YOLO26 instance segmentation on an image, folder, or URL. The default model
is `yolo26n-seg.pt`, suitable for fast segmentation demos.

Example:
    python scripts/segment.py --source https://ultralytics.com/images/bus.jpg
"""

from __future__ import annotations

import argparse

import cv2
import numpy as np
from pathlib import Path

from ultralytics import YOLO

DEFAULT_SOURCE = "https://ultralytics.com/images/bus.jpg"
MODEL_NAME = "yolo26n-seg.pt"
DEFAULT_MODEL = str(Path(__file__).resolve().parents[2] / "models" / MODEL_NAME)
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
    """Parse CLI arguments for the segmentation demo."""
    parser = argparse.ArgumentParser(description="YOLO26 instance segmentation demo")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="YOLO26 segmentation model checkpoint",
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
        default="runs/segment",
        help="Project directory for outputs",
    )
    parser.add_argument(
        "--name",
        default="predict",
        help="Name of the run inside the project directory",
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


def main() -> None:
    """Run YOLO26 instance segmentation with the provided arguments."""
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
            annotated = annotate_segmentation(result)
            window_name = f"YOLO Segmentation ({index}/{len(results)})"
            cv2.imshow(window_name, annotated)
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)

    print("Segmentation complete.")
    if save_outputs:
        print("Check the output in:", args.project)


if __name__ == "__main__":
    main()

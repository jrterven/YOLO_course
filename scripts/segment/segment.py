#!/usr/bin/env python3
"""YOLO26 Instance Segmentation Script.

Run YOLO26 instance segmentation on an image, folder, or URL. The default model
is `yolo26n-seg.pt`, suitable for fast segmentation demos.

Example:
    python scripts/segment.py --source https://ultralytics.com/images/bus.jpg
"""

from __future__ import annotations

import argparse

from ultralytics import YOLO

DEFAULT_SOURCE = "https://ultralytics.com/images/bus.jpg"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the segmentation demo."""
    parser = argparse.ArgumentParser(description="YOLO26 instance segmentation demo")
    parser.add_argument(
        "--model",
        default="yolo26n-seg.pt",
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


def main() -> None:
    """Run YOLO26 instance segmentation with the provided arguments."""
    args = parse_args()
    save_outputs = args.save and not args.no_save

    model = YOLO(args.model)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        save=save_outputs,
        project=args.project,
        name=args.name,
    )

    print("Segmentation complete.")
    if save_outputs:
        print("Check the output in:", args.project)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""YOLO26 Image Classification Script.

Run YOLO26 classification on an image, folder, or URL. The default model is
`yolo26n-cls.pt`, which is the smallest and fastest YOLO26 classifier.

Example:
    python scripts/classify/classify.py --source images/bus.jpg
"""

from __future__ import annotations
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

DEFAULT_SOURCE = str(Path(__file__).resolve().parents[2] / "images" / "bus.jpg")
MODEL_NAME = "yolo26n-cls.pt"
DEFAULT_MODEL = str(Path(__file__).resolve().parents[2] / "models" / MODEL_NAME)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the classification demo."""
    parser = argparse.ArgumentParser(description="YOLO26 image classification demo")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="YOLO26 classification model checkpoint",
    )
    parser.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="Path/URL to an image or folder to classify",
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
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save outputs to the runs/ directory",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=True,
        help="Display the annotated image in an OpenCV window",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Disable saving outputs",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable displaying the OpenCV window",
    )
    parser.add_argument(
        "--project",
        default="runs/classify",
        help="Project directory for outputs",
    )
    parser.add_argument(
        "--name",
        default="predict",
        help="Name of the run inside the project directory",
    )
    return parser.parse_args()


def main() -> None:
    """Run YOLO26 classification with the provided arguments."""
    args = parse_args()
    save_outputs = args.save and not args.no_save
    show_outputs = args.show and not args.no_show

    model = YOLO(args.model)
    results = model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=save_outputs,
        project=args.project,
        name=args.name,
    )

    if show_outputs:
        for index, result in enumerate(results, start=1):
            annotated = result.plot()
            window_name = f"YOLO Classification ({index}/{len(results)})"
            cv2.imshow(window_name, annotated)
            cv2.waitKey(0)
            cv2.destroyWindow(window_name)

    print("Classification complete.")
    if save_outputs:
        print("Check the output in:", args.project)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate a trained Ants vs Bees classifier on the test split.

Example:
    python scripts/classify/test_ants_bees.py \
      --model /Users/juanterven/dev/yolo_course/notebooks/runs/ants_bees_cls/weights/best.pt \
      --data /Users/juanterven/data/ants_bees_yolo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

DEFAULT_DATA = Path("/Users/juanterven/data/ants_bees_yolo")
DEFAULT_MODEL = Path(
    "/Users/juanterven/dev/yolo_course/notebooks/runs/ants_bees_cls/weights/best.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Ants vs Bees classification model")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to the trained model weights",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="Path to the prepared Ants vs Bees dataset root",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=224,
        help="Image size for evaluation",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device for evaluation (e.g., 'cpu', '0', 'cuda:0')",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    model = YOLO(args.model)
    metrics = model.val(
        data=str(data_path),
        split="test",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="runs",
        name="ants_bees_test",
    )

    print("Test metrics:")
    print(metrics)


if __name__ == "__main__":
    main()

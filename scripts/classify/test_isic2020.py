#!/usr/bin/env python3
"""Evaluate a trained ISIC 2020 classifier on the test split.

Example:
    python scripts/classify/test_isic2020.py \
      --model runs/isic2020_cls/weights/best.pt \
      --data data/isic2020_yolo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO

DEFAULT_DATA = Path(__file__).resolve().parents[2] / "data" / "isic2020_yolo"
DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "runs" / "isic2020_cls" / "weights" / "best.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ISIC 2020 classification model")
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to the trained model weights",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="Path to the prepared ISIC 2020 dataset root",
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
        name="isic2020_test",
    )

    print("Test metrics:")
    print(metrics)


if __name__ == "__main__":
    main()

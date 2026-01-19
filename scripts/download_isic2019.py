#!/usr/bin/env python3
"""Download and extract the ISIC 2019 dataset assets.

Example:
    python scripts/download_isic2019.py --output-dir /path/to/isic2019_raw
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

TRAIN_IMAGES_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_Input.zip"
)
TRAIN_LABELS_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Training_GroundTruth.csv"
)
TEST_IMAGES_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_Input.zip"
)
TEST_LABELS_URL = (
    "https://isic-archive.s3.amazonaws.com/challenges/2019/ISIC_2019_Test_GroundTruth.csv"
)

ASSETS = (
    ("ISIC_2019_Training_Input", TRAIN_IMAGES_URL, True),
    ("ISIC_2019_Training_GroundTruth.csv", TRAIN_LABELS_URL, False),
    ("ISIC_2019_Test_Input", TEST_IMAGES_URL, True),
    ("ISIC_2019_Test_GroundTruth.csv", TEST_LABELS_URL, False),
)

DEFAULT_OUTPUT_DIR = Path("data/isic2019_raw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ISIC 2019 assets")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to store the raw ISIC 2019 files",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract assets even if they already exist",
    )
    return parser.parse_args()


def download_file(url: str, destination: Path, force: bool) -> None:
    if destination.exists() and not force:
        print(f"Skipping download (already exists): {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and force:
        destination.unlink()

    def reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = min(block_num * block_size, total_size)
        percent = downloaded / total_size * 100
        print(
            f"\rDownloading {destination.name}: {percent:5.1f}% ({downloaded/1e6:.1f}/{total_size/1e6:.1f} MB)",
            end="",
        )

    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, destination, reporthook=reporthook)
    print()


def extract_zip(zip_path: Path, output_dir: Path, target_dir: Path, force: bool) -> None:
    if target_dir.exists():
        if not force:
            print(f"Skipping extraction (already exists): {target_dir}")
            return
        shutil.rmtree(target_dir)

    print(f"Extracting {zip_path.name} -> {output_dir}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, url, is_zip in ASSETS:
        if is_zip:
            zip_path = output_dir / f"{name}.zip"
            target_dir = output_dir / name
            download_file(url, zip_path, args.force)
            extract_zip(zip_path, output_dir, target_dir, args.force)
        else:
            csv_path = output_dir / name
            download_file(url, csv_path, args.force)

    print("\nDownload complete. Raw ISIC 2019 data stored at:", output_dir)


if __name__ == "__main__":
    main()

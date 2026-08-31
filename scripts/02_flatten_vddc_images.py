#!/usr/bin/env python
"""One-off VDD-C raw-layout fix: flatten its 33 per-session image folders.

VDD-C's images.zip extracts to one directory per dive session
(data/raw/vddc/barbados_scuba_001_A/*.jpg, .../pool_flipper_002_A/*.jpg, ...).
06_unify_datasets.py's yolo reader wants one flat images/ + labels/ pair
(configs/datasets.yaml -> vddc.layout), so this moves every session's images
into data/raw/vddc/images_flat/, prefixing each filename with its session id
("barbados_scuba_011_A_2550.jpg") so the pair with 02_flatten_vddc_labels.py
stays unique. That prefix is also what vddc's group regex in
configs/datasets.yaml parses back out to keep a dive's frames together at
split time.

Moves are in-place (Path.rename), so a second run is a no-op: the per-session
folders are already empty of .jpg files.

Usage
    python scripts/02_flatten_vddc_images.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SKIP = {"yolo", "images_flat", "labels_flat", "annotations"}


def main() -> None:
    vddc_dir = ROOT / "data/raw/vddc"
    out_dir = vddc_dir / "images_flat"
    out_dir.mkdir(exist_ok=True)

    moved = 0
    for subdir in vddc_dir.iterdir():
        if subdir.is_dir() and subdir.name not in SKIP:
            for img in subdir.glob("*.jpg"):
                img.rename(out_dir / f"{subdir.name}_{img.name}")
                moved += 1

    print(f"Moved {moved} images")


if __name__ == "__main__":
    main()

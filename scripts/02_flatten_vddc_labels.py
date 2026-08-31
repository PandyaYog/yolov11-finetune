#!/usr/bin/env python
"""One-off VDD-C raw-layout fix: flatten its per-session YOLO label folders.

Companion to 02_flatten_vddc_images.py. VDD-C's yolo_labels.zip extracts to
train/val/test subfolders, with filenames already session-prefixed
("barbados_scuba_011_A_2550.txt") to match the images.zip frame numbering.
This just collects every .txt into data/raw/vddc/labels_flat/, matching the
flat images_flat/ tree that configs/datasets.yaml's vddc.layout expects and
that 02_flatten_vddc_images.py builds -- the yolo reader pairs a label to an
image by stem.

Moves are in-place (Path.rename), so a second run is a no-op: matching
train/val/test folders no longer hold any .txt files.

Usage
    python scripts/02_flatten_vddc_labels.py
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    vddc_dir = ROOT / "data/raw/vddc"
    out_dir = vddc_dir / "labels_flat"
    out_dir.mkdir(exist_ok=True)

    moved = 0
    for txt in vddc_dir.rglob("*.txt"):
        if txt.parent.name in ("train", "val", "test"):
            txt.rename(out_dir / txt.name)
            moved += 1

    print(f"Moved {moved} labels")


if __name__ == "__main__":
    main()

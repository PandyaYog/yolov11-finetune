#!/usr/bin/env python
"""Decode Singapore Maritime Dataset (SMD) videos + convert its .mat ground
truth into the flat YOLO images/labels tree configs/datasets.yaml -> smd
expects (data/raw/smd/images/, data/raw/smd/labels/).

SMD ships as three category folders, each with its own videos and ground
truth:
    VIS_Onshore/Videos/*.avi   VIS_Onshore/ObjectGT/*_ObjectGT.mat
    VIS_Onboard/Videos/*.avi   VIS_Onboard/ObjectGT/*_ObjectGT.mat
    NIR/Videos/*               NIR/ObjectGT/*_ObjectGT.mat        (skipped by
                                                                    default --
                                                                    thermal,
                                                                    not what
                                                                    the RGB
                                                                    OAK-D-Lite
                                                                    sees)

Each *_ObjectGT.mat holds one MATLAB struct array `structXML` with one entry
per video frame, fields `BB` (per-object [x_min, y_min, width, height], pixel
absolute) and `Object` (per-object class int, 1-10, matching the `names:`
list already configured for smd in configs/datasets.yaml -- 0 means "invalid
entry", skipped). Parsing logic here mirrors a working reference converter
(github.com/tilemmpon/Singapore-Maritime-Dataset-Frames-Ground-Truth-
Generation-and-Statistics/blob/master/load_mat_into_csv_xml.py), adapted to
write YOLO .txt instead of VOC XML, since that repo's own scipy.io.loadmat
indexing is the proven-correct way to unpack this specific struct layout --
not something worth re-deriving from scratch against a citation alone.

Only enabled by default: the single reason this is being built at all is
`buoy_marker`, which currently has ZERO examples anywhere in the unified
corpus (see configs/taxonomy.yaml -- smd is the only enabled-capable source
with a real Buoy class). SMD has ~180 videos across categories; decoding all
of them costs real time and disk for classes we don't need. So this parses
every .mat file first (cheap -- no video touched) and only extracts frames
for videos that contain at least one instance of --classes (default: Buoy).
Broaden with --classes if you want SMD for other thin classes too.

Usage
    python scripts/02_extract_smd_frames.py --dry-run    # which videos qualify, no decode
    python scripts/02_extract_smd_frames.py
    python scripts/02_extract_smd_frames.py --classes Buoy "Swimming person"
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uwd.util import log, setup_logging  # noqa: E402

CATEGORIES = ["VIS_Onshore", "VIS_Onboard"]  # NIR excluded by default -- see docstring

# 1-indexed in the .mat (0 = invalid entry) -> our 0-indexed configs/datasets.yaml smd.names
CLASSES = ["Ferry", "Buoy", "Vessel/ship", "Speed boat", "Boat", "Kayak",
          "Sail boat", "Swimming person", "Flying bird/plane", "Other"]


def parse_mat(mat_path: Path) -> list[tuple[int, list[tuple[int, float, float, float, float]]]]:
    """-> list of (frame_index, [(class_idx0, xmin, ymin, w, h), ...]) in pixels."""
    gt = loadmat(str(mat_path))
    struct = gt["structXML"][0]
    n_frames = len(struct)
    out = []
    for i in range(n_frames):
        bb = struct["BB"][i]
        objects = struct["Object"][i]
        boxes = []
        if len(objects) > 0 and len(objects[0]) > 0:
            for j in range(len(objects)):
                cls1 = int(objects[j][0])
                if cls1 == 0:
                    continue  # bad entry -- see docstring
                xmin, ymin, w, h = (float(v) for v in bb[j, :4])
                boxes.append((cls1 - 1, xmin, ymin, w, h))
        out.append((i, boxes))
    return out


def find_video(videos_dir: Path, key: str) -> Path | None:
    matches = list(videos_dir.glob(f"{key}.*"))
    return matches[0] if matches else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "data/raw/smd"),
                    help="unzipped SMD root -- must contain VIS_Onshore/, VIS_Onboard/ (see docstring)")
    ap.add_argument("--categories", nargs="+", default=CATEGORIES)
    ap.add_argument("--classes", nargs="+", default=["Buoy"],
                    help="only extract videos containing at least one of these classes")
    ap.add_argument("--images-out", default=str(ROOT / "data/raw/smd/images"))
    ap.add_argument("--labels-out", default=str(ROOT / "data/raw/smd/labels"))
    ap.add_argument("--dry-run", action="store_true",
                    help="report which videos qualify and how many objects they'd contribute, decode nothing")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)

    if not args.dry_run and shutil.which("ffmpeg") is None:
        log.error("ffmpeg not found on PATH -- install it (e.g. apt install ffmpeg)")
        return 1

    root = Path(args.root)
    if not root.exists():
        log.error("no such directory: %s -- download SMD manually first "
                  "(see the note in configs/datasets.yaml -> smd), unzip it here", root)
        return 1

    target_classes = {c.lower() for c in args.classes}
    target_idx = {i for i, name in enumerate(CLASSES) if name.lower() in target_classes}
    if not target_idx:
        log.error("--classes %s matched none of: %s", args.classes, CLASSES)
        return 1

    images_out = Path(args.images_out)
    labels_out = Path(args.labels_out)
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    qualifying: list[tuple[str, Path, Path, list]] = []  # (key, mat_path, videos_dir, per_frame)
    for category in args.categories:
        gt_dir = root / category / "ObjectGT"
        videos_dir = root / category / "Videos"
        if not gt_dir.exists():
            log.warning("[%s] no such directory: %s -- skipping category", category, gt_dir)
            continue
        for mat_path in sorted(gt_dir.glob("*.mat")):
            key = mat_path.name.replace("_ObjectGT.mat", "")
            try:
                per_frame = parse_mat(mat_path)
            except Exception as e:  # mat internals can vary; never let one bad file kill the run
                log.error("[%s] failed to parse %s: %s", category, mat_path.name, e)
                continue
            n_target_boxes = sum(1 for _, boxes in per_frame for cid, *_ in boxes if cid in target_idx)
            if n_target_boxes == 0:
                continue
            qualifying.append((key, mat_path, videos_dir, per_frame))
            log.info("[%s] %-24s qualifies -- %d target-class boxes across %d frames",
                     category, key, n_target_boxes, len(per_frame))

    log.info("%d videos qualify out of the categories scanned", len(qualifying))
    if args.dry_run:
        log.info("dry run -- nothing decoded")
        return 0
    if not qualifying:
        return 0

    failed = 0
    for i, (key, mat_path, videos_dir, per_frame) in enumerate(qualifying, 1):
        video = find_video(videos_dir, key)
        if video is None:
            log.error("[%d/%d] %s: no matching video under %s", i, len(qualifying), key, videos_dir)
            failed += 1
            continue

        already = sorted(images_out.glob(f"{key}_frame*.jpg"))
        if len(already) >= len(per_frame):
            log.info("[%d/%d] %s already extracted (%d frames) -- skipping",
                     i, len(qualifying), key, len(already))
        else:
            log.info("[%d/%d] decoding %s (%d frames)", i, len(qualifying), key, len(per_frame))
            cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video),
                  "-start_number", "0", str(images_out / f"{key}_frame%d.jpg")]
            rc = subprocess.call(cmd)
            if rc != 0:
                log.error("  ffmpeg failed (rc=%d) on %s", rc, video)
                failed += 1
                continue

        img_w = img_h = None
        for frame_idx, boxes in per_frame:
            if not boxes:
                continue
            img_path = images_out / f"{key}_frame{frame_idx}.jpg"
            if not img_path.exists():
                continue  # video shorter than the GT claims -- not fatal, just skip
            if img_w is None:
                img = cv2.imread(str(img_path))
                if img is None:
                    log.error("  unreadable frame, skipping label pass: %s", img_path)
                    break
                img_h, img_w = img.shape[:2]

            lines = []
            for cid, xmin, ymin, w, h in boxes:
                cx, cy = (xmin + w / 2) / img_w, (ymin + h / 2) / img_h
                wn, hn = w / img_w, h / img_h
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < wn <= 1 and 0 < hn <= 1):
                    continue
                lines.append(f"{cid} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
            if lines:
                (labels_out / f"{key}_frame{frame_idx}.txt").write_text("\n".join(lines) + "\n",
                                                                        encoding="utf-8")

    n_images = sum(1 for _ in images_out.glob("*.jpg"))
    n_labels = sum(1 for _ in labels_out.glob("*.txt"))
    log.info("done -- %s holds %d frames, %s holds %d labels (%d videos failed)",
             images_out, n_images, labels_out, n_labels, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

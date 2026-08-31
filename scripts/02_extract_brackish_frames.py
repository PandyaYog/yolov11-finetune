#!/usr/bin/env python
"""Decode Brackish's raw .avi videos into the PNG frames its COCO annotations
expect.

The Kaggle/GitHub mirror of Brackish ships 89 short video clips
(data/raw/brackish/dataset/videos/<class>/*.avi) plus the dataset's own
scripts/frameExtractor.py to turn them into frames -- but running that script
is a manual step nobody did, so data/raw/brackish has no images/ directory at
all and 01_download_datasets.py / 06_unify_datasets.py have nothing to read.

This reproduces frameExtractor.py's exact behaviour (ffmpeg, scale=960:540,
bicubic, "<video-stem>-%04d.png") so filenames match what
annotations/annotations_COCO/{train,valid,test}_groundtruth.json already
reference, into one flat output directory (video timestamps are unique across
categories, so a flat pool is safe and is what configs/datasets.yaml expects).

Usage
    python scripts/02_extract_brackish_frames.py
    python scripts/02_extract_brackish_frames.py --dry-run
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uwd.util import log, setup_logging  # noqa: E402

VIDEO_EXTS = {".avi", ".mp4"}


def already_extracted(video: Path, out_dir: Path) -> bool:
    return any(out_dir.glob(f"{video.stem}-*.png"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", default=str(ROOT / "data/raw/brackish/dataset/videos"))
    ap.add_argument("--out", default=str(ROOT / "data/raw/brackish/images_extracted"))
    ap.add_argument("--dry-run", action="store_true", help="list what would run, decode nothing")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)

    if shutil.which("ffmpeg") is None:
        log.error("ffmpeg not found on PATH -- install it (e.g. apt install ffmpeg)")
        return 1

    videos_dir = Path(args.videos)
    out_dir = Path(args.out)
    if not videos_dir.exists():
        log.error("no such directory: %s (has brackish been downloaded?)", videos_dir)
        return 1

    videos = sorted(p for p in videos_dir.rglob("*") if p.suffix.lower() in VIDEO_EXTS)
    if not videos:
        log.error("no video files under %s", videos_dir)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    todo = [v for v in videos if not already_extracted(v, out_dir)]
    log.info("%d videos total, %d already extracted, %d to decode",
             len(videos), len(videos) - len(todo), len(todo))

    if args.dry_run:
        for v in todo:
            log.info("  would decode %s", v.relative_to(videos_dir))
        return 0

    failed = 0
    for i, video in enumerate(todo, 1):
        log.info("[%d/%d] %s", i, len(todo), video.relative_to(videos_dir))
        prefix = out_dir / video.stem
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video),
               "-vf", "scale=960:540", "-sws_flags", "bicubic",
               f"{prefix}-%04d.png"]
        rc = subprocess.call(cmd)
        if rc != 0 or not already_extracted(video, out_dir):
            log.error("  ffmpeg failed (rc=%d) on %s", rc, video)
            failed += 1

    n_frames = sum(1 for _ in out_dir.glob("*.png"))
    log.info("done -- %s now holds %d frames (%d videos failed)",
             out_dir, n_frames, failed)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

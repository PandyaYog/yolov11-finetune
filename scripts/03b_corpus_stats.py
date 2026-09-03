#!/usr/bin/env python
"""Report what a built corpus actually contains: images and boxes per class,
per split, and per source.

scripts/03_unify_datasets.py prints a class summary as it builds. This reads a
corpus that already exists, so you can re-check one without rebuilding it, and
diff two builds against each other after a config change.

Counts BOTH images and boxes per class, which is the distinction that matters
for a detector: 31k benthic_invert boxes sounds like plenty until you notice
they come from 6k images at ~5 boxes each, while a class with 900 boxes spread
over 500 images has far less scene diversity than the box count suggests.

Usage
    python scripts/03b_corpus_stats.py
    python scripts/03b_corpus_stats.py --data data/unified_enhanced
    python scripts/03b_corpus_stats.py --by-source
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uwd.util import log, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "unified"),
                    help="corpus root (must contain dataset.yaml + labels/)")
    ap.add_argument("--manifest", default=None,
                    help="defaults to <data>/manifest.csv, else data/unified/manifest.csv")
    ap.add_argument("--by-source", action="store_true",
                    help="also break each class down by which dataset supplied it")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    root = Path(args.data)
    yaml_path = root / "dataset.yaml"
    if not yaml_path.exists():
        log.error("no dataset.yaml at %s -- run scripts/03_unify_datasets.py first", root)
        return 1
    names: dict[int, str] = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))["names"]

    manifest = Path(args.manifest) if args.manifest else root / "manifest.csv"
    if not manifest.exists():
        manifest = ROOT / "data" / "unified" / "manifest.csv"
    src_of: dict[str, str] = {}
    if manifest.exists():
        with manifest.open(encoding="utf-8") as f:
            src_of = {r["out_name"]: r["dataset"] for r in csv.DictReader(f)}
    elif args.by_source:
        log.warning("--by-source needs a manifest.csv; none found")

    boxes: Counter = Counter()
    imgs: Counter = Counter()
    per_split_imgs: Counter = Counter()
    per_split_cls_imgs: Counter = Counter()
    by_source: Counter = Counter()
    negatives = 0
    total_imgs = 0

    for split in ("train", "val", "test"):
        ldir = root / "labels" / split
        if not ldir.is_dir():
            continue
        for lbl in sorted(ldir.glob("*.txt")):
            total_imgs += 1
            per_split_imgs[split] += 1
            present: set[int] = set()
            for line in lbl.read_text(encoding="utf-8").splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                cid = int(p[0])
                boxes[cid] += 1
                present.add(cid)
            if not present:
                negatives += 1
            for cid in present:
                imgs[cid] += 1
                per_split_cls_imgs[(cid, split)] += 1
                if src_of:
                    by_source[(cid, src_of.get(lbl.stem, "?"))] += 1

    tot_boxes = sum(boxes.values())
    print()
    print(f"  corpus: {root}")
    print(f"  {total_imgs} images  ({', '.join(f'{s} {per_split_imgs[s]}' for s in ('train','val','test'))})")
    print(f"  {tot_boxes} boxes across {len(names)} classes")
    print(f"  {negatives} object-free images ({100*negatives/total_imgs:.1f}%)" if total_imgs else "")
    print()
    print(f"  {'class':16s} {'images':>7s} {'boxes':>8s} {'box%':>7s} {'bx/img':>7s} "
          f"{'train':>7s} {'val':>6s} {'test':>6s}")
    print("  " + "-" * 74)
    for cid in sorted(names):
        b, i = boxes[cid], imgs[cid]
        print(f"  {names[cid]:16s} {i:7d} {b:8d} "
              f"{(100*b/tot_boxes if tot_boxes else 0):6.1f}% "
              f"{(b/i if i else 0):7.1f} "
              f"{per_split_cls_imgs[(cid,'train')]:7d} "
              f"{per_split_cls_imgs[(cid,'val')]:6d} "
              f"{per_split_cls_imgs[(cid,'test')]:6d}")
    print("  " + "-" * 74)
    print(f"  {'TOTAL':16s} {total_imgs:7d} {tot_boxes:8d}")

    # A class with zero val or test images has no honest held-out read, however
    # good its training numbers look -- this is how rope_net and buoy_marker
    # went unnoticed until the first real evaluation.
    blind = [names[c] for c in sorted(names)
             if imgs[c] and not (per_split_cls_imgs[(c, "val")] and per_split_cls_imgs[(c, "test")])]
    if blind:
        print()
        print(f"  WARNING: no val AND/OR test images for: {', '.join(blind)}")
        print("           these classes cannot be evaluated honestly on this split.")
    empty = [names[c] for c in sorted(names) if not imgs[c]]
    if empty:
        print(f"  WARNING: zero instances anywhere for: {', '.join(empty)}")

    if args.by_source and by_source:
        print()
        print("  per-class source breakdown (images):")
        for cid in sorted(names):
            rows = sorted(((s, n) for (c, s), n in by_source.items() if c == cid),
                          key=lambda x: -x[1])
            if not rows:
                continue
            print(f"    {names[cid]}")
            for s, n in rows:
                print(f"      {s:16s} {n:6d}  ({100*n/imgs[cid]:5.1f}%)")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

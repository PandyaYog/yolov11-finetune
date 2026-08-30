#!/usr/bin/env python
"""Merge every source dataset into one YOLO-format corpus at the robot's geometry.

Pipeline per dataset:
    read native format  ->  map to unified taxonomy  ->  temporal stride
    ->  near-duplicate removal  ->  rarity-aware cap  ->  group-aware split
    ->  crop/resize to 640x480  ->  write image + label

Output (data/unified/):
    images/{train,val,test}/*.jpg      640x480, 4:3, robot geometry
    labels/{train,val,test}/*.txt      YOLO normalised cx cy w h
    dataset.yaml                       plain Ultralytics config
    dataset_rfs.yaml                   same, but train list is repeat-factor sampled
    train_rfs.txt                      oversampled train list for rare classes
    manifest.csv                       provenance for every output image
    report.txt                         class counts, drops, per-source contribution

Usage
    python scripts/unify_datasets.py --dry-run
    python scripts/unify_datasets.py
    python scripts/unify_datasets.py --dataset trashcan --dataset suim
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import logging
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uwd.geometry import clamp_boxes, fit_image          # noqa: E402
from uwd.readers import READERS, Box, Sample             # noqa: E402
from uwd.util import (  # noqa: E402
    dhash, hamming, log, resolve_path, setup_logging, stable_bucket,
)


# --------------------------------------------------------------------------- #
# Taxonomy mapping
# --------------------------------------------------------------------------- #

class Taxonomy:
    def __init__(self, tax: dict):
        self.names: dict[int, str] = dict(tax["names"])
        self.id_of: dict[str, int] = {v: k for k, v in self.names.items()}
        self.maps: dict[str, dict] = tax.get("maps", {})
        self.dropped: Counter = Counter()

    def map_label(self, dataset: str, label: str) -> int | None:
        """Source label -> class id, or None if this label is deliberately dropped."""
        table = self.maps.get(dataset, {})
        if label in table:
            return self.id_of.get(table[label])

        lowered = label.strip().lower()
        for key, val in table.items():
            if key.lower() == lowered:
                return self.id_of.get(val)

        # Identity pass-through: the FathomNet exporter already writes unified
        # class names as its COCO categories, so no per-concept table is needed.
        if lowered in self.id_of:
            return self.id_of[lowered]

        self.dropped[f"{dataset}:{label}"] += 1
        return None


# --------------------------------------------------------------------------- #
# Per-dataset collection
# --------------------------------------------------------------------------- #

class Mapped:
    """A Sample with taxonomy applied and class ids resolved."""

    __slots__ = ("sample", "labels", "split", "out_name")

    def __init__(self, sample: Sample, labels: list[tuple[int, float, float, float, float]]):
        self.sample = sample
        self.labels = labels
        self.split = ""
        self.out_name = ""


def collect(name: str, cfg: dict, raw_root: Path, tax: Taxonomy) -> list[Mapped]:
    layout = cfg["layout"]
    reader = READERS.get(layout["reader"])
    if reader is None:
        log.error("[%s] unknown reader %r", name, layout["reader"])
        return []

    extra = {k: v for k, v in layout.items() if k not in ("reader", "parts")}
    out: list[Mapped] = []
    stats = Counter()

    # Layout paths are relative to this dataset's own directory, and are
    # resolved by glob so a Kaggle mirror that nests things differently from
    # the GitHub release does not need a config edit.
    ds_root = raw_root / name
    ann_is_dir = layout["reader"] != "coco"

    for part in layout["parts"]:
        ann = resolve_path(ds_root, part["ann"], want_dir=ann_is_dir)
        images = resolve_path(ds_root, part["images"], want_dir=True)
        if ann is None or images is None:
            log.warning("[%s] could not resolve part ann=%r images=%r -- "
                        "run: python scripts/download_datasets.py --inspect %s",
                        name, part["ann"], part["images"], name)
            continue
        for sample in reader(name, ann, images, cfg.get("group", {}), **extra):
            stats["read"] += 1
            source_boxes = len(sample.boxes)

            labels = []
            for b in sample.boxes:
                cid = tax.map_label(name, b.label)
                if cid is not None:
                    labels.append((cid, b.cx, b.cy, b.w, b.h))

            if not labels:
                # An image whose annotations were ALL dropped by the mapping is
                # not a negative -- it demonstrably contains an object we have
                # chosen not to name. Training on it as empty teaches the model
                # that a real object is background. Only source-level
                # annotation-free frames are safe negatives.
                if source_boxes > 0:
                    stats["dropped_all_labels"] += 1
                    continue
                if not sample.negative:
                    stats["no_annotations"] += 1
                    continue
                stats["negative"] += 1
            else:
                stats["kept"] += 1

            out.append(Mapped(sample, labels))

    log.info("[%s] read=%d kept=%d neg=%d all-labels-dropped=%d unannotated=%d",
             name, stats["read"], stats["kept"], stats["negative"],
             stats["dropped_all_labels"], stats["no_annotations"])
    return out


# --------------------------------------------------------------------------- #
# Reduction: stride -> dedupe -> cap
# --------------------------------------------------------------------------- #

def by_group(items: list[Mapped]) -> dict[str, list[Mapped]]:
    groups: dict[str, list[Mapped]] = defaultdict(list)
    for m in items:
        groups[m.sample.group].append(m)
    for g in groups.values():
        g.sort(key=lambda m: m.sample.image_path.name)
    return groups


def apply_stride(items: list[Mapped], stride: int) -> list[Mapped]:
    """Temporal subsampling WITHIN each group. Video-derived sets carry huge
    frame-to-frame redundancy; 14.5k Brackish frames from 89 videos is closer
    to 2-3k images of real information."""
    if stride <= 1:
        return items
    out: list[Mapped] = []
    for group in by_group(items).values():
        out.extend(group[::stride])
    return out


def dedupe(items: list[Mapped], threshold: int, workers: int) -> list[Mapped]:
    """Drop near-identical frames by difference-hash, comparing only within a
    group (cross-group collisions are almost always coincidence)."""
    if threshold <= 0:
        return items

    def hash_one(m: Mapped) -> tuple[Mapped, int | None]:
        img = cv2.imread(str(m.sample.image_path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
        return m, (dhash(img) if img is not None else None)

    hashes: dict[Mapped, int] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for m, h in tqdm(pool.map(hash_one, items), total=len(items),
                         desc="  dedupe-hash", leave=False):
            if h is not None:
                hashes[m] = h

    kept: list[Mapped] = []
    for group in by_group([m for m in items if m in hashes]).values():
        seen: list[int] = []
        for m in group:
            h = hashes[m]
            if any(hamming(h, s) <= threshold for s in seen):
                continue
            seen.append(h)
            kept.append(m)
    return kept


def cap(items: list[Mapped], limit: int | None) -> list[Mapped]:
    """Reduce to `limit` images, preferring images that carry rare classes while
    round-robining across groups so no whole video/dive is lost."""
    if not limit or len(items) <= limit:
        return items

    freq = Counter(cid for m in items for cid, *_ in m.labels)
    total = sum(freq.values()) or 1

    def rarity(m: Mapped) -> float:
        if not m.labels:
            return 0.0
        return sum(total / max(freq[cid], 1) for cid, *_ in m.labels)

    groups = by_group(items)
    for g in groups.values():
        g.sort(key=rarity, reverse=True)

    out: list[Mapped] = []
    cursors = {k: 0 for k in groups}
    while len(out) < limit:
        progressed = False
        for key, group in groups.items():
            i = cursors[key]
            if i < len(group):
                out.append(group[i])
                cursors[key] = i + 1
                progressed = True
                if len(out) >= limit:
                    break
        if not progressed:
            break
    return out


# --------------------------------------------------------------------------- #
# Splitting
# --------------------------------------------------------------------------- #

def assign_splits(items: list[Mapped], split_cfg: dict) -> None:
    """Assign train/val/test BY GROUP. This is the single most important step in
    the whole script: a random per-image split on video-derived data puts
    near-duplicate adjacent frames on both sides and inflates mAP by tens of
    points, which you would not discover until the robot was in the water."""
    train_pct = int(round(split_cfg["train"] * 100))
    val_pct = int(round(split_cfg["val"] * 100))
    seed = split_cfg.get("seed", 1337)

    for m in items:
        key = f"{seed}:{m.sample.dataset}:{m.sample.group}"
        bucket = stable_bucket(key, 100)
        if bucket < train_pct:
            m.split = "train"
        elif bucket < train_pct + val_pct:
            m.split = "val"
        else:
            m.split = "test"


# --------------------------------------------------------------------------- #
# Writing
# --------------------------------------------------------------------------- #

def out_name(m: Mapped) -> str:
    digest = hashlib.sha1(str(m.sample.image_path).encode("utf-8")).hexdigest()[:10]
    return f"{m.sample.dataset}_{digest}"


def write_one(m: Mapped, out_root: Path, geom: dict) -> tuple[bool, list[int]]:
    img = cv2.imread(str(m.sample.image_path), cv2.IMREAD_COLOR)
    if img is None:
        return False, []

    boxes = [Box(str(cid), cx, cy, w, h) for cid, cx, cy, w, h in m.labels]

    img, boxes = fit_image(img, boxes, geom["width"], geom["height"],
                           geom.get("fit", "crop"), geom.get("min_box_visible", 0.3))
    boxes = clamp_boxes(boxes)

    # Cropping can evict every box from an annotated image. Writing it as empty
    # would be the same mislabelling we guard against in collect(), so skip it.
    if m.labels and not boxes:
        return False, []

    img_path = out_root / "images" / m.split / f"{m.out_name}.jpg"
    lbl_path = out_root / "labels" / m.split / f"{m.out_name}.txt"

    ok = cv2.imwrite(str(img_path), img,
                     [cv2.IMWRITE_JPEG_QUALITY, int(geom.get("jpeg_quality", 95))])
    if not ok:
        return False, []

    lines = [f"{b.label} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}" for b in boxes]
    lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return True, [int(b.label) for b in boxes]


def repeat_factors(items: list[Mapped], threshold: float = 0.10) -> dict[str, int]:
    """Repeat-factor sampling (LVIS-style). Ultralytics has no sampler hook, so
    oversampling is done by physically repeating lines in the train list."""
    train = [m for m in items if m.split == "train" and m.labels]
    n = len(train) or 1

    img_freq: Counter = Counter()
    for m in train:
        for cid in {cid for cid, *_ in m.labels}:
            img_freq[cid] += 1

    cls_rf = {cid: max(1.0, (threshold / (c / n)) ** 0.5) for cid, c in img_freq.items()}

    out: dict[str, int] = {}
    for m in train:
        rf = max((cls_rf.get(cid, 1.0) for cid, *_ in m.labels), default=1.0)
        out[m.out_name] = max(1, int(round(rf)))
    return out


# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets-config", default=str(ROOT / "configs" / "datasets.yaml"))
    ap.add_argument("--taxonomy", default=str(ROOT / "configs" / "taxonomy.yaml"))
    ap.add_argument("--out", default=str(ROOT / "data" / "unified"))
    ap.add_argument("--dataset", action="append", default=[],
                    help="restrict to these sources (repeatable)")
    ap.add_argument("--dry-run", action="store_true",
                    help="count and report, write nothing")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)
    conf = yaml.safe_load(Path(args.datasets_config).read_text(encoding="utf-8"))
    tax = Taxonomy(yaml.safe_load(Path(args.taxonomy).read_text(encoding="utf-8")))

    raw_root = ROOT / conf["root"]
    out_root = Path(args.out)
    geom = conf["geometry"]

    selected = {
        k: v for k, v in conf["datasets"].items()
        if v.get("enabled", True) and (not args.dataset or k in args.dataset)
    }
    if not selected:
        log.error("no datasets selected")
        return 2

    # ---- collect + reduce -------------------------------------------------- #
    everything: list[Mapped] = []
    contribution: dict[str, int] = {}

    for name, cfg in selected.items():
        if not (raw_root / name).exists():
            log.warning("[%s] not downloaded -- skipping", name)
            continue

        log.info("--- %s ---", name)
        items = collect(name, cfg, raw_root, tax)
        if not items:
            continue

        items = apply_stride(items, int(cfg.get("stride", 1)))
        log.info("[%s] after stride: %d", name, len(items))

        items = dedupe(items, int(cfg.get("dedupe_phash", 0)), args.workers)
        log.info("[%s] after dedupe: %d", name, len(items))

        items = cap(items, cfg.get("cap"))
        log.info("[%s] after cap: %d", name, len(items))

        contribution[name] = len(items)
        everything.extend(items)

    if not everything:
        log.error("nothing collected -- run download_datasets.py --check")
        return 1

    assign_splits(everything, conf["split"])
    for m in everything:
        m.out_name = out_name(m)

    # ---- report ------------------------------------------------------------ #
    total = len(everything)
    split_counts = Counter(m.split for m in everything)
    class_counts: Counter = Counter()
    neg = 0
    for m in everything:
        if not m.labels:
            neg += 1
        for cid, *_ in m.labels:
            class_counts[cid] += 1

    lines: list[str] = []
    lines.append(f"images total         {total}")
    lines.append(f"  train/val/test     {split_counts['train']}/"
                 f"{split_counts['val']}/{split_counts['test']}")
    lines.append(f"  negatives          {neg}  ({neg / max(total, 1):.1%}, "
                 f"target {conf.get('negative_fraction', 0):.0%})")
    lines.append("")
    lines.append("boxes per class")
    for cid in sorted(tax.names):
        share = class_counts[cid] / max(sum(class_counts.values()), 1)
        lines.append(f"  {cid:>2} {tax.names[cid]:<16} {class_counts[cid]:>8}  {share:6.1%}")
    lines.append("")
    lines.append("images per source")
    for name, n in sorted(contribution.items(), key=lambda kv: -kv[1]):
        flag = "  <-- OVER 25% CAP" if n / max(total, 1) > 0.25 else ""
        lines.append(f"  {name:<16} {n:>8}  {n / max(total, 1):6.1%}{flag}")
    if tax.dropped:
        lines.append("")
        lines.append("dropped source labels (top 30)")
        for key, n in tax.dropped.most_common(30):
            lines.append(f"  {key:<44} {n:>8}")

    report = "\n".join(lines)
    print()
    print(report)
    print()

    if args.dry_run:
        log.info("dry run -- nothing written")
        return 0

    # ---- write ------------------------------------------------------------- #
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    written: list[Mapped] = []
    failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(write_one, m, out_root, geom): m for m in everything}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="writing"):
            m = futures[fut]
            ok, _ = fut.result()
            if ok:
                written.append(m)
            else:
                failed += 1

    log.info("wrote %d images (%d failed/skipped)", len(written), failed)

    # ---- configs + manifest ------------------------------------------------ #
    base = {
        "path": str(out_root.resolve()),
        "val": "images/val",
        "test": "images/test",
        "names": tax.names,
    }
    (out_root / "dataset.yaml").write_text(
        yaml.safe_dump({**base, "train": "images/train"}, sort_keys=False),
        encoding="utf-8")

    rf = repeat_factors(written)
    rfs_lines: list[str] = []
    for m in written:
        if m.split != "train":
            continue
        rfs_lines.extend([f"./images/train/{m.out_name}.jpg"] * rf.get(m.out_name, 1))
    (out_root / "train_rfs.txt").write_text("\n".join(rfs_lines) + "\n", encoding="utf-8")
    (out_root / "dataset_rfs.yaml").write_text(
        yaml.safe_dump({**base, "train": "train_rfs.txt"}, sort_keys=False),
        encoding="utf-8")
    log.info("repeat-factor train list: %d entries (from %d unique images)",
             len(rfs_lines), sum(1 for m in written if m.split == "train"))

    with open(out_root / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["out_name", "split", "dataset", "group", "n_boxes", "source_path"])
        for m in written:
            w.writerow([m.out_name, m.split, m.sample.dataset, m.sample.group,
                        len(m.labels), str(m.sample.image_path)])

    (out_root / "report.txt").write_text(report + "\n", encoding="utf-8")
    log.info("done -> %s", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

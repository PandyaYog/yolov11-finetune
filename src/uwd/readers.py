"""Format readers: native annotation formats -> canonical Sample records.

Everything is normalised to a single intermediate representation so the twelve
source datasets collapse into five readers rather than twelve adapters:

    coco           TrashCan, DUO, FloW, MODS, FathomNet (via download export)
    voc            SeaShips, VDD-C
    yolo           Brackish, Trash-ICRA19, SMD
    mask_rgb       SUIM              (RGB-coded semantic masks -> boxes)
    mask_negative  MaSTr1325         (object-free frame mining)

Boxes are stored NORMALISED (cx, cy, w, h in 0..1) with the SOURCE label
string. Taxonomy mapping happens later, in scripts/03_unify_datasets.py, so
that dropped labels can be counted and reported rather than silently
vanishing here.
"""

from __future__ import annotations

import json
import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from .util import IMAGE_EXTS

log = logging.getLogger("uwd")


@dataclass
class Box:
    label: str          # source label, pre-mapping
    cx: float           # all normalised 0..1
    cy: float
    w: float
    h: float


@dataclass
class Sample:
    dataset: str
    image_path: Path
    boxes: list[Box] = field(default_factory=list)
    group: str = ""     # video / dive / site id -- splits are assigned by this
    negative: bool = False


# --------------------------------------------------------------------------- #
# Group id extraction
# --------------------------------------------------------------------------- #

def make_group(stem: str, cfg: dict, meta: dict | None = None) -> str:
    """Derive the split group for one image.

    mode=none   -> the image is its own group (correct for independent stills)
    mode=regex  -> first capture group of the pattern (video / dive id)
    mode=field  -> a field carried in the annotation record (e.g. FathomNet dive)
    """
    mode = (cfg or {}).get("mode", "none")
    if mode == "regex":
        m = re.match(cfg["pattern"], stem)
        return m.group(1) if m else stem
    if mode == "field" and meta:
        return str(meta.get(cfg["field"]) or stem)
    return stem


def _xyxy_to_norm(x1: float, y1: float, x2: float, y2: float,
                  w: float, h: float) -> tuple[float, float, float, float]:
    cx = ((x1 + x2) / 2.0) / w
    cy = ((y1 + y2) / 2.0) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return cx, cy, bw, bh


def _valid(b: Box) -> bool:
    return (
        b.w > 1e-4 and b.h > 1e-4
        and -0.5 < b.cx < 1.5 and -0.5 < b.cy < 1.5
        and b.w <= 1.5 and b.h <= 1.5
    )


def _find_image(images_dir: Path, name: str) -> Path | None:
    """Resolve an annotation's file_name against the image dir, tolerating
    extension mismatches (common in these datasets) and nested paths."""
    direct = images_dir / name
    if direct.exists():
        return direct
    stem = Path(name).stem
    for ext in IMAGE_EXTS:
        cand = images_dir / (stem + ext)
        if cand.exists():
            return cand
    hits = list(images_dir.rglob(stem + ".*"))
    for h in hits:
        if h.suffix.lower() in IMAGE_EXTS:
            return h
    return None


# --------------------------------------------------------------------------- #
# COCO
# --------------------------------------------------------------------------- #

def read_coco(dataset: str, ann_path: Path, images_dir: Path,
              group_cfg: dict, **_) -> Iterator[Sample]:
    if not ann_path.exists():
        log.warning("[%s] missing annotation file: %s", dataset, ann_path)
        return
    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    imgs = {im["id"]: im for im in coco.get("images", [])}

    per_image: dict[int, list[Box]] = {i: [] for i in imgs}
    for ann in coco.get("annotations", []):
        if ann.get("iscrowd"):
            continue
        img = imgs.get(ann["image_id"])
        if img is None:
            continue
        bbox = ann.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x, y, bw, bh = bbox                       # COCO is xywh, top-left origin
        # FathomNet's export leaves width/height null for some records
        # (56/10339 as of this writing) -- float(None) would crash the whole
        # run rather than just dropping that one image's boxes.
        if not img.get("width") or not img.get("height"):
            continue
        W, H = float(img["width"]), float(img["height"])
        if W <= 0 or H <= 0:
            continue
        cx, cy, nw, nh = _xyxy_to_norm(x, y, x + bw, y + bh, W, H)
        box = Box(cats.get(ann["category_id"], str(ann["category_id"])), cx, cy, nw, nh)
        if _valid(box):
            per_image[ann["image_id"]].append(box)

    for img_id, img in imgs.items():
        path = _find_image(images_dir, img["file_name"])
        if path is None:
            continue
        yield Sample(
            dataset=dataset,
            image_path=path,
            boxes=per_image.get(img_id, []),
            group=make_group(path.stem, group_cfg, img),
        )


# --------------------------------------------------------------------------- #
# Pascal VOC
# --------------------------------------------------------------------------- #

def read_voc(dataset: str, ann_dir: Path, images_dir: Path,
             group_cfg: dict, **_) -> Iterator[Sample]:
    if not ann_dir.exists():
        log.warning("[%s] missing annotation dir: %s", dataset, ann_dir)
        return

    for xml_path in sorted(ann_dir.rglob("*.xml")):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError:
            log.debug("[%s] unparseable xml: %s", dataset, xml_path)
            continue

        size = root.find("size")
        if size is None:
            continue
        W = float(size.findtext("width") or 0)
        H = float(size.findtext("height") or 0)
        if W <= 0 or H <= 0:
            continue

        fname = root.findtext("filename")
        path = None
        if fname:
            path = _find_image(images_dir, fname)
        if path is None:
            path = _find_image(images_dir, xml_path.stem + ".jpg")
        if path is None:
            continue

        boxes: list[Box] = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip()
            bnd = obj.find("bndbox")
            if not name or bnd is None:
                continue
            try:
                x1 = float(bnd.findtext("xmin")); y1 = float(bnd.findtext("ymin"))
                x2 = float(bnd.findtext("xmax")); y2 = float(bnd.findtext("ymax"))
            except (TypeError, ValueError):
                continue
            box = Box(name, *_xyxy_to_norm(x1, y1, x2, y2, W, H))
            if _valid(box):
                boxes.append(box)

        yield Sample(dataset, path, boxes, make_group(path.stem, group_cfg))


# --------------------------------------------------------------------------- #
# YOLO (darknet txt)
# --------------------------------------------------------------------------- #

def read_yolo(dataset: str, labels_dir: Path, images_dir: Path,
              group_cfg: dict, names: list[str] | None = None, **_) -> Iterator[Sample]:
    if not images_dir.exists():
        log.warning("[%s] missing image dir: %s", dataset, images_dir)
        return
    names = names or []

    for img_path in sorted(images_dir.rglob("*")):
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue
        label_path = labels_dir / (img_path.stem + ".txt")

        boxes: list[Box] = []
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) < 5:
                    continue
                try:
                    idx = int(float(parts[0]))
                    cx, cy, w, h = (float(v) for v in parts[1:5])
                except ValueError:
                    continue
                label = names[idx] if 0 <= idx < len(names) else str(idx)
                box = Box(label, cx, cy, w, h)
                if _valid(box):
                    boxes.append(box)

        yield Sample(dataset, img_path, boxes, make_group(img_path.stem, group_cfg))


# --------------------------------------------------------------------------- #
# Semantic masks -> boxes (SUIM)
# --------------------------------------------------------------------------- #

def _boxes_from_binary(mask: np.ndarray, label: str,
                       min_area_frac: float, max_area_frac: float) -> list[Box]:
    """Connected components of a binary mask -> one box each."""
    H, W = mask.shape[:2]
    area = float(H * W)
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)

    out: list[Box] = []
    for i in range(1, n):                          # 0 is background
        x, y, w, h, a = stats[i]
        frac = a / area
        # Reject specks, and reject full-frame regions: a semantic class like
        # "seafloor" or "open water" covers most of the frame and its bounding
        # box carries no localisation information at all.
        if frac < min_area_frac or frac > max_area_frac:
            continue
        box = Box(label, *_xyxy_to_norm(x, y, x + w, y + h, W, H))
        if _valid(box):
            out.append(box)
    return out


def read_mask_rgb(dataset: str, mask_dir: Path, images_dir: Path,
                  group_cfg: dict, palette: dict | None = None,
                  min_area_frac: float = 0.0015, max_area_frac: float = 0.6,
                  **_) -> Iterator[Sample]:
    if not mask_dir.exists():
        log.warning("[%s] missing mask dir: %s", dataset, mask_dir)
        return
    if not palette:
        log.error("[%s] mask_rgb reader needs a palette", dataset)
        return

    # "R,G,B" -> label, tolerating the near-pure colours JPEG compression makes.
    lut = {tuple(int(v) for v in k.split(",")): lab for k, lab in palette.items()}

    for mask_path in sorted(mask_dir.rglob("*")):
        if mask_path.suffix.lower() not in IMAGE_EXTS:
            continue
        img_path = _find_image(images_dir, mask_path.name)
        if img_path is None:
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
        if mask is None:
            continue
        mask = mask[:, :, ::-1]                    # BGR -> RGB
        # SUIM masks are 3-bit codes; threshold at mid-grey to undo any
        # compression drift before matching against the palette.
        quant = (mask > 127).astype(np.uint8) * 255

        boxes: list[Box] = []
        for rgb, label in lut.items():
            sel = np.all(quant == np.array(rgb, dtype=np.uint8), axis=-1)
            if not sel.any():
                continue
            boxes.extend(_boxes_from_binary(sel, label, min_area_frac, max_area_frac))

        yield Sample(dataset, img_path, boxes, make_group(img_path.stem, group_cfg))


# --------------------------------------------------------------------------- #
# Negative mining (MaSTr1325)
# --------------------------------------------------------------------------- #

def read_mask_negative(dataset: str, mask_dir: Path, images_dir: Path,
                       group_cfg: dict, obstacle_values: list[int] | None = None,
                       max_obstacle_frac: float = 0.01, **_) -> Iterator[Sample]:
    """Emit only frames that are provably object-free.

    MaSTr1325 cannot give useful boxes -- its "environment" class is one blob
    covering shoreline, boats and buoys alike. But that same labelling proves
    which frames contain NO obstacle at all, which is exactly the negative data
    the unknown-bucket design needs.
    """
    if not mask_dir.exists():
        log.warning("[%s] missing mask dir: %s", dataset, mask_dir)
        return
    obstacle_values = obstacle_values or [0]

    for mask_path in sorted(mask_dir.rglob("*")):
        # MaSTr ships masks and source images side by side in the same flat
        # directory (ann="." images="."). Only "<stem>m.png" is a label mask;
        # without this filter every plain "<stem>.jpg" photo also gets
        # visited as a candidate mask_path and its raw pixel values get read
        # as if they were the {0,1,2,4} obstacle/water/sky/ignore label
        # codes, yielding a bogus second (often wrongly-negative) Sample for
        # the same image.
        if mask_path.suffix.lower() != ".png" or not mask_path.stem.endswith("m"):
            continue
        stem = mask_path.stem[:-1]
        img_path = _find_image(images_dir, stem)
        if img_path is None:
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue

        obstacle = np.isin(mask, obstacle_values)
        if obstacle.mean() > max_obstacle_frac:
            continue

        yield Sample(dataset, img_path, [], make_group(img_path.stem, group_cfg),
                     negative=True)


READERS = {
    "coco": read_coco,
    "voc": read_voc,
    "yolo": read_yolo,
    "mask_rgb": read_mask_rgb,
    "mask_negative": read_mask_negative,
}

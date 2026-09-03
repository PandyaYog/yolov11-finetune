"""Shared logic for splitting a corpus split (val or test) by source domain
(submerged vs surface) and validating a checkpoint against each half
separately, instead of one blended mAP across both.

Why this exists: the corpus mixes two visually different domains (WaterNet-
corrected submerged frames, raw surface frames), and several classes are
domain-exclusive (vessel/buoy_marker only ever appear in surface sources;
fish/benthic_invert/gelatinous/flora/rope_net only in submerged ones). A
single blended mAP can hide "does great underwater, quietly bad on the
surface" (or vice versa) behind one number that looks fine on average.

Used by scripts/05b_validate_by_domain.py (split="val", run during/after
training) and scripts/05c_evaluate_test.py (split="test", the one-time final
held-out check -- never touched by training or per-epoch validation).

Domain per image comes from configs/datasets.yaml (`<dataset>.domain`)
cross-referenced against data/unified/manifest.csv's out_name -> dataset map
-- manifest.csv is only written under data/unified/, not
data/unified_enhanced/, but out_names are identical between the two corpora
(scripts/04_enhance_waternet.py preserves them), so it's the right source
for either.
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

from .util import log


def build_domain_split_lists(manifest_path: Path, domain_of: dict[str, str],
                             corpus_root: Path, split: str) -> dict[str, Path]:
    """-> {"submerged": path/to/<split>_submerged.txt, "surface": ...}, each
    listing that domain's slice of `split` as "./images/<split>/<name>.jpg",
    same convention scripts/03_unify_datasets.py uses for train_rfs.txt."""
    by_domain: dict[str, list[str]] = {"submerged": [], "surface": []}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            domain = domain_of.get(row["dataset"])
            if domain not in by_domain:
                continue
            by_domain[domain].append(f"./images/{split}/{row['out_name']}.jpg")

    out: dict[str, Path] = {}
    for domain, lines in by_domain.items():
        list_path = corpus_root / f"{split}_{domain}.txt"
        list_path.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
        out[domain] = list_path
        log.info("%s: %d %s images -> %s", domain, len(lines), split, list_path)
    return out


def write_domain_yaml(base_yaml: Path, list_path: Path, corpus_root: Path,
                      domain: str, split: str) -> Path:
    base = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
    base.pop("path", None)
    base[split] = list_path.name
    # model.val(split=...) only reads that one key, but check_det_dataset
    # still requires 'train'/'val' to be present for a yaml to parse as
    # valid -- keep whatever the base yaml already had, val()/split=test
    # ignores it either way.
    base.setdefault("train", "images/train")
    base.setdefault("val", "images/val")
    out_path = corpus_root / f"dataset_{split}_{domain}.yaml"
    out_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return out_path


def format_class_map(r, cid: int) -> str:
    """r.box.maps is nc-long, but Ultralytics fills any class with zero
    ground-truth instances in THIS split with the split's overall mean AP as
    a placeholder, not a real per-class score (see DetMetrics) --
    r.ap_class_index is the actual list of classes that got a real one.
    Printing the filler as if it were a score would be worse than not having
    this table at all, so make "no instances" explicit instead."""
    if cid not in r.ap_class_index:
        return "n/a (0 instances)"
    return f"{r.box.maps[cid]:.4f}"


def print_domain_report(split: str, lists: dict[str, Path], results: dict) -> None:
    print()
    for domain, list_path in lists.items():
        n_images = sum(1 for line in list_path.read_text().splitlines() if line.strip())
        print(f"[{domain}] {n_images} {split} images")
    print()
    for domain, r in results.items():
        print(f"[{domain}] mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}  "
             f"precision={r.box.mp:.4f}  recall={r.box.mr:.4f}")

    print()
    print(f"per-class mAP50-95 ({split}, submerged vs surface):")
    names = results["submerged"].names
    for cid in sorted(names):
        sub, surf = results["submerged"], results["surface"]
        print(f"  {cid:>2} {names[cid]:<16} "
             f"submerged={format_class_map(sub, cid):<18} surface={format_class_map(surf, cid)}")

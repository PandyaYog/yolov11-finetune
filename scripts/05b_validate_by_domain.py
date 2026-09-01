#!/usr/bin/env python
"""Validate a trained checkpoint separately on submerged-domain and
surface-domain val images, instead of one blended mAP across both.

Why this exists: the corpus mixes two visually different domains (WaterNet-
corrected submerged frames, raw surface frames), and several classes are
domain-exclusive (vessel/buoy_marker only ever appear in surface sources;
fish/benthic_invert/gelatinous/flora/rope_net only in submerged ones). A
single blended mAP can hide "does great underwater, quietly bad on the
surface" (or vice versa) behind one number that looks fine on average. This
was flagged as worth checking before trusting a training run, not after.

Domain per image comes from configs/datasets.yaml (`<dataset>.domain`)
cross-referenced against data/unified/manifest.csv's out_name -> dataset
map -- manifest.csv is only written under data/unified/, not
data/unified_enhanced/, but out_names are identical between the two corpora
(scripts/04_enhance_waternet.py preserves them), so it's the right source
for either.

Writes two small val-list files and matching dataset yamls next to whichever
corpus --data points at (val_submerged.txt / dataset_val_submerged.yaml, and
the surface equivalents) -- cheap, and left in place so a re-run reuses them
instead of regenerating.

Usage
    python scripts/05b_validate_by_domain.py --weights runs/train/yolo11n_enhanced/weights/best.pt
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uwd.util import log, setup_logging  # noqa: E402


def build_domain_val_lists(manifest_path: Path, domain_of: dict[str, str],
                           corpus_root: Path) -> dict[str, Path]:
    """-> {"submerged": path/to/val_submerged.txt, "surface": ...}, each
    listing that domain's val-split images as "./images/val/<name>.jpg",
    same convention scripts/03_unify_datasets.py uses for train_rfs.txt."""
    by_domain: dict[str, list[str]] = {"submerged": [], "surface": []}
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != "val":
                continue
            domain = domain_of.get(row["dataset"])
            if domain not in by_domain:
                continue
            by_domain[domain].append(f"./images/val/{row['out_name']}.jpg")

    out: dict[str, Path] = {}
    for domain, lines in by_domain.items():
        list_path = corpus_root / f"val_{domain}.txt"
        list_path.write_text("\n".join(sorted(lines)) + "\n", encoding="utf-8")
        out[domain] = list_path
        log.info("%s: %d val images -> %s", domain, len(lines), list_path)
    return out


def write_domain_yaml(base_yaml: Path, val_list: Path, corpus_root: Path, domain: str) -> Path:
    base = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
    base.pop("path", None)
    base["val"] = val_list.name
    base["train"] = base.get("train", "images/train")  # val() ignores this, kept for a valid yaml
    out_path = corpus_root / f"dataset_val_{domain}.yaml"
    out_path.write_text(yaml.safe_dump(base, sort_keys=False), encoding="utf-8")
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="trained checkpoint, e.g. runs/.../weights/best.pt")
    ap.add_argument("--data", default=str(ROOT / "data" / "unified_enhanced" / "dataset.yaml"),
                    help="which corpus to validate against (its own directory is where "
                         "val_<domain>.txt / dataset_val_<domain>.yaml get written)")
    ap.add_argument("--manifest", default=str(ROOT / "data" / "unified" / "manifest.csv"),
                    help="source of the out_name -> dataset -> domain mapping (see docstring "
                         "for why this is always data/unified/'s, not data/unified_enhanced/'s)")
    ap.add_argument("--datasets-config", default=str(ROOT / "configs" / "datasets.yaml"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)

    weights = Path(args.weights)
    data_yaml = Path(args.data)
    manifest_path = Path(args.manifest)
    for p, label in [(weights, "--weights"), (data_yaml, "--data"), (manifest_path, "--manifest")]:
        if not p.exists():
            log.error("%s not found: %s", label, p)
            return 1

    conf = yaml.safe_load(Path(args.datasets_config).read_text(encoding="utf-8"))
    domain_of = {name: cfg.get("domain") for name, cfg in conf["datasets"].items()}

    corpus_root = data_yaml.parent
    val_lists = build_domain_val_lists(manifest_path, domain_of, corpus_root)
    empty = [d for d, p in val_lists.items() if p.stat().st_size == 0]
    if empty:
        log.error("no val images found for domain(s): %s -- check %s has a `domain` for "
                  "every enabled source", ", ".join(empty), args.datasets_config)
        return 1

    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = {}
    for domain, val_list in val_lists.items():
        domain_yaml = write_domain_yaml(data_yaml, val_list, corpus_root, domain)
        log.info("--- validating on %s (%s) ---", domain, domain_yaml)
        results[domain] = model.val(data=str(domain_yaml), split="val",
                                    imgsz=args.imgsz, batch=args.batch, device=args.device,
                                    plots=False, verbose=False)

    print()
    for domain, val_list in val_lists.items():
        n_images = sum(1 for line in val_list.read_text().splitlines() if line.strip())
        print(f"[{domain}] {n_images} val images")
    print()
    for domain, r in results.items():
        print(f"[{domain}] mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}  "
             f"precision={r.box.mp:.4f}  recall={r.box.mr:.4f}")

    print()
    print("per-class mAP50-95 (submerged vs surface):")
    names = results["submerged"].names

    def fmt(r, cid: int) -> str:
        # r.box.maps is nc-long, but Ultralytics fills any class with zero
        # ground-truth instances in THIS split with the split's overall mean
        # AP as a placeholder, not a real per-class score (see DetMetrics) --
        # ap_class_index is the actual list of classes that got a real one.
        # Printing the filler as if it were a score would be worse than not
        # having this table at all, so make "no instances" explicit instead.
        if cid not in r.ap_class_index:
            return "n/a (0 instances)"
        return f"{r.box.maps[cid]:.4f}"

    for cid in sorted(names):
        sub, surf = results["submerged"], results["surface"]
        print(f"  {cid:>2} {names[cid]:<16} submerged={fmt(sub, cid):<18} surface={fmt(surf, cid)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Validate a trained checkpoint on the val split, with a per-class table and
-- when the corpus spans more than one source domain -- a per-domain
breakdown instead of one blended mAP across all of them.

Since taxonomy v2 retired the surface sources the corpus is entirely
submerged, so the breakdown is skipped automatically and this reports a
single table. Re-enable a surface source in configs/datasets.yaml and the
per-domain path comes back on its own; see src/uwd/domain_eval.py.

This targets the val split (the same one Ultralytics already checks every
epoch during training) -- for the one-time final held-out check, use
scripts/05c_evaluate_test.py instead.

When a per-domain run does happen it writes one small val-list file and a
matching dataset yaml per domain next to whichever corpus --data points at
(val_submerged.txt / dataset_val_submerged.yaml, ...) -- cheap, and left in
place so a re-run reuses them instead of regenerating.

Usage
    python scripts/05b_validate_by_domain.py --weights runs/train/yolo11n_enhanced/weights/best.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uwd.domain_eval import (  # noqa: E402
    build_domain_split_lists, check_taxonomy_match, domains_in_split,
    print_domain_report, print_report, write_domain_yaml,
)
from uwd.util import log, setup_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="trained checkpoint, e.g. runs/.../weights/best.pt")
    ap.add_argument("--data", default=str(ROOT / "data" / "unified_enhanced" / "dataset.yaml"),
                    help="which corpus to validate against (its own directory is where "
                         "val_<domain>.txt / dataset_val_<domain>.yaml get written)")
    ap.add_argument("--manifest", default=str(ROOT / "data" / "unified" / "manifest.csv"),
                    help="source of the out_name -> dataset -> domain mapping (see "
                         "src/uwd/domain_eval.py for why this is always data/unified/'s, "
                         "not data/unified_enhanced/'s)")
    ap.add_argument("--datasets-config", default=str(ROOT / "configs" / "datasets.yaml"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="0")
    ap.add_argument("--augment", action="store_true",
                    help="test-time augmentation: run each image at several scales and "
                         "flips and merge the detections. Typically worth 1-2 mAP points "
                         "with no retraining, at ~3x inference cost. Legitimate for this "
                         "model's job -- auto-labelling is offline, so the cost never "
                         "reaches the robot -- but report it as TTA, since it is not what "
                         "a plain deployed forward pass would produce.")
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
    present = domains_in_split(manifest_path, domain_of, "val")
    if not present:
        log.error("no val images have a known domain -- check %s gives every enabled "
                  "source a `domain: submerged|surface`", args.datasets_config)
        return 1

    from ultralytics import YOLO

    model = YOLO(str(weights))
    if not check_taxonomy_match(model, data_yaml):
        return 1

    # See the matching note in scripts/05c_evaluate_test.py: with the surface
    # sources retired in taxonomy v2 the corpus is single-domain, so the split
    # has nothing to separate and the per-domain path would build an empty
    # image list for the missing domain.
    if len(present) == 1:
        only = next(iter(present))
        log.info("corpus is entirely '%s' (%d val images) -- reporting one table, "
                 "no domain split", only, present[only])
        r = model.val(data=str(data_yaml), split="val",
                      imgsz=args.imgsz, batch=args.batch, device=args.device,
                      augment=args.augment, plots=False, verbose=False)
        print_report("val", r, present[only], label=only)
        return 0

    corpus_root = data_yaml.parent
    val_lists = build_domain_split_lists(manifest_path, domain_of, corpus_root, "val")

    results = {}
    for domain, val_list in val_lists.items():
        domain_yaml = write_domain_yaml(data_yaml, val_list, corpus_root, domain, "val")
        log.info("--- validating on %s (%s) ---", domain, domain_yaml)
        results[domain] = model.val(data=str(domain_yaml), split="val",
                                    imgsz=args.imgsz, batch=args.batch, device=args.device,
                                    augment=args.augment, plots=False, verbose=False)

    print_domain_report("val", val_lists, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

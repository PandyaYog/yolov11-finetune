#!/usr/bin/env python
"""Validate a trained checkpoint separately on submerged-domain and
surface-domain val images, instead of one blended mAP across both.

See src/uwd/domain_eval.py for why this matters and how domain is
determined. This script targets the val split (the same one Ultralytics
already checks every epoch during training) -- for the one-time final
held-out check, use scripts/05c_evaluate_test.py instead.

Writes two small val-list files and matching dataset yamls next to whichever
corpus --data points at (val_submerged.txt / dataset_val_submerged.yaml, and
the surface equivalents) -- cheap, and left in place so a re-run reuses them
instead of regenerating.

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

from uwd.domain_eval import build_domain_split_lists, print_domain_report, write_domain_yaml  # noqa: E402
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
    val_lists = build_domain_split_lists(manifest_path, domain_of, corpus_root, "val")
    empty = [d for d, p in val_lists.items() if p.stat().st_size == 0]
    if empty:
        log.error("no val images found for domain(s): %s -- check %s has a `domain` for "
                  "every enabled source", ", ".join(empty), args.datasets_config)
        return 1

    from ultralytics import YOLO

    model = YOLO(str(weights))
    results = {}
    for domain, val_list in val_lists.items():
        domain_yaml = write_domain_yaml(data_yaml, val_list, corpus_root, domain, "val")
        log.info("--- validating on %s (%s) ---", domain, domain_yaml)
        results[domain] = model.val(data=str(domain_yaml), split="val",
                                    imgsz=args.imgsz, batch=args.batch, device=args.device,
                                    plots=False, verbose=False)

    print_domain_report("val", val_lists, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

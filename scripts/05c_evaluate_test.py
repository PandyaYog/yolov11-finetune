#!/usr/bin/env python
"""Final held-out check: evaluate a trained checkpoint against the test
split -- the one corpus slice neither training nor the per-epoch validation
checks ever touch.

Reports headline metrics plus a per-class table, which is the part that
matters: a single mAP says nothing about WHICH class is failing, and that is
exactly how flora sat at 0.019 and structure at 0.078 behind a respectable-
looking 0.53 average.

If the corpus spans more than one domain it additionally breaks the numbers
down per domain. Since taxonomy v2 retired the surface sources the corpus is
entirely submerged, so that breakdown is skipped automatically -- there is
nothing to separate. Re-enable any surface source in configs/datasets.yaml
and the per-domain reporting comes back on its own; see src/uwd/domain_eval.py.

Meant to run once, after you're done iterating on a training run, not as a
repeated tuning signal (that's what val -- scripts/05b_validate_by_domain.py
-- is for; using test to guide hyperparameter choices defeats the point of
holding it out).

NOTE: a checkpoint is only comparable to a corpus built with the SAME
taxonomy. v1 checkpoints (11 classes) will silently mis-map against a v2
corpus (8 classes), because labels are class INDICES.

Usage
    # matches the workflow described when this was built: train on Kaggle,
    # download runs/train/<name>/ locally into this repo's runs/train/,
    # then evaluate here against the local data/unified_enhanced/ corpus
    python scripts/05c_evaluate_test.py --weights runs/train/yolo11n_enhanced/weights/best.pt
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
                    help="which corpus to evaluate against (its own directory is where "
                         "test_<domain>.txt / dataset_test_<domain>.yaml get written)")
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
    ap.add_argument("--no-domain-split", action="store_true",
                    help="skip the submerged/surface breakdown, just report one blended "
                         "test-set number (e.g. for a like-for-like ablation comparison)")
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
    present = domains_in_split(manifest_path, domain_of, "test")
    if not present:
        log.error("no test images have a known domain -- check %s gives every enabled "
                  "source a `domain: submerged|surface`", args.datasets_config)
        return 1

    from ultralytics import YOLO

    model = YOLO(str(weights))
    if not check_taxonomy_match(model, data_yaml):
        return 1

    # A breakdown across one domain is just the blended number with extra
    # steps, and building it anyway writes an empty image list for the absent
    # domain that Ultralytics then fails on without naming the cause. Since
    # taxonomy v2 retired the surface sources this is the normal case, not an
    # edge case -- so take the plain path and say why.
    if args.no_domain_split or len(present) == 1:
        only = next(iter(present))
        if len(present) == 1 and not args.no_domain_split:
            log.info("corpus is entirely '%s' (%d test images) -- reporting one table, "
                     "no domain split", only, present[only])
        r = model.val(data=str(data_yaml), split="test",
                      imgsz=args.imgsz, batch=args.batch, device=args.device,
                      augment=args.augment, plots=False, verbose=False)
        print_report("test", r, sum(present.values()),
                     label=only if len(present) == 1 else "all")
        return 0

    corpus_root = data_yaml.parent
    test_lists = build_domain_split_lists(manifest_path, domain_of, corpus_root, "test")

    results = {}
    for domain, test_list in test_lists.items():
        domain_yaml = write_domain_yaml(data_yaml, test_list, corpus_root, domain, "test")
        log.info("--- evaluating on %s (%s) ---", domain, domain_yaml)
        results[domain] = model.val(data=str(domain_yaml), split="test",
                                    imgsz=args.imgsz, batch=args.batch, device=args.device,
                                    augment=args.augment, plots=False, verbose=False)

    print_domain_report("test", test_lists, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Fine-tune YOLOv11n on the unified corpus.

Trains against the WaterNet-enhanced corpus by default
(data/unified_enhanced/), not the raw one -- that's the one whose pixels
match what the robot's inference pipeline actually produces (WaterNet output
when submerged, raw ISP crop on the surface; see scripts/04_enhance_waternet.py).
The plain data/unified/ corpus is kept only for a raw-vs-enhanced ablation,
via --data.

Meant to run in two places:
  - here, for a quick --fraction smoke test that the data/labels/model line up
    before spending real compute on it;
  - on Kaggle with 2x T4, for the real run (--device 0,1). The repo (this
    script) and the corpus (data/unified_enhanced/) reach Kaggle separately --
    code via git clone, data via an attached Kaggle Dataset mounted under
    /kaggle/input/. --data's default won't exist there, so this falls back to
    searching /kaggle/input/*/ for a same-named yaml automatically; see the
    README's "Kaggle" section for the upload + notebook steps.

Usage
    # local smoke test: 2% of train, 1 epoch, confirms the whole loop runs
    python scripts/05_train.py --fraction 0.02 --epochs 1 --batch 8

    # real run, single GPU
    python scripts/05_train.py --epochs 100

    # real run, Kaggle 2x T4
    python scripts/05_train.py --epochs 100 --device 0,1 --batch 64

    # resume an interrupted run (Kaggle sessions have a wall-clock limit)
    python scripts/05_train.py --resume --name <same --name as the run you're resuming>

    # like-for-like ablation against the raw (non-WaterNet) corpus
    python scripts/05_train.py --data data/unified/dataset_rfs.yaml --name ablation_raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uwd.util import log, setup_logging  # noqa: E402


def resolve_data_path(given: Path) -> Path:
    """The repo (this script) and the dataset (data/unified_enhanced/) travel
    to Kaggle separately -- code via git clone, data via an attached Kaggle
    Dataset mounted under /kaggle/input/<dataset-slug>/ -- so the in-repo
    default path won't exist there. If `given` isn't found, look for a
    same-named yaml under /kaggle/input/*/ before giving up, so the exact
    same command works unmodified on both machines."""
    if given.exists():
        return given
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        # rglob, not one-level glob: Kaggle nests an extra folder when you
        # drag-and-drop a whole directory (/kaggle/input/<slug>/unified_enhanced/...)
        # but not when you upload a folder's contents directly
        # (/kaggle/input/<slug>/...) -- don't assume which one was done.
        matches = sorted(kaggle_input.rglob(given.name))
        if len(matches) == 1:
            log.info("%s not found locally -- using Kaggle input dataset at %s", given, matches[0])
            return matches[0]
        if len(matches) > 1:
            log.error("%s not found locally, and multiple Kaggle datasets have a %s -- "
                      "pass --data explicitly:\n  %s",
                      given, given.name, "\n  ".join(str(m) for m in matches))
    return given


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(ROOT / "data" / "unified_enhanced" / "dataset_rfs.yaml"),
                    help="dataset yaml (default: enhanced corpus, RFS-oversampled train list)")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="starting weights: a yolo11n.pt to fine-tune from COCO, or a "
                         "runs/.../last.pt to continue a specific checkpoint")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="matches configs/datasets.yaml geometry -- letterboxed to square "
                         "by Ultralytics' default (non-rect) dataloader")
    ap.add_argument("--batch", default=16,
                    help="int, or -1 for Ultralytics' auto-batch (fits GPU memory at "
                         "~60%% utilisation -- a good default on Kaggle's T4s)")
    ap.add_argument("--device", default="0",
                    help="'0' one GPU, '0,1' both Kaggle T4s (DDP), 'cpu'")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=50, help="early-stop patience, epochs")
    ap.add_argument("--fraction", type=float, default=1.0,
                    help="use only this fraction of the train set -- for a fast local "
                         "smoke test, not for the real run")
    ap.add_argument("--cache", default=False, help="'ram', 'disk', or False")
    ap.add_argument("--project", default=str(ROOT / "runs" / "train"))
    ap.add_argument("--name", default="yolo11n_enhanced")
    ap.add_argument("--resume", action="store_true",
                    help="continue the run at --project/--name from its last checkpoint")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)

    data_path = resolve_data_path(Path(args.data))
    if not data_path.exists():
        log.error("no dataset yaml at %s and nothing matching under /kaggle/input/ either -- "
                  "locally: run scripts/03_unify_datasets.py and scripts/04_enhance_waternet.py "
                  "first; on Kaggle: check the dataset is attached, or pass --data explicitly",
                  data_path)
        return 1

    import torch
    from ultralytics import YOLO

    log.info("torch %s | cuda available: %s | devices: %d",
             torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())
    if args.device != "cpu" and not torch.cuda.is_available():
        log.warning("--device=%s requested but no CUDA device is visible -- falling back to CPU, "
                    "training will be very slow", args.device)
    log.info("data=%s model=%s epochs=%d imgsz=%d fraction=%s",
             data_path, args.model, args.epochs, args.imgsz, args.fraction)

    model = YOLO(args.model)
    batch = int(args.batch) if str(args.batch).lstrip("-").isdigit() else args.batch
    device = "cpu" if args.device == "cpu" else args.device

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        fraction=args.fraction,
        cache=args.cache,
        project=args.project,
        name=args.name,
        resume=args.resume,
        exist_ok=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

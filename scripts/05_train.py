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


def install_blur_haze_augmentations(trainer, p: float) -> None:
    """Point extra augmentation at post-WaterNet residual blur/haze, not
    synthetic underwater colour-cast: WaterNet already corrects colour cast
    before these images ever reach YOLO (see scripts/04_enhance_waternet.py),
    so training the detector to tolerate a raw color cast it will never
    actually see live wastes a nano model's limited capacity. What WaterNet
    doesn't fully correct is scattering-softness/turbidity, which is what
    this targets instead.

    Ultralytics' own default Albumentations set exists but is nearly inert
    (Blur/MedianBlur/CLAHE all at p=0.01 -- see ultralytics/data/augment.py),
    and isn't reachable through any model.train() kwarg. The one documented
    way in is `hyp.augmentations` (ultralytics/data/augment.py:v8_transforms
    docstring), which has to be set before the trainer builds its dataset --
    hence this runs on the on_pretrain_routine_end callback, which fires
    before _setup_train() builds the dataloader.
    """
    import albumentations as A

    trainer.args.augmentations = [
        A.OneOf([
            A.Blur(blur_limit=7, p=1.0),
            A.MedianBlur(blur_limit=7, p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
        ], p=p),
        A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.1, p=p * 0.5),
        A.CLAHE(p=0.01),
        A.RandomGamma(p=0.0),
        A.ImageCompression(quality_range=(75, 100), p=0.0),
    ]
    log.info("albumentations: blur/haze augmentation installed (p=%.2f)", p)


def warn_if_ram_cache_wont_fit(data_path: Path, device: str, imgsz: int) -> None:
    """`--cache ram` holds DECODED images, and under DDP every rank keeps its
    own full copy. That is the trap: the number Ultralytics prints while
    caching ("Caching images (12.8GB RAM)") is PER RANK, so on two GPUs the
    real footprint is double it, and nothing warns you.

    A 640x480 image is 0.92 MB decoded against ~100 KB as a JPEG -- roughly
    9x -- so a corpus that looks like 1.7 GB on disk needs ~15 GB per rank.
    On Kaggle's 30 GB that fits on one GPU and dies on two, and the way it
    dies is unhelpful: the kernel OOM-killer takes a dataloader worker, and
    torchrun reports `DataLoader worker (pid N) is killed by signal: Killed`
    / `exitcode: -9`, which reads like a torch or DDP bug rather than "you ran
    out of host RAM". Confirmed on a 2xT4 run of this corpus.

    Counting label files is enough for the estimate -- there is exactly one
    per image -- and avoids parsing the train list, which may be an RFS file
    whose repeated entries do NOT each cost a separate cache slot.
    """
    corpus = data_path.parent
    n = sum(len(list((corpus / "labels" / s).glob("*.txt")))
            for s in ("train", "val") if (corpus / "labels" / s).is_dir())
    if not n:
        return
    ranks = max(1, len([d for d in str(device).split(",") if d.strip() != ""])) \
        if device != "cpu" else 1

    # Measure a real image rather than assuming imgsz x imgsz: Ultralytics
    # caches the image resized so its LONG side is imgsz, keeping the aspect
    # ratio, and letterboxing happens later per batch. This corpus is 4:3
    # (640x480), so guessing a square would understate the footprint by a
    # third -- which is the difference between "fits" and the OOM killer.
    bytes_per = imgsz * imgsz * 3
    sample = next((corpus / "images" / "train").glob("*.jpg"), None)
    if sample is not None:
        try:
            import cv2
            im = cv2.imread(str(sample))
            if im is not None:
                h, w = im.shape[:2]
                scale = imgsz / max(h, w)
                bytes_per = int(h * scale) * int(w * scale) * 3
        except Exception:  # noqa: BLE001 -- fall back to the square estimate
            pass
    per_rank_gb = n * bytes_per / 1e9
    total_gb = per_rank_gb * ranks

    available_gb = None
    try:
        import os
        available_gb = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 1e9
    except (ValueError, OSError, AttributeError):
        pass

    log.warning("--cache ram: ~%.1f GB per rank x %d rank(s) = ~%.1f GB of host RAM"
                "%s", per_rank_gb, ranks, total_gb,
                f" (about {available_gb:.1f} GB free)" if available_gb else "")
    if available_gb and total_gb > available_gb * 0.85:
        log.error("that will not fit -- the OOM killer will take a dataloader worker "
                  "mid-epoch and torchrun will report it as 'killed by signal: Killed "
                  "/ exitcode: -9', which looks like a DDP bug but is not.\n"
                  "    Drop --cache (the default, and what the 100-epoch 2xT4 run that "
                  "produced runs/train/yolo11n_enhanced actually used), or copy the "
                  "corpus to local disk and train from there.")
        raise SystemExit(2)


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
                    help="starting weights: a yolo11n.pt to fine-tune from COCO, a "
                         "runs/.../last.pt to continue a specific checkpoint, or a model "
                         "DEFINITION yaml (e.g. configs/yolo11-p2.yaml) -- in which case "
                         "--pretrained supplies the weights to warm-start from")
    ap.add_argument("--pretrained", default="yolo11n.pt",
                    help="only used when --model is a .yaml: checkpoint to copy matching "
                         "tensors from, so a custom architecture still inherits COCO "
                         "pretraining instead of starting from random init. '' to disable")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr0", type=float, default=0.001)
    ap.add_argument("--imgsz", type=int, default=640,
                    help="matches configs/datasets.yaml geometry -- letterboxed to square "
                         "by Ultralytics' default (non-rect) dataloader")
    ap.add_argument("--batch", default=16,
                    help="int, or -1 for Ultralytics' auto-batch (fits GPU memory at "
                         "~60%% utilisation -- a good default on Kaggle's T4s)")
    ap.add_argument("--device", default="0",
                    help="'0' one GPU, '0,1' both Kaggle T4s (DDP), 'cpu'")
    ap.add_argument("--optimizer", default="AdamW",
                    help="Never leave this on Ultralytics' 'auto': for >10000 total "
                         "iterations (any real run -- e.g. 100 epochs here is ~39000) "
                         "it picks 'MuSGD', whose ultralytics/optim/muon.py has a real bug "
                         "under DDP (.view() on a non-contiguous tensor -> RuntimeError, "
                         "crashes both ranks on the first optimizer step). A short "
                         "--fraction/--epochs smoke test stays under that threshold and "
                         "won't catch it -- confirmed via a 100-epoch Kaggle 2xT4 run.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--patience", type=int, default=50, help="early-stop patience, epochs")
    ap.add_argument("--flipud", type=float, default=0.1,
                    help="vertical-flip probability. Low, not 0: benefits submerged classes "
                         "(no fixed 'up' underwater) without corrupting the surface fraction "
                         "(vessels/buoys ARE gravity-constrained) too often")
    ap.add_argument("--degrees", type=float, default=10.0,
                    help="rotation augmentation range, degrees. Modest, not aggressive -- "
                         "same surface-domain reasoning as --flipud")
    # --- scale-preserving augmentation ------------------------------------- #
    # These four default to Ultralytics' own values, i.e. they change nothing
    # unless you pass them. That is deliberate: the v1 baseline
    # (runs/train/yolo11n_enhanced) trained with these defaults, so leaving
    # them alone keeps a v2 run comparable to it and isolates what the DATA
    # repair bought. Tune them in a SECOND run, once you have that number.
    #
    # Why they matter here specifically -- measured over this corpus's 65k
    # boxes, by shorter side in pixels at 640x480:
    #
    #                              median   <16px (2 P3 cells)   <8px (1 cell)
    #   native                     44.9px          9.2%              0.3%
    #   mosaic (0.5x linear)       22.4px         34.6%              9.2%
    #   mosaic x scale-min (0.25x) 11.2px         63.8%             34.6%
    #
    # Mosaic tiles 4 images into one, halving every object; `scale` can shrink
    # another 0.5x on top. YOLO11's finest head is P3 at stride 8, so an object
    # under ~8px occupies a single grid cell and is effectively unlearnable.
    # benthic_invert and reef -- 60% of all boxes -- have median short sides of
    # 33px and 32px NATIVELY, so they are exactly the population this pushes
    # under the floor. Mosaic is a good default for COCO-like data; this corpus
    # is not COCO-like.
    ap.add_argument("--mosaic", type=float, default=1.0,
                    help="4-image mosaic probability. Ultralytics default 1.0 (every "
                         "image). Try 0.5 on this corpus -- see the note above")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="random resize gain, +/- this fraction. Default 0.5 means an "
                         "object can be shrunk to half size on top of mosaic. Try 0.3")
    ap.add_argument("--close-mosaic", type=int, default=10,
                    help="disable mosaic for the final N epochs, so training ends on "
                         "native-scale images. Try 20-30 with mosaic on")
    ap.add_argument("--cos-lr", action="store_true",
                    help="cosine LR schedule instead of linear decay; usually worth a "
                         "few tenths of a point and costs nothing")
    ap.add_argument("--blur-p", type=float, default=0.15,
                    help="probability of the custom blur/haze augmentation (see "
                         "install_blur_haze_augmentations); 0 to disable and fall back to "
                         "Ultralytics' own near-inert (p=0.01) defaults")
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
    if str(args.model).endswith(".yaml") and args.pretrained:
        # A .yaml is a model DEFINITION -- YOLO(yaml) starts from random init,
        # throwing away the COCO pretraining that makes fine-tuning on 18k
        # images work at all. .load() copies every tensor whose shape matches
        # and leaves the rest random, which is exactly right for a variant like
        # configs/yolo11-p2.yaml: the backbone and the P3/P4/P5 head transfer,
        # only the new P2 layers start cold.
        log.info("loading pretrained weights from %s into %s", args.pretrained, args.model)
        model = model.load(args.pretrained)
    batch = int(args.batch) if str(args.batch).lstrip("-").isdigit() else args.batch
    device = "cpu" if args.device == "cpu" else args.device

    if str(args.cache).lower() == "ram":
        warn_if_ram_cache_wont_fit(data_path, device, args.imgsz)

    if args.blur_p > 0:
        try:
            import albumentations  # noqa: F401
        except ImportError:
            log.error("--blur-p=%.2f needs the albumentations package -- "
                      "pip install -r requirements-train.txt", args.blur_p)
            return 1
        model.add_callback("on_pretrain_routine_end",
                           lambda trainer: install_blur_haze_augmentations(trainer, args.blur_p))

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        optimizer=args.optimizer,
        lr0=args.lr0,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        workers=args.workers,
        patience=args.patience,
        fraction=args.fraction,
        cache=args.cache,
        flipud=args.flipud,
        degrees=args.degrees,
        mosaic=args.mosaic,
        scale=args.scale,
        close_mosaic=args.close_mosaic,
        cos_lr=args.cos_lr,
        project=args.project,
        name=args.name,
        resume=args.resume,
        exist_ok=args.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Run WaterNet over the unified corpus's submerged-domain images only.

The robot only runs WaterNet when the depth sensor says it's underwater --
on the surface, WaterNet would be "correcting" a colour cast that isn't
there, since the algorithm targets underwater light attenuation/backscatter
specifically. Training has to see the same split, or the model is fine-tuned
on a distribution it will never actually meet at inference:

    domain=submerged sources -> WaterNet output
    domain=surface    sources -> raw ISP crop, untouched

Domain is read per-source from configs/datasets.yaml (`<dataset>.domain`);
which output image belongs to which source comes from data/unified/manifest.csv,
so this must run after scripts/03_unify_datasets.py.

Output is a SEPARATE corpus root, data/unified_enhanced/, laid out exactly
like data/unified/ (images/{train,val,test}, labels/{train,val,test}, plain
+ RFS dataset yaml) rather than a parallel images_enhanced/ tree inside
data/unified/. This isn't a style choice: Ultralytics finds an image's label
file by textually replacing "/images/" with "/labels/" in its path
(ultralytics/data/utils.py:img2label_paths) -- a sibling "images_enhanced/"
directory doesn't contain that substring, so training against it would
silently resolve to zero labels for every image. labels/ here is a byte-for-
byte copy of data/unified/labels/ (enhancement doesn't move boxes), copied
rather than symlinked so the corpus survives being zipped for Kaggle.

Usage
    python scripts/04_enhance_waternet.py --dry-run
    python scripts/04_enhance_waternet.py
    python scripts/04_enhance_waternet.py --checkpoint models/waternet_checkpoint/coarse_112
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from pathlib import Path

import cv2
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uwd.util import log, setup_logging  # noqa: E402


def already_done(out_path: Path) -> bool:
    return out_path.exists()


def atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def atomic_imwrite(dst: Path, img, quality: int) -> bool:
    # cv2.imwrite picks its codec from the extension, so the temp name has to
    # keep ".jpg" -- ".jpg.tmp" made it unrecognisable and imwrite failed.
    tmp = dst.with_name(dst.stem + ".tmp" + dst.suffix)
    ok = cv2.imwrite(str(tmp), img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        tmp.unlink(missing_ok=True)
        return False
    os.replace(tmp, dst)
    return True


def finalize_outputs(unified: Path, out_root: Path) -> None:
    """Copy labels/ and generate dataset.yaml/dataset_rfs.yaml for the
    enhanced corpus. Only called for a full (non---limit) invocation, so a
    smoke test never marks a partial corpus as training-ready."""
    for split in ("train", "val", "test"):
        dst = out_root / "labels" / split
        if not dst.exists():
            shutil.copytree(unified / "labels" / split, dst)

    base_yaml = unified / "dataset.yaml"
    if not base_yaml.exists():
        log.warning("no %s -- skipping dataset.yaml generation for the enhanced corpus", base_yaml)
        return

    base = yaml.safe_load(base_yaml.read_text(encoding="utf-8"))
    base.pop("path", None)  # let Ultralytics default it to the yaml's own
    # directory -- portable across machines and Kaggle dataset mounts; see
    # the matching comment in scripts/03_unify_datasets.py.
    (out_root / "dataset.yaml").write_text(
        yaml.safe_dump({**base, "train": "images/train"}, sort_keys=False), encoding="utf-8")
    (out_root / "dataset_rfs.yaml").write_text(
        yaml.safe_dump({**base, "train": "train_rfs.txt"}, sort_keys=False), encoding="utf-8")

    rfs_src = unified / "train_rfs.txt"
    if rfs_src.exists():
        shutil.copyfile(rfs_src, out_root / "train_rfs.txt")
    log.info("wrote dataset.yaml, dataset_rfs.yaml, labels/, train_rfs.txt -> %s", out_root)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets-config", default=str(ROOT / "configs" / "datasets.yaml"))
    ap.add_argument("--unified", default=str(ROOT / "data" / "unified"))
    ap.add_argument("--out", default=str(ROOT / "data" / "unified_enhanced"),
                    help="separate corpus root -- see module docstring for why this can't "
                         "be a subdirectory of --unified")
    ap.add_argument("--checkpoint", default=str(ROOT / "models" / "waternet_checkpoint" / "coarse_112"))
    ap.add_argument("--dry-run", action="store_true", help="count work, load nothing, write nothing")
    ap.add_argument("--limit", type=int, default=None,
                    help="process at most N images (debugging / smoke-testing a checkpoint)")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)
    conf = yaml.safe_load(Path(args.datasets_config).read_text(encoding="utf-8"))
    domain_of = {name: cfg.get("domain") for name, cfg in conf["datasets"].items()}
    geom = conf["geometry"]

    unified = Path(args.unified)
    out_root = Path(args.out)
    manifest_path = unified / "manifest.csv"
    if not manifest_path.exists():
        log.error("no manifest at %s -- run scripts/03_unify_datasets.py first", manifest_path)
        return 1

    with open(manifest_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    missing_domain = sorted({r["dataset"] for r in rows if domain_of.get(r["dataset"]) is None})
    if missing_domain:
        log.error("no domain configured for: %s -- add `domain: submerged|surface` "
                  "to these entries in configs/datasets.yaml", ", ".join(missing_domain))
        return 1

    work = []
    for r in rows:
        img_path = unified / "images" / r["split"] / f"{r['out_name']}.jpg"
        out_path = out_root / "images" / r["split"] / f"{r['out_name']}.jpg"
        work.append((img_path, out_path, domain_of[r["dataset"]]))

    n_submerged = sum(1 for _, _, d in work if d == "submerged")
    n_surface = len(work) - n_submerged
    log.info("submerged (WaterNet): %d   surface (raw copy): %d", n_submerged, n_surface)

    if args.dry_run:
        log.info("dry run -- nothing written")
        return 0

    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)

    todo = [(i, o, d) for i, o, d in work if not already_done(o)]
    log.info("%d already done, %d remaining", len(work) - len(todo), len(todo))
    if args.limit is not None:
        todo = todo[:args.limit]

    if todo:
        # Import here, not at module load: pulls in TensorFlow and flips the
        # process into TF1 graph-mode (uwd.waternet.tf1.disable_v2_behavior()),
        # which a plain --dry-run has no reason to pay for.
        from uwd.waternet import WaterNet  # noqa: E402

        wn = WaterNet(Path(args.checkpoint), height=geom["height"], width=geom["width"])
        n_enhanced = 0
        n_copied = 0
        n_failed = 0
        try:
            for img_path, out_path, domain in tqdm(todo, desc="enhancing"):
                if domain == "surface":
                    if img_path.exists():
                        atomic_copy(img_path, out_path)
                        n_copied += 1
                    else:
                        n_failed += 1
                    continue

                img = cv2.imread(str(img_path))
                if img is None or img.shape[:2] != (geom["height"], geom["width"]):
                    n_failed += 1
                    continue
                out = wn.enhance(img)
                if atomic_imwrite(out_path, out, int(geom.get("jpeg_quality", 95))):
                    n_enhanced += 1
                else:
                    n_failed += 1
        finally:
            wn.close()

        log.info("enhanced=%d copied=%d failed=%d", n_enhanced, n_copied, n_failed)

    if args.limit is None:
        # A --limit run is a smoke test on a partial corpus; only a full
        # invocation gets to declare the enhanced corpus training-ready.
        finalize_outputs(unified, out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

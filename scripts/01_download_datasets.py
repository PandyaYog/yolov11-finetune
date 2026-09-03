#!/usr/bin/env python
"""Fetch every source dataset into data/raw/, resumably.

Progress is checkpointed at two levels, so Ctrl+C at any point is safe and a
re-run picks up where it stopped:

  dataset level   data/raw/_state.json          which stage each source reached
  item level      data/raw/<name>/_progress.jsonl   per-image, for FathomNet

Ctrl+C finishes the item in flight, flushes, and exits cleanly. A second Ctrl+C
exits immediately.

Usage
    python scripts/01_download_datasets.py --check
    python scripts/01_download_datasets.py --all
    python scripts/01_download_datasets.py --dataset trashcan --dataset suim
    python scripts/01_download_datasets.py --inspect duo
    python scripts/01_download_datasets.py --dataset fathomnet --list-concepts
    python scripts/01_download_datasets.py --dataset duo --force
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uwd.state import (  # noqa: E402
    COMPLETE, DOWNLOADING, EXTRACTING, FAILED, MANUAL, PENDING,
    JsonlProgress, State, install_signal_handlers, stop_requested,
)
from uwd.util import (  # noqa: E402
    extract, flatten_single_dir, http_download, human, log, run,
    setup_logging, tree_summary,
)

DIVE_RE = re.compile(r"/([A-Za-z]+/(?:D|T|V)?\d{3,5})/")


# --------------------------------------------------------------------------- #
# Archive discovery
# --------------------------------------------------------------------------- #

def find_archive(expect: str, archives: Path, data_root: Path) -> Path | None:
    """Look for a manually-downloaded archive in the places people actually put
    them, not just the one place the docs asked for."""
    candidates = [
        archives / expect,
        data_root / expect,
        ROOT / expect,
        data_root / "raw" / expect,
        Path.home() / "Downloads" / expect,
    ]
    for c in candidates:
        if c.exists() and c.stat().st_size > 0:
            return c

    stem = Path(expect).stem.lower()
    for c in sorted(data_root.glob("*")):
        if c.is_file() and c.suffix.lower() in (".zip", ".tar", ".gz", ".tgz") \
                and stem in c.stem.lower():
            return c
    return None


# --------------------------------------------------------------------------- #
# Per-method handlers -> (ok, note)
# --------------------------------------------------------------------------- #

def do_http(name: str, src: dict, archives: Path, target: Path) -> bool:
    urls = src.get("urls") or ([src["url"]] if src.get("url") else [])
    for url in urls:
        if stop_requested():
            return False
        dest = archives / (src.get("expect") or Path(url).name)
        if not dest.exists():
            http_download(url, dest)      # resumes from .part automatically
        extract(dest, target)
    return bool(urls)


def do_gdown(name: str, src: dict, archives: Path, target: Path) -> bool:
    dest = archives / src["expect"]
    if not dest.exists():
        gdrive_url = f"https://drive.google.com/uc?id={src['id']}"
        if run([sys.executable, "-m", "gdown", gdrive_url, "-O", str(dest)]) != 0 or not dest.exists():
            log.warning("[%s] gdown failed (quota or private link?)", name)
            return False
    extract(dest, target)
    flatten_single_dir(target)
    return True


def do_kaggle(name: str, src: dict, archives: Path, target: Path) -> bool:
    slug = src["slug"]
    stem = slug.split("/")[-1]
    dest = archives / f"{stem}.zip"

    if not dest.exists():
        code = run([sys.executable, "-m", "kaggle", "datasets", "download", "-d", slug, "-p", str(archives)])
        if code != 0:
            log.warning("[%s] kaggle CLI failed. Set up ~/.kaggle/kaggle.json "
                        "(Kaggle > Account > Create New API Token).", name)
            return False
    if not dest.exists():
        hits = sorted(archives.glob(f"{stem}*.zip"))
        if not hits:
            return False
        dest = hits[0]

    extract(dest, target)
    flatten_single_dir(target)
    return True


def do_roboflow(name: str, src: dict, archives: Path, target: Path) -> bool:
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        log.warning("[%s] ROBOFLOW_API_KEY not set -- skipping roboflow source", name)
        return False
    try:
        from roboflow import Roboflow
    except ImportError:
        log.warning("[%s] pip install roboflow to use this source", name)
        return False
    rf = Roboflow(api_key=key)
    (rf.workspace(src["workspace"]).project(src["project"])
       .version(src["version"]).download(src.get("format", "yolov8"),
                                         location=str(target)))
    return True


def do_manual(name: str, src: dict, archives: Path, target: Path,
              data_root: Path) -> bool:
    found = find_archive(src["expect"], archives, data_root)
    if found:
        log.info("[%s] found %s (%s) at %s", name, found.name,
                 human(found.stat().st_size), found.parent)
        extract(found, target)
        flatten_single_dir(target)
        return True

    print()
    print(f"  MANUAL DOWNLOAD REQUIRED: {name}")
    print(f"  {'-' * 62}")
    for line in (src.get("note") or "").rstrip().splitlines():
        print(f"  {line}")
    print(f"  Save as: {archives / src['expect']}")
    print(f"  (also accepted: {data_root}/, the project root, or ~/Downloads)")
    print()
    return False


# --------------------------------------------------------------------------- #
# FathomNet: round-robin, per-image resumable
# --------------------------------------------------------------------------- #

def do_fathomnet(name: str, cfg: dict, target: Path, state: State,
                 list_only: bool = False) -> bool:
    try:
        from fathomnet.api import boundingboxes, images as fn_images
    except ImportError:
        log.error("[%s] pip install fathomnet", name)
        return False

    concept_map: dict[str, str] = {}          # concept -> unified class
    for cls, concepts in cfg.get("concepts", {}).items():
        for c in concepts:
            concept_map[c] = cls

    if list_only:
        try:
            available = set(boundingboxes.find_concepts())
        except Exception as exc:                          # noqa: BLE001
            log.error("[%s] could not reach the API: %s", name, exc)
            return False
        log.info("[%s] %d concepts available upstream", name, len(available))
        missing = 0
        for concept, cls in concept_map.items():
            ok = concept in available
            missing += not ok
            print(f"  {'OK     ' if ok else 'MISSING'}  {cls:<16} {concept}")
        print(f"\n  {len(concept_map) - missing}/{len(concept_map)} configured "
              f"concepts exist upstream.\n")
        return True

    img_dir = target / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    index_path = target / "_index.json"
    progress = JsonlProgress(target / "_progress.jsonl")

    per_cap = int(cfg.get("per_concept_cap", 400))
    budget = int(cfg.get("target_total", cfg.get("cap") or 10000))

    # ---- phase 1: index (cached, so a resume costs no API calls) ---------- #
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        log.info("[%s] reusing cached concept index (%d concepts)", name, len(index))
    else:
        index = {}
        for concept in concept_map:
            if stop_requested():
                break
            try:
                results = fn_images.find_by_concept(concept)
            except Exception as exc:                      # noqa: BLE001
                log.warning("[%s] concept %r unavailable: %s", name, concept, exc)
                continue
            index[concept] = [
                {"uuid": r.uuid, "url": r.url, "width": r.width, "height": r.height,
                 "boxes": [{"concept": b.concept, "x": b.x, "y": b.y,
                            "w": b.width, "h": b.height}
                           for b in (r.boundingBoxes or [])]}
                for r in results if r.url
            ][:per_cap]
            log.info("[%s] indexed %-30s %5d images", name, concept, len(index[concept]))
        index_path.write_text(json.dumps(index), encoding="utf-8")

    # ---- phase 2: round-robin download ------------------------------------ #
    # Round-robin rather than concept-by-concept so that an interrupted run
    # still yields balanced coverage. Draining concept 1 before starting
    # concept 2 means a stop at 40% gives you 100% of a few morphologies and
    # none of the rest -- exactly the diversity failure we are guarding against.
    done = progress.keys("uuid")
    log.info("[%s] %d images already fetched", name, len(done))

    cursors = {c: 0 for c in index}
    fetched = len(done)
    interrupted = False

    with progress:
        while fetched < budget:
            progressed = False
            for concept, records in index.items():
                if fetched >= budget:
                    break
                if stop_requested():
                    interrupted = True
                    break
                i = cursors[concept]
                if i >= len(records):
                    continue
                cursors[concept] = i + 1
                progressed = True
                rec = records[i]

                if rec["uuid"] in done:
                    continue

                ext = Path(rec["url"]).suffix or ".jpg"
                out = img_dir / f"{rec['uuid']}{ext}"
                if not (out.exists() and out.stat().st_size > 0):
                    try:
                        http_download(rec["url"], out)
                    except Exception as exc:              # noqa: BLE001
                        log.debug("[%s] fetch failed %s: %s", name, rec["uuid"], exc)
                        continue

                dive = DIVE_RE.search(rec["url"])
                progress.append({
                    "uuid": rec["uuid"],
                    "file_name": out.name,
                    "width": rec["width"],
                    "height": rec["height"],
                    "dive": dive.group(1) if dive else rec["uuid"],
                    "concept": concept,
                    "cls": concept_map[concept],
                    "boxes": rec["boxes"],
                })
                done.add(rec["uuid"])
                fetched += 1
                if fetched % 100 == 0:
                    log.info("[%s] %d / %d", name, fetched, budget)

            if interrupted or not progressed:
                break

    state.update(name, images=fetched, budget=budget,
                 status=DOWNLOADING if interrupted else EXTRACTING)

    # ---- phase 3: compile COCO from the progress log ---------------------- #
    compile_fathomnet_coco(name, target, cfg,
                           ROOT / "configs" / "fathomnet_concept_map.csv")
    return not interrupted


def load_concept_map(path: Path) -> dict[str, str]:
    """concept -> unified class, from the reviewed table written by
    scripts/01b_resolve_fathomnet_concepts.py.

    Rows with a blank class (CONFLICT / unresolved) are deliberately excluded:
    an unreviewed ambiguity must not enter the corpus by default."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            cls = (row.get("unified_class") or "").strip()
            if cls:
                out[row["concept"]] = cls
    return out


def compile_fathomnet_coco(name: str, target: Path, cfg: dict,
                           concept_map_path: Path | None = None) -> None:
    """Rebuild annotations.json from the append-only progress log. Safe to run
    at any point -- it always reflects exactly what is on disk.

    Boxes are labelled by resolving THEIR OWN concept through the reviewed
    concept map, not by assuming they share the class of the concept the image
    was queried for. The previous rule -- keep a box only if its concept
    string equals the queried concept -- discarded 66% of everything FathomNet
    returned (23,262 of 44,663 boxes as "out of taxonomy", plus 6,302 that were
    already in our own concept list), because a species name never string-
    matches its parent taxon. See scripts/01b_resolve_fathomnet_concepts.py.

    Images whose annotation completeness falls below
    `min_annotation_completeness` are dropped whole. This is the part that
    actually removes training noise rather than just adding labels: a box we
    cannot resolve is still a real object sitting in the frame, and an
    unlabelled real object teaches the detector to suppress a correct
    detection. An image mostly full of them is worth less than nothing.
    """
    progress = JsonlProgress(target / "_progress.jsonl")
    classes = list(cfg.get("concepts", {}).keys())
    cat_id = {c: i + 1 for i, c in enumerate(classes)}

    cmap = load_concept_map(concept_map_path) if concept_map_path else {}
    if not cmap:
        log.warning("[%s] no reviewed concept map at %s -- falling back to exact "
                    "concept matching, which is known to discard ~66%% of available "
                    "boxes. Run scripts/01b_resolve_fathomnet_concepts.py first.",
                    name, concept_map_path)
    else:
        log.info("[%s] concept map: %d concepts resolve to a class", name, len(cmap))

    min_complete = float(cfg.get("min_annotation_completeness", 0.0))

    coco = {"images": [], "annotations": [],
            "categories": [{"id": i, "name": c} for c, i in cat_id.items()]}
    ann_id = 1
    kept_boxes = dropped_incomplete = unresolved_boxes = 0

    for rec in progress.read_all():
        if not (target / "images" / rec["file_name"]).exists():
            continue

        resolved: list[tuple[str, dict]] = []
        unresolved = 0
        # FathomNet can carry the same localisation twice on one image -- two
        # annotators boxing the same organism, or the same box filed under a
        # concept and its synonym. Under the old exact-match rule at most one
        # of the pair survived, so this never showed; keeping every resolvable
        # concept surfaces them as literally identical rows in the label file.
        # Harmless (Ultralytics drops them and says "N duplicate labels
        # removed") but it makes every run's scan log noisy for no reason, and
        # a duplicated box would double-count in any box statistics we compute
        # ourselves. Dedupe on the resolved class plus the geometry.
        seen: set[tuple] = set()
        for b in rec["boxes"]:
            if cmap:
                cls = cmap.get(b["concept"])
            else:
                cls = rec["cls"] if b["concept"] == rec["concept"] else None
            # A concept can resolve to a class that is not in this config's
            # concept list (e.g. after a class is retired); skip rather than
            # KeyError on cat_id.
            if cls and cls in cat_id:
                key = (cls, b["x"], b["y"], b["w"], b["h"])
                if key in seen:
                    continue
                seen.add(key)
                resolved.append((cls, b))
            else:
                unresolved += 1

        total = len(resolved) + unresolved
        if total and (len(resolved) / total) < min_complete:
            dropped_incomplete += 1
            continue
        unresolved_boxes += unresolved

        img_id = len(coco["images"]) + 1
        coco["images"].append({
            "id": img_id,
            "file_name": rec["file_name"],
            "width": rec["width"],
            "height": rec["height"],
            "dive": rec["dive"],
            "concept": rec["concept"],
        })
        for cls, b in resolved:
            coco["annotations"].append({
                "id": ann_id, "image_id": img_id,
                "category_id": cat_id[cls],
                "bbox": [b["x"], b["y"], b["w"], b["h"]], "iscrowd": 0,
            })
            ann_id += 1
            kept_boxes += 1

    (target / "annotations.json").write_text(json.dumps(coco), encoding="utf-8")
    log.info("[%s] compiled %d images / %d boxes -> annotations.json",
             name, len(coco["images"]), kept_boxes)
    if min_complete:
        log.info("[%s] dropped %d images below %.0f%% annotation completeness; "
                 "%d unresolved boxes remain in the kept images",
                 name, dropped_incomplete, 100 * min_complete, unresolved_boxes)


# --------------------------------------------------------------------------- #

def fetch(name: str, cfg: dict, root: Path, archives: Path, state: State,
          list_concepts: bool = False, force: bool = False) -> bool:
    if force:
        state.reset(name)

    if state.is_complete(name) and not list_concepts:
        log.info("[%s] complete (--force to redo)", name)
        return True

    target = root / name
    target.mkdir(parents=True, exist_ok=True)
    data_root = root.parent

    sources = cfg.get("sources", [])

    # Separate sources into auto-downloadable (alternatives — first success wins)
    # and manual (all must succeed, since a dataset may ship as multiple archives).
    auto_sources = [s for s in sources if s["method"] != "manual"]
    manual_sources = [s for s in sources if s["method"] == "manual"]

    # --- try auto sources first (alternatives) ---
    for src in auto_sources:
        if stop_requested():
            state.update(name, status=state.status(name) or PENDING)
            return False

        method = src["method"]
        state.update(name, status=DOWNLOADING, source=method)
        try:
            if method == "http":
                ok = do_http(name, src, archives, target)
            elif method == "gdown":
                ok = do_gdown(name, src, archives, target)
            elif method == "kaggle":
                ok = do_kaggle(name, src, archives, target)
            elif method == "roboflow":
                ok = do_roboflow(name, src, archives, target)
            elif method == "fathomnet":
                ok = do_fathomnet(name, cfg, target, state, list_concepts)
            else:
                log.error("[%s] unknown method %r", name, method)
                ok = False
        except KeyboardInterrupt:
            raise
        except Exception as exc:                          # noqa: BLE001
            log.warning("[%s] source %s raised: %s", name, method, exc)
            ok = False

        if ok:
            if list_concepts:
                return True
            n = sum(1 for _ in target.rglob("*") if _.is_file())
            state.update(name, status=COMPLETE, files=n, source=method)
            log.info("[%s] complete (%d files)", name, n)
            return True

    # --- try manual sources (ALL must succeed) ---
    if manual_sources:
        all_ok = True
        for src in manual_sources:
            if stop_requested():
                state.update(name, status=state.status(name) or PENDING)
                return False
            state.update(name, status=DOWNLOADING, source="manual")
            try:
                ok = do_manual(name, src, archives, target, data_root)
            except KeyboardInterrupt:
                raise
            except Exception as exc:                      # noqa: BLE001
                log.warning("[%s] manual source %s raised: %s",
                            name, src.get("expect", "?"), exc)
                ok = False
            if not ok:
                all_ok = False
        if all_ok:
            n = sum(1 for _ in target.rglob("*") if _.is_file())
            state.update(name, status=COMPLETE, files=n, source="manual")
            log.info("[%s] complete (%d files)", name, n)
            return True

    status = MANUAL if manual_sources else FAILED
    state.update(name, status=status)
    return False


def check(root: Path, datasets: dict, state: State) -> None:
    print()
    print(f"  {'dataset':<16}{'status':<18}{'files':>8}   title")
    print(f"  {'-' * 80}")
    for name, cfg in datasets.items():
        if not cfg.get("enabled", True):
            print(f"  {name:<16}{'disabled':<18}{'-':>8}   {cfg.get('title', '')}")
            continue
        rec = state.get(name)
        status = rec.get("status", PENDING)
        n = rec.get("files") or sum(
            1 for _ in (root / name).rglob("*") if _.is_file()
        ) if (root / name).exists() else 0
        extra = ""
        if name == "fathomnet" and rec.get("images"):
            extra = f"  [{rec['images']}/{rec.get('budget', '?')} imgs]"
        print(f"  {name:<16}{status:<18}{n:>8}   {cfg.get('title', '')}{extra}")
    print()
    print("  Re-run the same command to resume anything not 'complete'.")
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "datasets.yaml"))
    ap.add_argument("--dataset", action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--inspect", metavar="NAME",
                    help="print the extracted tree so layout paths can be fixed")
    ap.add_argument("--list-concepts", action="store_true",
                    help="FathomNet only: verify configured concepts against the API")
    ap.add_argument("--force", action="store_true", help="re-fetch even if complete")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    setup_logging(args.verbose)
    install_signal_handlers()

    conf = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    root = ROOT / conf["root"]
    archives = ROOT / conf["archives"]
    root.mkdir(parents=True, exist_ok=True)
    archives.mkdir(parents=True, exist_ok=True)

    state = State(root / "_state.json")
    datasets = conf["datasets"]

    if args.inspect:
        print(f"\n  {args.inspect}  ->  {root / args.inspect}\n")
        print(tree_summary(root / args.inspect))
        print()
        return 0

    if args.check:
        check(root, datasets, state)
        return 0

    if args.dataset:
        selected = {k: v for k, v in datasets.items() if k in args.dataset}
        unknown = set(args.dataset) - set(selected)
        if unknown:
            log.error("unknown dataset(s): %s", ", ".join(sorted(unknown)))
            return 2
    elif args.all:
        selected = {k: v for k, v in datasets.items() if v.get("enabled", True)}
    else:
        ap.print_help()
        return 0

    results: dict[str, bool] = {}
    for name, cfg in selected.items():
        if stop_requested():
            log.warning("stopping before %s (state saved)", name)
            break
        log.info("=== %s ===", name)
        results[name] = fetch(name, cfg, root, archives, state,
                              args.list_concepts, args.force)

    print()
    ok = [n for n, r in results.items() if r]
    bad = [n for n, r in results.items() if not r]
    log.info("complete: %s", ", ".join(ok) if ok else "(none)")
    if bad:
        log.warning("incomplete: %s", ", ".join(bad))
        print("\n  Resume with the same command -- finished work is not redone.\n")
    return 130 if stop_requested() else 0


if __name__ == "__main__":
    raise SystemExit(main())

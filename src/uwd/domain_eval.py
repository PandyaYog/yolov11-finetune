"""Shared evaluation reporting for a corpus split (val or test), optionally
broken down by source domain.

Why the domain breakdown exists: when the corpus mixed two visually different
domains (WaterNet-corrected submerged frames, raw surface frames), a single
blended mAP could hide "does great underwater, quietly bad on the surface"
behind one average that looked fine. Several classes were domain-exclusive
too (vessel/buoy_marker were surface-only), so a blended per-class table said
nothing about where a class actually failed.

As of taxonomy v2 the surface sources are disabled in configs/datasets.yaml
and the corpus is entirely submerged, so the split has nothing to separate.
Rather than delete the machinery -- surface capture is still part of the
robot's job and the sources can be re-enabled -- everything here is written
for N domains: with one domain it reports a single straightforward table, with
two or more it reports a column per domain. Nothing needs editing if surface
comes back.

Used by scripts/05b_validate_by_domain.py (split="val") and
scripts/05c_evaluate_test.py (split="test", the one-time final held-out check
that training and per-epoch validation never touch).

Domain per image comes from configs/datasets.yaml (`<dataset>.domain`)
cross-referenced against data/unified/manifest.csv's out_name -> dataset map
-- manifest.csv is only written under data/unified/, not
data/unified_enhanced/, but out_names are identical between the two corpora
(scripts/04_enhance_waternet.py preserves them), so it's the right source
for either.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import yaml

from .util import log


def domains_in_split(manifest_path: Path, domain_of: dict[str, str],
                     split: str) -> dict[str, int]:
    """-> {domain: image count} for the domains actually present in `split`.

    Call this before deciding whether a per-domain breakdown is even
    meaningful: a corpus with one domain does not need one, and building it
    anyway produces an empty image list that Ultralytics fails on in a way
    that does not name the real cause.
    """
    counts: dict[str, int] = defaultdict(int)
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            domain = domain_of.get(row["dataset"])
            if domain:
                counts[domain] += 1
    return dict(counts)


def build_domain_split_lists(manifest_path: Path, domain_of: dict[str, str],
                             corpus_root: Path, split: str) -> dict[str, Path]:
    """-> {domain: path/to/<split>_<domain>.txt}, each listing that domain's
    slice of `split` as "./images/<split>/<name>.jpg", the same convention
    scripts/03_unify_datasets.py uses for train_rfs.txt.

    Only domains that actually have images are returned. The previous version
    always emitted every known domain and left the caller to spot the empty
    ones by file size -- which did not work, because an empty list still
    writes a single newline, so the `st_size == 0` check never fired and an
    empty list reached model.val().
    """
    by_domain: dict[str, list[str]] = defaultdict(list)
    with open(manifest_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["split"] != split:
                continue
            domain = domain_of.get(row["dataset"])
            if domain:
                by_domain[domain].append(f"./images/{split}/{row['out_name']}.jpg")

    out: dict[str, Path] = {}
    for domain, lines in sorted(by_domain.items()):
        if not lines:
            continue
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


def check_taxonomy_match(model, data_yaml: Path) -> bool:
    """Refuse to evaluate a checkpoint whose class list disagrees with the
    corpus's.

    Labels are stored as class INDICES, so a mismatch does not fail loudly --
    it silently scores each prediction against whatever class now occupies
    that index and prints a plausible-looking table of nonsense. Taxonomy v2
    renumbered everything above index 4 (rope_net, vessel and buoy_marker were
    retired, structure became reef), so every v1 checkpoint under
    runs/train/yolo11n_enhanced* is in exactly this position with respect to a
    v2 corpus. Worth one explicit check before the run rather than a wrong
    number in a report.
    """
    want = yaml.safe_load(data_yaml.read_text(encoding="utf-8")).get("names") or {}
    want = {int(k): v for k, v in want.items()}
    got = {int(k): v for k, v in (getattr(model, "names", None) or {}).items()}
    if got == want:
        return True
    log.error(
        "checkpoint / corpus taxonomy mismatch -- refusing to report a number that "
        "would be wrong:\n"
        "    checkpoint (%d classes): %s\n"
        "    corpus     (%d classes): %s\n"
        "Labels are class INDICES, so this scores each prediction against whichever "
        "class now holds that index. Evaluate this checkpoint against the corpus it "
        "was trained on, or retrain on the current one.",
        len(got), got, len(want), want)
    return False


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


def _headline(label: str, r) -> str:
    return (f"[{label}] mAP50={r.box.map50:.4f}  mAP50-95={r.box.map:.4f}  "
            f"precision={r.box.mp:.4f}  recall={r.box.mr:.4f}")


def print_report(split: str, result, n_images: int, label: str = "all") -> None:
    """Single-population report: headline metrics plus the full per-class
    table. Used when the corpus has one domain, and for --no-domain-split.

    The per-class table is the point of this script -- a single mAP tells you
    nothing about WHICH class is failing, which is how flora sat at 0.019 and
    structure at 0.078 behind a respectable-looking 0.53 average.
    """
    print()
    print(f"[{label}] {n_images} {split} images")
    print()
    print(_headline(label, result))
    print()
    print(f"per-class ({split}):")
    print(f"  {'':>2} {'class':<16} {'mAP50-95':>10}")
    for cid in sorted(result.names):
        print(f"  {cid:>2} {result.names[cid]:<16} {format_class_map(result, cid):>10}")


def print_domain_report(split: str, lists: dict[str, Path], results: dict) -> None:
    """Per-domain report, for however many domains are present."""
    domains = list(results)
    print()
    for domain, list_path in lists.items():
        n_images = sum(1 for line in list_path.read_text().splitlines() if line.strip())
        print(f"[{domain}] {n_images} {split} images")
    print()
    for domain in domains:
        print(_headline(domain, results[domain]))

    print()
    print(f"per-class mAP50-95 ({split}, {' vs '.join(domains)}):")
    names = results[domains[0]].names
    for cid in sorted(names):
        cells = "  ".join(f"{d}={format_class_map(results[d], cid):<18}" for d in domains)
        print(f"  {cid:>2} {names[cid]:<16} {cells}")

#!/usr/bin/env python
"""Resolve every FathomNet concept seen in the download log to a unified class,
via taxonomic ancestry rather than exact string equality.

WHY THIS EXISTS
    FathomNet localisations are labelled with whatever concept the annotator
    used -- usually a species or genus, e.g. "Strongylocentrotus fragilis".
    configs/datasets.yaml queries at a much coarser level ("Echinoidea"), and
    the exporter used to keep a box only when its concept string EXACTLY matched
    the queried concept. A species name never string-matches its parent taxon,
    so 23,262 of 44,663 available boxes (52.1%) were discarded as "outside our
    taxonomy" when in fact most of them are descendants of taxa we query for.

    Those rejected objects stayed in the frame with no label, which for a
    detector is worse than useless: an unlabelled real object is a false
    negative baked into the ground truth, and it trains the model to suppress
    correct detections. Measured effect: on the held-out test split the model's
    unmatched-detection rate was 9.3% on exhaustively-annotated DUO imagery but
    27-52% on FathomNet imagery.

HOW
    Rather than looking up ancestors for each of the ~949 unknown concepts (one
    slow API call each), this asks the API for the DESCENDANTS of each concept
    we already query (~60 calls) and inverts the result. Same semantics, an
    order of magnitude fewer requests, and the whole thing caches to disk so a
    re-run costs nothing.

OUTPUT
    configs/fathomnet_concept_map.csv -- one row per concept actually seen in
    the download log, with the class it resolves to and how. This file is meant
    to be READ AND EDITED BY A HUMAN before it is trusted: taxonomic ancestry is
    a good prior, not an oracle. Rows marked CONFLICT descend from two queried
    taxa that disagree about the class (the Anthozoa/Actiniaria case is the
    known one) and MUST be settled by hand -- leaving them contradicts the
    model's supervision on the same organism.

    scripts/01_download_datasets.py reads this file when recompiling
    annotations.json. Concepts left `unresolved` are not labelled, and count
    against their image's annotation-completeness score.

Usage
    python scripts/01b_resolve_fathomnet_concepts.py
    python scripts/01b_resolve_fathomnet_concepts.py --refresh   # re-fetch taxa
"""

from __future__ import annotations

import argparse
import csv
import json
import urllib.parse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uwd.util import log, setup_logging  # noqa: E402

TAXA_URL = "https://database.fathomnet.org/api/taxa/query/mbari/{}"


def fetch_descendants(concept: str, timeout: int) -> list[str] | None:
    """Every taxon at or below `concept` in the MBARI tree. None on failure --
    distinct from [] (a valid leaf taxon with no descendants)."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(TAXA_URL.format(
                urllib.parse.quote(concept)), timeout=timeout) as r:
            data = json.load(r)
    except Exception as exc:  # noqa: BLE001
        log.warning("  taxa lookup failed for %-28s %s", concept, exc)
        return None
    return [d["name"] for d in data if d.get("name")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs" / "datasets.yaml"))
    ap.add_argument("--progress", default=str(ROOT / "data" / "raw" / "fathomnet" / "_progress.jsonl"))
    ap.add_argument("--cache", default=str(ROOT / "data" / "raw" / "fathomnet" / "_taxa_cache.json"),
                    help="descendant lists, so a re-run costs no API calls")
    ap.add_argument("--out", default=str(ROOT / "configs" / "fathomnet_concept_map.csv"))
    ap.add_argument("--timeout", type=int, default=90,
                    help="per-request timeout; the taxa endpoint is slow (~10s typical)")
    ap.add_argument("--refresh", action="store_true", help="ignore the cache and re-fetch")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    setup_logging(args.verbose)

    progress_path = Path(args.progress)
    if not progress_path.exists():
        log.error("no download log at %s -- run scripts/01_download_datasets.py "
                  "--dataset fathomnet first", progress_path)
        return 1

    conf = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    fn = conf["datasets"]["fathomnet"]
    concept_map: dict[str, str] = {}
    for cls, concepts in fn.get("concepts", {}).items():
        for c in concepts:
            concept_map[c] = cls

    # ---- every concept actually present in the download log ---------------- #
    seen: Counter = Counter()
    for line in progress_path.open(encoding="utf-8"):
        rec = json.loads(line)
        for b in rec["boxes"]:
            seen[b["concept"]] += 1
    log.info("%d distinct concepts across %d boxes in the download log",
             len(seen), sum(seen.values()))

    # ---- descendants of each queried taxon (cached) ------------------------ #
    cache_path = Path(args.cache)
    cache: dict[str, list[str]] = {}
    if cache_path.exists() and not args.refresh:
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        log.info("reusing cached taxa for %d concepts", len(cache))

    todo = [c for c in concept_map if c not in cache]
    if todo:
        log.info("fetching descendants for %d queried concepts (~10s each)", len(todo))
        for i, concept in enumerate(todo, 1):
            got = fetch_descendants(concept, args.timeout)
            if got is None:
                continue
            cache[concept] = got
            log.info("  [%2d/%2d] %-30s %4d descendants", i, len(todo), concept, len(got))
            cache_path.write_text(json.dumps(cache), encoding="utf-8")  # checkpoint
            time.sleep(0.2)
    else:
        log.info("all queried concepts already cached")

    # ---- invert: descendant -> the queried taxa it belongs to -------------- #
    owners: dict[str, set[str]] = defaultdict(set)
    for queried, descendants in cache.items():
        for d in descendants:
            owners[d].add(queried)

    # ---- resolve every seen concept ---------------------------------------- #
    rows = []
    tally: Counter = Counter()
    for concept, n in seen.most_common():
        # Exact match always wins: it is what the taxonomy explicitly asked for,
        # and it is how a concept that is BOTH queried directly and a descendant
        # of another queried taxon gets settled (e.g. Actiniaria, which we query
        # as benthic_invert but which also descends from Anthozoa -> structure).
        if concept in concept_map:
            rows.append((concept, n, concept_map[concept], "exact", concept))
            tally["exact"] += n
            continue

        via = sorted(owners.get(concept, ()))
        classes = {concept_map[v] for v in via if v in concept_map}
        if len(classes) == 1:
            cls = classes.pop()
            rows.append((concept, n, cls, "ancestry", ";".join(via)))
            tally["ancestry"] += n
        elif len(classes) > 1:
            # Two queried taxa claim this concept for different classes. Usually
            # they are not siblings but NESTED: we query Actiniaria (anemones)
            # as benthic_invert AND Anthozoa as reef, and Actiniaria sits inside
            # Anthozoa, so every anemone species descends from both.
            #
            # The narrower taxon is the more deliberate statement about this
            # organism -- querying Actiniaria specifically says something that
            # querying the whole of Anthozoa does not -- so the deepest queried
            # ancestor wins. This is the same principle as the exact-match rule
            # one level up, just applied to descendants.
            deepest = [
                v for v in via
                if all(v == o or v in set(cache.get(o, ())) for o in via)
            ]
            if len(deepest) == 1:
                cls = concept_map[deepest[0]]
                rows.append((concept, n, cls, "ancestry-specific", ";".join(via)))
                tally["ancestry"] += n
                continue
            # Genuinely incomparable (siblings, not nested) -- do not guess.
            # Blank class, so an unreviewed row cannot enter the corpus.
            rows.append((concept, n, "", "CONFLICT", ";".join(via)))
            tally["conflict"] += n
        else:
            rows.append((concept, n, "", "unresolved", ""))
            tally["unresolved"] += n

    out = Path(args.out)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["concept", "boxes", "unified_class", "method", "via"])
        w.writerows(rows)

    total = sum(seen.values())
    print()
    print(f"  wrote {out.relative_to(ROOT)}  ({len(rows)} concepts)")
    print()
    for k in ("exact", "ancestry", "conflict", "unresolved"):
        n = tally[k]
        print(f"    {k:12s} {n:7d} boxes  ({100*n/total:5.1f}%)")
    print(f"    {'TOTAL':12s} {total:7d}")
    resolved = tally["exact"] + tally["ancestry"]
    print()
    print(f"  resolvable now: {resolved} / {total} boxes ({100*resolved/total:.1f}%)")
    if tally["conflict"]:
        print(f"  {tally['conflict']} boxes are in CONFLICT rows and need a human decision "
              f"before they can be used -- see the `via` column.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

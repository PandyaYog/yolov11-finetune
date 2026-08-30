"""Filesystem, download and hashing helpers."""

from __future__ import annotations

import hashlib
import logging
import shutil
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import numpy as np
import requests
from tqdm import tqdm

log = logging.getLogger("uwd")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def setup_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:3.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}PB"


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #

def http_download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    """Download with resume support. Returns dest."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    existing = tmp.stat().st_size if tmp.exists() else 0

    headers = {"Range": f"bytes={existing}-"} if existing else {}
    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        if r.status_code == 416:          # already complete
            tmp.rename(dest)
            return dest
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0)) + existing
        mode = "ab" if existing and r.status_code == 206 else "wb"
        if mode == "wb":
            existing = 0
        with open(tmp, mode) as f, tqdm(
            total=total or None, initial=existing, unit="B",
            unit_scale=True, desc=dest.name,
        ) as bar:
            for block in r.iter_content(chunk_size=chunk):
                f.write(block)
                bar.update(len(block))

    tmp.rename(dest)
    return dest


def run(cmd: list[str], cwd: Path | None = None) -> int:
    """Run a subprocess, streaming output. Returns the exit code."""
    log.info("$ %s", " ".join(cmd))
    try:
        return subprocess.call(cmd, cwd=str(cwd) if cwd else None)
    except FileNotFoundError:
        log.error("command not found: %s", cmd[0])
        return 127


def extract(archive: Path, dest: Path) -> None:
    """Extract a zip/tar into dest, guarding against path traversal."""
    dest.mkdir(parents=True, exist_ok=True)
    log.info("extracting %s -> %s", archive.name, dest)

    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as zf:
            _safe_extract_zip(zf, dest)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tf:
            _safe_extract_tar(tf, dest)
    else:
        raise ValueError(f"unrecognised archive format: {archive}")


JUNK_PARTS = ("__MACOSX", ".DS_Store", "Thumbs.db")


def _is_junk(name: str) -> bool:
    """macOS/Windows archive litter. DUO.zip ships a __MACOSX tree that would
    otherwise confuse the layout globs."""
    return any(part in name for part in JUNK_PARTS)


def _is_within(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _safe_extract_zip(zf: zipfile.ZipFile, dest: Path) -> None:
    for member in tqdm(zf.infolist(), desc="unzip", unit="f"):
        if _is_junk(member.filename):
            continue
        out = dest / member.filename
        if not _is_within(dest, out):
            raise RuntimeError(f"refusing unsafe path in archive: {member.filename}")
        zf.extract(member, dest)


def _safe_extract_tar(tf: tarfile.TarFile, dest: Path) -> None:
    for member in tqdm(tf.getmembers(), desc="untar", unit="f"):
        if _is_junk(member.name):
            continue
        out = dest / member.name
        if not _is_within(dest, out):
            raise RuntimeError(f"refusing unsafe path in archive: {member.name}")
        tf.extract(member, dest)


def flatten_single_dir(path: Path) -> None:
    """If path contains exactly one directory and nothing else, hoist its contents."""
    entries = list(path.iterdir())
    if len(entries) == 1 and entries[0].is_dir():
        inner = entries[0]
        for item in list(inner.iterdir()):
            shutil.move(str(item), str(path / item.name))
        inner.rmdir()


# --------------------------------------------------------------------------- #
# Hashing / dedupe
# --------------------------------------------------------------------------- #

def stable_bucket(key: str, buckets: int = 100) -> int:
    """Deterministic bucket for group-aware splitting. Stable across runs and
    machines, unlike hash() which is salted per-process."""
    digest = hashlib.md5(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % buckets


def dhash(gray: np.ndarray, size: int = 8) -> int:
    """Difference hash of a grayscale image. Near-duplicate video frames land
    within a small Hamming distance of each other."""
    import cv2

    small = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA)
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return bits


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------- #
# Layout resolution
# --------------------------------------------------------------------------- #

def resolve_path(root: Path, pattern: str, want_dir: bool = False) -> Path | None:
    """Resolve a configured layout path against an extracted tree.

    Tries, in order: the literal path, the pattern as a glob, then a recursive
    hunt for the basename. Mirrors of the same dataset (Kaggle vs GitHub vs
    institutional zip) nest and capitalise differently, and hard-coding one
    layout means every mirror swap becomes a config edit.
    """
    literal = root / pattern
    if literal.exists() and (literal.is_dir() if want_dir else True):
        return literal

    def pick(cands: list[Path]) -> Path | None:
        good = [c for c in cands if (c.is_dir() if want_dir else c.is_file())]
        # Shortest path wins: prefer the top-most match over something buried
        # inside a nested duplicate of the same tree.
        return sorted(good, key=lambda c: (len(c.parts), str(c)))[0] if good else None

    if any(ch in pattern for ch in "*?["):
        hit = pick(list(root.glob(pattern)))
        if hit:
            log.info("    resolved %r -> %s", pattern, hit.relative_to(root))
            return hit

    base = Path(pattern).name
    hit = pick(list(root.rglob(base)))
    if hit:
        log.info("    resolved %r -> %s", pattern, hit.relative_to(root))
        return hit

    lowered = base.lower()
    cands = [c for c in root.rglob("*") if c.name.lower() == lowered]
    hit = pick(cands)
    if hit:
        log.info("    resolved %r -> %s (case-insensitive)", pattern, hit.relative_to(root))
        return hit

    return None


def tree_summary(root: Path, max_dirs: int = 40) -> str:
    """Human-readable summary of an extracted dataset: which directories hold
    images, and which annotation files exist. Used by --inspect."""
    if not root.exists():
        return f"  {root} does not exist"

    lines: list[str] = []
    img_dirs: list[tuple[Path, int]] = []
    ann_files: list[Path] = []

    for d in sorted(root.rglob("*")):
        if d.is_dir():
            n = sum(1 for f in d.iterdir()
                    if f.is_file() and f.suffix.lower() in IMAGE_EXTS)
            if n:
                img_dirs.append((d, n))
        elif d.suffix.lower() in (".json", ".xml", ".csv", ".txt", ".mat"):
            ann_files.append(d)

    lines.append(f"  image directories ({len(img_dirs)}):")
    for d, n in sorted(img_dirs, key=lambda x: -x[1])[:max_dirs]:
        lines.append(f"    {n:>7} imgs  {d.relative_to(root)}")

    by_ext: dict[str, list[Path]] = {}
    for f in ann_files:
        by_ext.setdefault(f.suffix.lower(), []).append(f)

    lines.append(f"  annotation-ish files ({len(ann_files)}):")
    for ext, files in sorted(by_ext.items()):
        lines.append(f"    {ext}  x{len(files)}")
        for f in sorted(files, key=lambda c: (len(c.parts), str(c)))[:6]:
            lines.append(f"        {f.relative_to(root)}")
        if len(files) > 6:
            lines.append(f"        ... and {len(files) - 6} more")
    return "\n".join(lines)

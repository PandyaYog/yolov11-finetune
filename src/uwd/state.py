"""Resumable download state.

Two layers:

  Dataset level   data/raw/_state.json   what stage each dataset reached
  Item level      per-dataset progress files (FathomNet writes JSONL as it goes)

Both survive Ctrl+C. The signal handler flips a module-level flag rather than
raising, so in-flight work finishes its current item, flushes, and exits with a
consistent state file instead of a half-written one.
"""

from __future__ import annotations

import json
import os
import signal
import tempfile
import time
from pathlib import Path
from typing import Any, Iterator

from .util import log

# Stages a dataset moves through.
PENDING = "pending"
DOWNLOADING = "downloading"
EXTRACTING = "extracting"
COMPLETE = "complete"
FAILED = "failed"
MANUAL = "manual_required"

_STOP = False


def install_signal_handlers() -> None:
    """Ctrl+C requests a clean stop; a second one is left to the default handler
    so an unresponsive run can still be killed."""
    def handler(signum, frame):          # noqa: ANN001, ARG001
        global _STOP
        if _STOP:
            log.warning("second interrupt -- exiting immediately")
            signal.signal(signal.SIGINT, signal.SIG_DFL)
            raise KeyboardInterrupt
        _STOP = True
        print()
        log.warning("interrupt received -- finishing current item, then saving state")

    try:
        signal.signal(signal.SIGINT, handler)
    except ValueError:
        pass                             # not on the main thread; ignore


def stop_requested() -> bool:
    return _STOP


def atomic_write(path: Path, text: str) -> None:
    """Write via a temp file + replace so an interrupt can never leave a
    truncated state file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class State:
    """Dataset-level progress, persisted to data/raw/_state.json."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {"version": 1, "datasets": {}}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                log.warning("state file unreadable (%s) -- starting fresh", exc)
        self.data.setdefault("datasets", {})

    def get(self, name: str) -> dict:
        return self.data["datasets"].setdefault(name, {"status": PENDING})

    def status(self, name: str) -> str:
        return self.get(name).get("status", PENDING)

    def is_complete(self, name: str) -> bool:
        return self.status(name) == COMPLETE

    def update(self, name: str, **fields) -> None:
        rec = self.get(name)
        rec.update(fields)
        rec["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.save()

    def reset(self, name: str) -> None:
        self.data["datasets"].pop(name, None)
        self.save()

    def save(self) -> None:
        atomic_write(self.path, json.dumps(self.data, indent=2))


class JsonlProgress:
    """Append-only item log for long per-item jobs (the FathomNet image pull).

    Appending one line per completed item means a resume never re-downloads and
    never loses work, without holding the whole result set in memory or
    rewriting a large file on every item.
    """

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = None

    def keys(self, field: str = "uuid") -> set[str]:
        """Ids already recorded. Tolerates a truncated final line from a hard
        kill by skipping records that will not parse."""
        done: set[str] = set()
        if not self.path.exists():
            return done
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if field in rec:
                    done.add(rec[field])
        return done

    def read_all(self) -> Iterator[dict]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    def append(self, record: dict) -> None:
        if self._fh is None:
            self._fh = open(self.path, "a", encoding="utf-8")
        self._fh.write(json.dumps(record) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __enter__(self) -> JsonlProgress:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

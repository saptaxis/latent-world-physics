"""Log-based training profiler.

Writes one JSONL line per phase completion to {run_dir}/profile.jsonl.
Each line records: timestamp, step, epoch, phase name, duration in ms.

Designed for streaming analysis — every line is flushed immediately,
so you can read the log mid-training (jq, pandas, profile_summary.py).
Zero overhead when disabled (path=None).

Usage:
    log = ProfileLogger(run_dir / "profile.jsonl")
    for batch in loader:
        with log.phase("forward", step=ctx.global_step, epoch=ctx.epoch):
            loss = model(batch)
        with log.phase("backward", step=ctx.global_step, epoch=ctx.epoch):
            loss.backward()
        for cb in callbacks:
            with log.phase(f"cb/{type(cb).__name__}", step=ctx.global_step, epoch=ctx.epoch):
                cb.on_step(ctx)
    log.close()
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from pathlib import Path


class ProfileLogger:
    """Append-only JSONL profiler for training phases.

    Args:
        path: Path to write profile.jsonl. None to disable (zero overhead).
    """

    def __init__(self, path: str | Path | None):
        self._file = None
        if path is not None:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._file = open(path, "a")

    @property
    def enabled(self) -> bool:
        return self._file is not None

    @contextmanager
    def phase(self, name: str, step: int = 0, epoch: int = 0):
        if self._file is None:
            yield
            return
        t0 = time.perf_counter()
        yield
        dur_ms = (time.perf_counter() - t0) * 1000
        entry = {
            "ts": time.time(),
            "step": step,
            "epoch": epoch,
            "phase": name,
            "dur_ms": round(dur_ms, 4),
        }
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()

    def log_event(self, name: str, dur_ms: float, step: int = 0, epoch: int = 0):
        """Write a single event without using the context manager."""
        if self._file is None:
            return
        entry = {
            "ts": time.time(),
            "step": step,
            "epoch": epoch,
            "phase": name,
            "dur_ms": round(dur_ms, 4),
        }
        self._file.write(json.dumps(entry) + "\n")
        self._file.flush()

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

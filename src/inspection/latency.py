"""Latency, throughput and memory measurement (single image, model only)."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import psutil
import torch


def rss_mb() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 1024**2


class PeakRSS:
    """Samples resident memory on a background thread to capture the peak during a block."""

    def __init__(self, interval: float = 0.01) -> None:
        import threading

        self.interval, self.peak, self._stop = interval, 0.0, threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, rss_mb())
            time.sleep(self.interval)

    def __enter__(self) -> PeakRSS:
        self.peak = rss_mb()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join()
        self.peak = max(self.peak, rss_mb())


def measure_latency(
    fn: Callable[[torch.Tensor], Any], tensors: list[torch.Tensor], warmup: int = 5
) -> dict[str, float]:
    """One image at a time from decoded tensor to result. Warm-up calls are excluded."""
    for t in tensors[:warmup] if len(tensors) >= warmup else tensors:
        fn(t[None])
    times: list[float] = []
    with PeakRSS() as peak:
        for t in tensors:
            t0 = time.perf_counter()
            fn(t[None])
            times.append((time.perf_counter() - t0) * 1000.0)
    a = np.asarray(times)
    return {
        "n": int(a.size),
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "mean_ms": float(a.mean()),
        "fps": float(1000.0 / a.mean()),
        "peak_rss_mb": float(peak.peak),
    }

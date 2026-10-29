"""Bounded rolling measurements; timestamps use the same monotonic clock."""
from collections import deque
import threading
import time

import numpy as np


class Measurements:
    def __init__(self, limit=2048):
        self._lock = threading.Lock()
        self.samples = deque(maxlen=limit)
        self.times = deque(maxlen=240)
        self.count = 0

    def add(self, latency_ms=None):
        with self._lock:
            self.count += 1
            self.times.append(time.perf_counter())
            if latency_ms is not None:
                self.samples.append(float(latency_ms))

    def snapshot(self):
        with self._lock:
            now = time.perf_counter()
            recent = [stamp for stamp in self.times if now - stamp <= 2.0]
            fps = (len(recent) - 1) / (recent[-1] - recent[0]) if len(recent) > 1 else 0.0
            p50, p95 = np.percentile(self.samples, [50, 95]) if self.samples else (0, 0)
            return {"count": self.count, "fps": round(fps, 2), "p50_ms": round(float(p50), 2),
                    "p95_ms": round(float(p95), 2), "sample_count": len(self.samples)}

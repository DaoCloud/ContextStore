from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field


@dataclass
class ContextStoreMetrics:
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    tokens_served_from_cache: int = 0
    tokens_recomputed: int = 0
    bytes_loaded: int = 0
    bytes_saved: int = 0
    load_latencies_ms: list[float] = field(default_factory=list)
    save_latencies_ms: list[float] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return self.cache_hits / total

    @property
    def avg_load_latency_ms(self) -> float:
        if not self.load_latencies_ms:
            return 0.0
        return sum(self.load_latencies_ms) / len(self.load_latencies_ms)

    @property
    def avg_save_latency_ms(self) -> float:
        if not self.save_latencies_ms:
            return 0.0
        return sum(self.save_latencies_ms) / len(self.save_latencies_ms)

    @property
    def token_hit_rate(self) -> float:
        total = self.tokens_served_from_cache + self.tokens_recomputed
        if total == 0:
            return 0.0
        return self.tokens_served_from_cache / total

    def record_hit(self, num_tokens: int) -> None:
        with self._lock:
            self.cache_hits += 1
            self.tokens_served_from_cache += num_tokens

    def record_miss(self, num_tokens: int) -> None:
        with self._lock:
            self.cache_misses += 1
            self.tokens_recomputed += num_tokens

    def record_request(self) -> None:
        with self._lock:
            self.total_requests += 1

    def record_load(self, num_bytes: int, latency_ms: float) -> None:
        with self._lock:
            self.bytes_loaded += num_bytes
            self.load_latencies_ms.append(latency_ms)

    def record_save(self, num_bytes: int, latency_ms: float) -> None:
        with self._lock:
            self.bytes_saved += num_bytes
            self.save_latencies_ms.append(latency_ms)

    def summary(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "hit_rate": f"{self.hit_rate:.2%}",
            "token_hit_rate": f"{self.token_hit_rate:.2%}",
            "bytes_loaded_mb": self.bytes_loaded / 1024**2,
            "bytes_saved_mb": self.bytes_saved / 1024**2,
            "avg_load_latency_ms": f"{self.avg_load_latency_ms:.2f}",
            "avg_save_latency_ms": f"{self.avg_save_latency_ms:.2f}",
        }

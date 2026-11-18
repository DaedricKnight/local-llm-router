"""Telemetry and metrics collector for latency, token savings, and fallbacks."""

from __future__ import annotations

import threading
from typing import Dict, List

from local_llm_router.models import TelemetryStats


class TelemetryCollector:
    """Thread-safe telemetry collector."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total_requests = 0
        self._successful_requests = 0
        self._failed_requests = 0
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_bypasses = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._tokens_saved = 0
        self._validation_retries = 0
        self._fallback_events = 0
        self._latencies: List[float] = []
        self._provider_calls: Dict[str, int] = {}

    def record_request(
        self,
        success: bool,
        latency_ms: float,
        provider: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record completed request outcome."""
        with self._lock:
            self._total_requests += 1
            if success:
                self._successful_requests += 1
            else:
                self._failed_requests += 1

            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._provider_calls[provider] = self._provider_calls.get(provider, 0) + 1

            if len(self._latencies) >= 2000:
                self._latencies.pop(0)
            self._latencies.append(latency_ms)

    def record_cache_hit(self, tokens_saved: int) -> None:
        """Record a cache hit and saved tokens."""
        with self._lock:
            self._cache_hits += 1
            self._tokens_saved += tokens_saved

    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self._cache_misses += 1

    def record_cache_bypass(self) -> None:
        """Record a cache bypass request."""
        with self._lock:
            self._cache_bypasses += 1

    def record_fallback(self, provider: str, model: str) -> None:
        """Record failover to a backup provider/model."""
        with self._lock:
            self._fallback_events += 1

    def record_validation_retry(self) -> None:
        """Record a structured JSON retry attempt."""
        with self._lock:
            self._validation_retries += 1

    def get_stats(self) -> TelemetryStats:
        """Calculate and return snapshot of current telemetry metrics."""
        with self._lock:
            total_cache_lookups = self._cache_hits + self._cache_misses
            hit_rate = (
                (self._cache_hits / total_cache_lookups * 100.0)
                if total_cache_lookups > 0
                else 0.0
            )

            sorted_latencies = sorted(self._latencies) if self._latencies else [0.0]
            avg_lat = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
            p95_index = int(len(sorted_latencies) * 0.95)
            p95_lat = sorted_latencies[min(p95_index, len(sorted_latencies) - 1)]

            return TelemetryStats(
                total_requests=self._total_requests,
                successful_requests=self._successful_requests,
                failed_requests=self._failed_requests,
                cache_hits=self._cache_hits,
                cache_misses=self._cache_misses,
                cache_bypasses=self._cache_bypasses,
                cache_hit_rate_pct=round(hit_rate, 2),
                total_prompt_tokens=self._prompt_tokens,
                total_completion_tokens=self._completion_tokens,
                total_tokens_saved=self._tokens_saved,
                validation_retries_total=self._validation_retries,
                fallback_events_total=self._fallback_events,
                avg_latency_ms=round(avg_lat, 2),
                p95_latency_ms=round(p95_lat, 2),
                provider_calls=dict(self._provider_calls),
            )

    def reset(self) -> None:
        """Reset all metric counters."""
        with self._lock:
            self._total_requests = 0
            self._successful_requests = 0
            self._failed_requests = 0
            self._cache_hits = 0
            self._cache_misses = 0
            self._cache_bypasses = 0
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._tokens_saved = 0
            self._validation_retries = 0
            self._fallback_events = 0
            self._latencies.clear()
            self._provider_calls.clear()

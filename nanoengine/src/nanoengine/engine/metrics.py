"""Prometheus metrics for ``/metrics``, kept in a registry per engine so tests stay isolated."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from nanoengine.constants import STEP_LATENCY_BUCKETS_SECONDS

STEP_KIND_PREFILL = "prefill"
STEP_KIND_DECODE = "decode"


class EngineMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self._peak_running_count = 0
        self.finished_requests = Counter(
            "nanoengine_finished_requests",
            "Requests that finished, by finish reason",
            ["finish_reason"],
            registry=self.registry,
        )
        self.generated_tokens = Counter(
            "nanoengine_generated_tokens", "Tokens generated", registry=self.registry
        )
        self.step_latency_seconds = Histogram(
            "nanoengine_step_latency_seconds",
            "Time for one model step; prefill runs whole prompts, decode one token per sequence",
            ["step_kind"],
            buckets=STEP_LATENCY_BUCKETS_SECONDS,
            registry=self.registry,
        )
        self.waiting_requests = Gauge(
            "nanoengine_waiting_requests", "Requests queued, not started", registry=self.registry
        )
        self.running_requests = Gauge(
            "nanoengine_running_requests", "Requests being generated", registry=self.registry
        )
        self.peak_running_requests = Gauge(
            "nanoengine_peak_running_requests",
            "Most requests generated at the same time since start",
            registry=self.registry,
        )
        self.preemptions = Counter(
            "nanoengine_preemptions",
            "Sequences evicted from the KV cache to be recomputed later",
            registry=self.registry,
        )
        self.kv_cache_total_slots = Gauge(
            "nanoengine_kv_cache_total_slots", "Token slots in the KV pool", registry=self.registry
        )
        self.kv_cache_reserved_slots = Gauge(
            "nanoengine_kv_cache_reserved_slots",
            "Token slots reserved by running sequences",
            registry=self.registry,
        )
        self.kv_cache_stored_tokens = Gauge(
            "nanoengine_kv_cache_stored_tokens",
            "Tokens actually stored; stored / reserved is the memory utilization",
            registry=self.registry,
        )

    def set_running_requests(self, count: int) -> None:
        self.running_requests.set(count)
        self._peak_running_count = max(self._peak_running_count, count)
        self.peak_running_requests.set(self._peak_running_count)

    def render(self) -> bytes:
        return generate_latest(self.registry)

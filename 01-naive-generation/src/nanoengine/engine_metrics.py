"""Prometheus metrics for ``/metrics``, kept in a registry per engine so tests stay isolated."""

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from nanoengine.constants import STEP_LATENCY_BUCKETS_SECONDS


class EngineMetrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
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
            "Time for one model step that produces one token",
            buckets=STEP_LATENCY_BUCKETS_SECONDS,
            registry=self.registry,
        )
        self.waiting_requests = Gauge(
            "nanoengine_waiting_requests", "Requests queued, not started", registry=self.registry
        )
        self.running_requests = Gauge(
            "nanoengine_running_requests", "Requests being generated", registry=self.registry
        )

    def render(self) -> bytes:
        return generate_latest(self.registry)

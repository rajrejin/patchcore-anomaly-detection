"""Prometheus metrics. Labels carry category and stage only, never image content."""

from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

REGISTRY = CollectorRegistry()

REQUESTS = Counter(
    "inspection_requests_total", "HTTP requests", ["route", "status"], registry=REGISTRY
)
ERRORS = Counter("inspection_errors_total", "API errors by code", ["code"], registry=REGISTRY)
INSPECTIONS = Counter(
    "inspection_results_total",
    "Inspection verdicts",
    ["category", "stage", "verdict"],
    registry=REGISTRY,
)
LATENCY = Histogram(
    "inspection_latency_seconds",
    "End-to-end inspection latency",
    ["category", "stage"],
    buckets=(0.05, 0.1, 0.2, 0.4, 0.8, 1.5, 3, 6, 12),
    registry=REGISTRY,
)
QUEUE_WAIT = Histogram(
    "inspection_queue_wait_seconds",
    "Time waiting for an inference slot",
    buckets=(0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 30),
    registry=REGISTRY,
)

"""
metrics.py — Prometheus metrics for the inspection API.

Defines the metrics exposed at /metrics (scraped by Prometheus, see
monitoring/prometheus.yml) and helper functions to record them from
routes.py without cluttering the endpoint logic itself.

Metrics exposed:
    inspection_requests_total{severity, risk_level} -- Counter
        Total inspections processed, labeled by outcome. Lets you graph
        "how many severe defects this week" or alert on a spike in
        critical-risk results.
    inspection_duration_seconds -- Histogram
        End-to-end request latency (CV inference + all 4 agents). Lets
        you track p50/p95/p99 latency and catch performance regressions.
    cv_inference_duration_seconds -- Histogram
        Just the CV model forward pass time, separate from the full
        pipeline -- helps distinguish "the model got slower" from "an
        agent (e.g. the LLM report) got slower".
    inspection_errors_total{stage} -- Counter
        Errors by which stage failed (cv_inference, pipeline), for
        alerting on elevated failure rates.
    model_confidence -- Histogram
        Distribution of CV model confidence scores. A sustained shift in
        this distribution is one of the simplest drift signals (see
        drift.py for a more formal statistical test on this data).

Run this file directly to verify metrics can be recorded and rendered
in Prometheus text-exposition format without needing a running server.
"""

from __future__ import annotations

import time
from contextlib import contextmanager

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

inspection_requests_total = Counter(
    "inspection_requests_total",
    "Total number of inspection requests processed, by outcome.",
    ["severity", "risk_level"],
)

inspection_duration_seconds = Histogram(
    "inspection_duration_seconds",
    "End-to-end inspection request latency (CV inference + all 4 agents).",
    buckets=(0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
)

cv_inference_duration_seconds = Histogram(
    "cv_inference_duration_seconds",
    "CV model forward-pass latency only (excludes agent pipeline).",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

inspection_errors_total = Counter(
    "inspection_errors_total",
    "Total inspection errors, by which stage failed.",
    ["stage"],
)

model_confidence = Histogram(
    "model_confidence",
    "Distribution of CV model max predicted probability (confidence) per inspection.",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
)


@contextmanager
def time_histogram(histogram: Histogram):
    """Context manager: times a block and records it into `histogram`."""
    start = time.perf_counter()
    try:
        yield
    finally:
        histogram.observe(time.perf_counter() - start)


def record_inspection_result(severity: str, risk_level: str, confidence: float) -> None:
    """Records the outcome of a completed inspection."""
    inspection_requests_total.labels(severity=severity, risk_level=risk_level).inc()
    model_confidence.observe(confidence)


def record_error(stage: str) -> None:
    """Records that an inspection failed at a given pipeline stage."""
    inspection_errors_total.labels(stage=stage).inc()


def render_metrics() -> tuple[bytes, str]:
    """Returns (body, content_type) for the /metrics endpoint."""
    return generate_latest(), CONTENT_TYPE_LATEST


if __name__ == "__main__":
    print("=== Metrics recording test ===")

    # Simulate a few inspections with varying outcomes.
    with time_histogram(inspection_duration_seconds):
        time.sleep(0.05)
    record_inspection_result(severity="none", risk_level="low", confidence=0.98)

    with time_histogram(inspection_duration_seconds):
        time.sleep(0.03)
    record_inspection_result(severity="moderate", risk_level="medium", confidence=0.87)

    with time_histogram(cv_inference_duration_seconds):
        time.sleep(0.01)

    record_error(stage="cv_inference")

    print("Recorded 2 inspections, 1 CV timing, 1 error.\n")

    print("=== Rendered Prometheus exposition format (excerpt) ===")
    body, content_type = render_metrics()
    text = body.decode("utf-8")
    print(f"Content-Type: {content_type}\n")

    # Verify the key metric names actually appear in the output.
    required_metrics = [
        "inspection_requests_total",
        "inspection_duration_seconds",
        "cv_inference_duration_seconds",
        "inspection_errors_total",
        "model_confidence",
    ]
    for metric_name in required_metrics:
        assert metric_name in text, f"Expected metric '{metric_name}' not found in output."
        print(f"  [OK] {metric_name} present in exposition output")

    # Print a relevant excerpt so it's visible what real output looks like.
    print("\n--- Sample lines ---")
    for line in text.splitlines():
        if line.startswith("inspection_requests_total") or line.startswith("inspection_errors_total"):
            print(f"  {line}")

    print("\nAll metrics checks passed.")
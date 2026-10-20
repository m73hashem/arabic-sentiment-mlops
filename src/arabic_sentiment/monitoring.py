"""Small, local prediction-drift utilities and Prometheus metrics."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

PSI_ALERT_THRESHOLD = 0.25
PSI_REPORT_PATH_ENV = "SENTIMENT_PSI_REPORT_PATH"
DEFAULT_PSI_REPORT_PATH = Path("reports/monitoring/drift-report.json")

REGISTRY = CollectorRegistry()
PREDICTION_REQUESTS = Counter(
    "sentiment_prediction_requests_total",
    "Prediction requests handled by the API",
    labelnames=("status",),
    registry=REGISTRY,
)
PREDICTION_CONFIDENCE = Histogram(
    "sentiment_prediction_confidence_score",
    "Distribution of returned prediction confidence scores",
    buckets=(0.0, 0.25, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0),
    registry=REGISTRY,
)
PREDICTION_DURATION = Histogram(
    "sentiment_prediction_request_duration_seconds",
    "Prediction request duration in seconds",
    labelnames=("status",),
    registry=REGISTRY,
)
PSI_SCORE = Gauge(
    "sentiment_prediction_psi",
    "Most recently completed prediction-confidence PSI comparison",
    labelnames=("source",),
    registry=REGISTRY,
)
PSI_THRESHOLD = Gauge(
    "sentiment_prediction_psi_alert_threshold",
    "Configured PSI alert threshold; alert when PSI is strictly greater",
    registry=REGISTRY,
)
PSI_THRESHOLD.set(PSI_ALERT_THRESHOLD)


def psi_alert(psi: float, threshold: float = PSI_ALERT_THRESHOLD) -> bool:
    """Return whether PSI strictly exceeds the configured alert threshold."""
    if not math.isfinite(psi) or not math.isfinite(threshold):
        raise ValueError("PSI and threshold must be finite")
    if psi < 0 or threshold < 0:
        raise ValueError("PSI and threshold must be non-negative")
    return psi > threshold


def population_stability_index(
    reference: Sequence[float], current: Sequence[float], *, bins: int = 10
) -> float:
    """Compute PSI with reference-quantile bins and epsilon-smoothed proportions.

    Inputs must be non-empty finite confidence scores in [0, 1]. Duplicate
    quantile edges are collapsed; a constant reference uses two open-ended bins.
    Epsilon smoothing keeps zero-frequency bins finite without hiding empty input.
    """
    if bins < 2:
        raise ValueError("bins must be at least 2")
    ref = np.asarray(reference, dtype=np.float64)
    cur = np.asarray(current, dtype=np.float64)
    if ref.size == 0 or cur.size == 0:
        raise ValueError("reference and current confidence samples must be non-empty")
    if not np.isfinite(ref).all() or not np.isfinite(cur).all():
        raise ValueError("confidence samples must be finite")
    if ((ref < 0) | (ref > 1)).any() or ((cur < 0) | (cur > 1)).any():
        raise ValueError("confidence samples must be in [0, 1]")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    unique_quantiles = np.unique(np.quantile(ref, quantiles))
    interior_edges = unique_quantiles[1:-1]
    if interior_edges.size == 0:
        if ref.min() == ref.max():
            split = np.nextafter(ref[0], np.inf)
            if split > 1.0:
                split = np.nextafter(ref[0], -np.inf)
            interior_edges = np.array([split], dtype=np.float64)
        else:
            interior_edges = np.array([(ref.min() + ref.max()) / 2])
    edges = np.concatenate(([-np.inf], interior_edges, [np.inf]))

    ref_counts = np.histogram(ref, bins=edges)[0].astype(np.float64)
    cur_counts = np.histogram(cur, bins=edges)[0].astype(np.float64)
    epsilon = 1e-6
    ref_share = (ref_counts + epsilon) / (ref.size + epsilon * ref_counts.size)
    cur_share = (cur_counts + epsilon) / (cur.size + epsilon * cur_counts.size)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def refresh_psi_metrics(report_path: Path | None = None) -> bool:
    """Publish the latest persisted PSI report, removing stale scores if absent."""
    selected_path = report_path or Path(
        os.environ.get(PSI_REPORT_PATH_ENV, DEFAULT_PSI_REPORT_PATH)
    )
    try:
        report = json.loads(selected_path.read_text(encoding="utf-8"))
        psi = report["psi"]
        threshold = report["threshold"]
        alert = report["alert"]
        if (
            report.get("metric") != "confidence_score"
            or isinstance(psi, bool)
            or not isinstance(psi, (int, float))
            or not math.isfinite(psi)
            or psi < 0
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or threshold != PSI_ALERT_THRESHOLD
            or not isinstance(alert, bool)
            or alert != psi_alert(psi, threshold)
        ):
            raise ValueError("invalid PSI report values")
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        PSI_SCORE.remove("latest")
        return False

    PSI_SCORE.labels(source="latest").set(psi)
    PSI_THRESHOLD.set(PSI_ALERT_THRESHOLD)
    return True


def prometheus_payload(report_path: Path | None = None) -> tuple[bytes, str]:
    """Refresh persisted PSI evidence, then render the API metrics registry."""
    refresh_psi_metrics(report_path)
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


def read_confidence_scores(path: Path) -> tuple[list[float], set[str]]:
    """Read confidence scores and unique source IDs without loading review text."""
    scores: list[float] = []
    identifiers: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"source_row_id", "confidence_score"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path.name} must contain source_row_id and confidence_score"
            )
        for line_number, row in enumerate(reader, start=2):
            identifier = row["source_row_id"].strip()
            if not identifier or identifier in identifiers:
                raise ValueError(
                    f"{path.name} has a blank or duplicate source_row_id "
                    f"at line {line_number}"
                )
            identifiers.add(identifier)
            try:
                score = float(row["confidence_score"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path.name} has invalid confidence_score at line {line_number}"
                ) from exc
            scores.append(score)
    return scores, identifiers


def file_sha256(path: Path) -> str:
    """Hash a file incrementally for reproducible evidence provenance."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_confidence_files(
    reference_path: Path, current_path: Path
) -> dict[str, object]:
    """Compare two disjoint confidence exports and return a text-free report."""
    if reference_path.resolve() == current_path.resolve():
        raise ValueError("reference and current files must be different")
    reference, reference_ids = read_confidence_scores(reference_path)
    current, current_ids = read_confidence_scores(current_path)
    overlap = reference_ids & current_ids
    if overlap:
        raise ValueError(
            f"reference and current data share {len(overlap)} source_row_id values"
        )
    psi = population_stability_index(reference, current)
    return {
        "metric": "confidence_score",
        "method": "reference-quantile bins with epsilon smoothing",
        "psi": psi,
        "threshold": PSI_ALERT_THRESHOLD,
        "alert": psi_alert(psi),
        "comparison": "alert when PSI > 0.25; equality does not alert",
        "reference": {
            "file": reference_path.name,
            "rows": len(reference),
            "sha256": file_sha256(reference_path),
        },
        "current": {
            "file": current_path.name,
            "rows": len(current),
            "sha256": file_sha256(current_path),
        },
    }

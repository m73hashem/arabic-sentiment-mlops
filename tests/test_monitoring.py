import csv
import importlib.util
import json
import math
from pathlib import Path

import pytest

from arabic_sentiment.monitoring import (
    PSI_ALERT_THRESHOLD,
    compare_confidence_files,
    population_stability_index,
    psi_alert,
)

DRIFT_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "check_prediction_drift.py"
)
DRIFT_SPEC = importlib.util.spec_from_file_location(
    "check_prediction_drift", DRIFT_SCRIPT
)
assert DRIFT_SPEC is not None and DRIFT_SPEC.loader is not None
DRIFT_MODULE = importlib.util.module_from_spec(DRIFT_SPEC)
DRIFT_SPEC.loader.exec_module(DRIFT_MODULE)
write_dashboard = DRIFT_MODULE.write_dashboard


def test_psi_is_zero_for_identical_confidence_distributions():
    values = [0.55, 0.61, 0.72, 0.83, 0.94]

    assert population_stability_index(values, values) == pytest.approx(0.0)


def test_psi_zero_frequency_bins_are_smoothed_and_finite():
    psi = population_stability_index([0.2] * 20, [0.9] * 20)

    assert math.isfinite(psi)
    assert psi > PSI_ALERT_THRESHOLD


@pytest.mark.parametrize("reference,current", [([], [0.5]), ([0.5], [])])
def test_psi_rejects_empty_samples(reference, current):
    with pytest.raises(ValueError, match="non-empty"):
        population_stability_index(reference, current)


def test_psi_rejects_non_finite_and_out_of_range_confidence_values():
    with pytest.raises(ValueError, match="finite"):
        population_stability_index([0.5], [float("nan")])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        population_stability_index([0.5], [1.1])


def test_alert_threshold_is_strictly_greater_than_point_25():
    assert not psi_alert(0.25)
    assert not psi_alert(PSI_ALERT_THRESHOLD)
    assert psi_alert(0.250001)


def test_comparison_requires_distinct_non_overlapping_source_rows(tmp_path):
    reference = tmp_path / "reference.csv"
    current = tmp_path / "current.csv"
    for path, identifier, score in (
        (reference, "reference-1", 0.7),
        (current, "current-1", 0.8),
    ):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=("source_row_id", "confidence_score")
            )
            writer.writeheader()
            writer.writerow({"source_row_id": identifier, "confidence_score": score})

    report = compare_confidence_files(reference, current)

    assert report["metric"] == "confidence_score"
    assert report["reference"]["rows"] == 1
    assert report["current"]["rows"] == 1
    assert report["threshold"] == PSI_ALERT_THRESHOLD

    current.write_text(
        "source_row_id,confidence_score\nreference-1,0.8\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="share 1 source_row_id"):
        compare_confidence_files(reference, current)


def test_comparison_rejects_same_path_and_missing_or_invalid_inputs(tmp_path):
    reference = tmp_path / "reference.csv"
    reference.write_text(
        "source_row_id,confidence_score\nreference-1,0.7\n", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="files must be different"):
        compare_confidence_files(reference, reference)
    with pytest.raises(FileNotFoundError):
        compare_confidence_files(reference, tmp_path / "missing.csv")

    invalid = tmp_path / "invalid.csv"
    invalid.write_text("source_row_id,review\nrow-1,secret review\n", encoding="utf-8")
    with pytest.raises(
        ValueError, match="must contain source_row_id and confidence_score"
    ):
        compare_confidence_files(reference, invalid)


def test_drift_report_and_html_do_not_include_review_text(tmp_path):
    reference = tmp_path / "reference.csv"
    current = tmp_path / "current.csv"
    private_review = "private review text that must not appear in evidence"
    for path, identifier, score in (
        (reference, "reference-1", 0.7),
        (current, "current-1", 0.8),
    ):
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=("source_row_id", "confidence_score", "review"),
            )
            writer.writeheader()
            writer.writerow(
                {
                    "source_row_id": identifier,
                    "confidence_score": score,
                    "review": private_review,
                }
            )

    report = compare_confidence_files(reference, current)
    json_evidence = json.dumps(report)
    html_path = tmp_path / "drift-report.html"
    write_dashboard(report, html_path)

    assert private_review not in json_evidence
    assert private_review not in html_path.read_text(encoding="utf-8")
    assert report["reference"]["rows"] == report["current"]["rows"] == 1
    assert report["threshold"] == PSI_ALERT_THRESHOLD

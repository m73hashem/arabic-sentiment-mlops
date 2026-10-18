"""Fail-closed comparison of candidate DVC metrics with a configured baseline."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


class QualityGateError(ValueError):
    """Raised when quality-gate inputs are missing or do not match the schema."""


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise QualityGateError(f"Cannot read {description} file {path}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise QualityGateError(
            f"Malformed JSON in {description} file {path}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise QualityGateError(
            f"Unsupported {description} schema in {path}: expected an object"
        )
    return value


def _score(value: Any, description: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QualityGateError(f"{description} must be a numeric f1_macro value")
    try:
        score = float(value)
    except (OverflowError, ValueError) as exc:
        raise QualityGateError(
            f"{description} must be finite and within [0, 1]"
        ) from exc
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise QualityGateError(f"{description} must be finite and within [0, 1]")
    return score


def check_quality(
    candidate_path: str | Path, baseline_path: str | Path
) -> tuple[float, float]:
    """Return candidate and baseline scores; equality passes, regression fails."""
    candidate_file = Path(candidate_path)
    baseline_file = Path(baseline_path)
    candidate = _read_json(candidate_file, "candidate metrics")
    baseline_config = _read_json(baseline_file, "quality-gate configuration")

    expected_baseline_keys = {"schema_version", "metric", "split", "baseline"}
    if set(baseline_config) != expected_baseline_keys:
        raise QualityGateError(
            "Unsupported quality-gate configuration schema: expected exactly "
            f"{sorted(expected_baseline_keys)}"
        )
    if (
        isinstance(baseline_config["schema_version"], bool)
        or baseline_config["schema_version"] != 1
    ):
        raise QualityGateError(
            f"Unsupported quality-gate schema version: {baseline_config['schema_version']!r}"
        )
    if baseline_config["metric"] != "f1_macro":
        raise QualityGateError(
            f"Unsupported quality metric {baseline_config['metric']!r}; expected 'f1_macro'"
        )
    split = baseline_config["split"]
    if split != "test":
        raise QualityGateError(
            f"Unsupported quality-gate split {split!r}; expected 'test'"
        )

    if set(candidate) != {"validation", "test"}:
        raise QualityGateError(
            "Unsupported candidate metrics schema: expected exactly validation and test splits"
        )
    for split_name in ("validation", "test"):
        split_metrics = candidate[split_name]
        if not isinstance(split_metrics, dict) or "f1_macro" not in split_metrics:
            raise QualityGateError(
                f"Candidate metrics are missing {split_name}.f1_macro in {candidate_file}"
            )
        _score(split_metrics["f1_macro"], f"Candidate {split_name}.f1_macro")

    baseline = baseline_config["baseline"]
    if not isinstance(baseline, dict) or set(baseline) != {"value", "source", "model"}:
        raise QualityGateError(
            "Unsupported baseline schema: expected value, source, and model fields"
        )
    if not all(
        isinstance(baseline[key], str) and baseline[key].strip()
        for key in ("source", "model")
    ):
        raise QualityGateError("Baseline source and model must be non-empty strings")

    candidate_score = _score(
        candidate[split]["f1_macro"], f"Candidate {split}.f1_macro"
    )
    baseline_score = _score(baseline["value"], "Configured baseline f1_macro")
    return candidate_score, baseline_score


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    args = parser.parse_args(argv)

    try:
        candidate, baseline = check_quality(args.candidate, args.baseline)
    except QualityGateError as exc:
        print(f"QUALITY GATE ERROR: {exc}", file=sys.stderr)
        return 2

    comparison = ">=" if candidate >= baseline else "<"
    print(
        f"Candidate test f1_macro={candidate:.10f} {comparison} "
        f"baseline={baseline:.10f}"
    )
    if candidate < baseline:
        print(
            "QUALITY GATE FAIL: candidate is below the configured baseline",
            file=sys.stderr,
        )
        return 1
    print("QUALITY GATE PASS: equality with the baseline is accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

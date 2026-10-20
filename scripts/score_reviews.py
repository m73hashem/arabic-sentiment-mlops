"""Batch-score a CSV of reviews with the configured local production model."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from arabic_sentiment.serving import LocalAraBERTService

MINIMUM_REVIEWS = 1_000
DEFAULT_OUTPUT = Path("data/scoring/output/predictions.csv")
IDENTIFIER_COLUMNS = ("source_row_id", "id", "no", "row_id")
OUTPUT_COLUMNS = ("source_row_id", "label", "confidence_score", "model_version")


def _read_reviews(input_path: Path) -> list[tuple[str, str]]:
    """Read reviews and stable source identifiers before loading the model."""
    with input_path.open(encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        if not reader.fieldnames or "review" not in reader.fieldnames:
            raise ValueError("input CSV must contain a review column")
        id_column = next(
            (column for column in IDENTIFIER_COLUMNS if column in reader.fieldnames),
            None,
        )
        records = []
        seen_ids: set[str] = set()
        for row_number, row in enumerate(reader, start=2):
            source_id = (
                row[id_column].strip()
                if id_column is not None and row[id_column] is not None
                else f"{input_path.stem}:{row_number}"
            )
            if not source_id:
                raise ValueError(f"blank row identifier at CSV line {row_number}")
            if source_id in seen_ids:
                raise ValueError(f"duplicate row identifier {source_id!r}")
            seen_ids.add(source_id)
            review = row["review"]
            if review is None or not review.strip():
                raise ValueError(
                    f"blank review at CSV line {row_number} (source_row_id={source_id})"
                )
            records.append((source_id, review))
    if len(records) < MINIMUM_REVIEWS:
        raise ValueError(
            f"input must contain at least {MINIMUM_REVIEWS} reviews; found {len(records)}"
        )
    return records


def score_reviews(
    input_path: Path,
    output_path: Path = DEFAULT_OUTPUT,
    *,
    batch_size: int = 32,
    service_factory: Callable[[], Any] = LocalAraBERTService,
) -> int:
    """Score all input reviews or fail without leaving a partial output file."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")

    records = _read_reviews(input_path)
    service = service_factory()
    model_version = str(service.model_version).strip()
    if not model_version:
        raise ValueError("model service returned an empty model_version")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output_file:
            temporary_path = Path(output_file.name)
            writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS)
            writer.writeheader()
            for offset in range(0, len(records), batch_size):
                batch = records[offset : offset + batch_size]
                batch_ids = [source_id for source_id, _ in batch]
                try:
                    predictions = service.predict_many([review for _, review in batch])
                except Exception as exc:
                    raise RuntimeError(
                        "inference failed for batch source_row_id values "
                        f"{batch_ids[0]!r} through {batch_ids[-1]!r}: {exc}"
                    ) from exc
                if len(predictions) != len(batch):
                    raise RuntimeError(
                        "predict_many returned an unexpected result count for "
                        f"source_row_id values {batch_ids[0]!r} through {batch_ids[-1]!r}"
                    )
                for (source_id, _), prediction in zip(batch, predictions):
                    label = prediction.get("label")
                    confidence = prediction.get("confidence")
                    if label not in {"negative", "positive"}:
                        raise ValueError(
                            f"invalid predicted label for source_row_id {source_id!r}"
                        )
                    if (
                        isinstance(confidence, bool)
                        or not isinstance(confidence, (int, float))
                        or not math.isfinite(confidence)
                        or not 0.0 <= confidence <= 1.0
                    ):
                        raise ValueError(
                            f"invalid confidence score for source_row_id {source_id!r}"
                        )
                    writer.writerow(
                        {
                            "source_row_id": source_id,
                            "label": label,
                            "confidence_score": confidence,
                            "model_version": model_version,
                        }
                    )
        os.replace(temporary_path, output_path)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="CSV with a review column"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="prediction CSV output path"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    try:
        count = score_reviews(args.input, args.output, batch_size=args.batch_size)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Batch scoring failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"Scored {count} reviews to {args.output}")


if __name__ == "__main__":
    main()

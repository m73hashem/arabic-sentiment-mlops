"""Export confidence scores from real local model predictions, without review text."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from arabic_sentiment.serving import LocalAraBERTService


def export_confidences(
    input_path: Path, output_path: Path, *, batch_size: int = 32
) -> int:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("input and output paths must differ")

    service = LocalAraBERTService()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with input_path.open(encoding="utf-8-sig", newline="") as source_file:
        reader = csv.DictReader(source_file)
        if not reader.fieldnames or "review" not in reader.fieldnames:
            raise ValueError("input CSV must contain a review column")
        has_id = "source_row_id" in reader.fieldnames
        with output_path.open("w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(
                output_file,
                fieldnames=("source_row_id", "confidence_score", "model_version"),
            )
            writer.writeheader()
            while rows := [row for _, row in zip(range(batch_size), reader)]:
                offset = count
                reviews = [row["review"] for row in rows]
                predictions = service.predict_many(reviews)
                if len(predictions) != len(rows):
                    raise RuntimeError(
                        "predict_many returned an unexpected result count"
                    )
                for position, (row, prediction) in enumerate(zip(rows, predictions)):
                    source_id = (
                        row["source_row_id"].strip()
                        if has_id
                        else f"{input_path.stem}:{offset + position + 1}"
                    )
                    if not source_id:
                        raise ValueError("source_row_id values must not be blank")
                    writer.writerow(
                        {
                            "source_row_id": source_id,
                            "confidence_score": prediction["confidence"],
                            "model_version": service.model_version,
                        }
                    )
                count += len(rows)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="CSV with review column"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    count = export_confidences(args.input, args.output, batch_size=args.batch_size)
    print(f"Exported {count} confidence scores to {args.output}")


if __name__ == "__main__":
    main()

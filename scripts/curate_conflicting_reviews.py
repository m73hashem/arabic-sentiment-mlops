"""Create a raw-data copy excluding reviews with conflicting mapped labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from arabic_sentiment.data import load_raw_data
from arabic_sentiment.preprocessing import clean_review_text, prepare_reviews


def curate(source: Path, destination: Path) -> dict[str, object]:
    """Write a curated UTF-16 TSV without changing source rows or row order."""
    source = source.resolve()
    destination = destination.resolve()
    if source == destination:
        raise ValueError("Source and destination must be different paths")

    raw = load_raw_data(source)
    prepared, _ = prepare_reviews(raw)
    label_counts = prepared.groupby("review", sort=False)["sentiment"].nunique()
    conflicting_reviews = set(label_counts[label_counts.gt(1)].index)
    normalized = clean_review_text(raw["review"])
    excluded = normalized.isin(conflicting_reviews)
    curated = raw.loc[~excluded].copy()

    destination.parent.mkdir(parents=True, exist_ok=True)
    curated.to_csv(
        destination,
        sep="\t",
        encoding="utf-16",
        index=False,
        lineterminator="\r\n",
    )

    curated_prepared, _ = prepare_reviews(curated)
    return {
        "source": str(source),
        "destination": str(destination),
        "source_rows": len(raw),
        "conflicting_normalized_review_groups": len(conflicting_reviews),
        "excluded_rows": int(excluded.sum()),
        "excluded_mapped_label_counts": {
            str(label): int(count)
            for label, count in prepared.loc[
                prepared["review"].isin(conflicting_reviews), "sentiment"
            ]
            .value_counts()
            .items()
        },
        "output_rows": len(curated),
        "output_mapped_label_counts": {
            str(label): int(count)
            for label, count in curated_prepared["sentiment"].value_counts().items()
        },
        "remaining_conflicting_groups": int(
            curated_prepared.groupby("review", sort=False)["sentiment"]
            .nunique()
            .gt(1)
            .sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(curate(args.source, args.destination), indent=2, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()

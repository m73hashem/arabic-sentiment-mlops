"""Run the Phase 1 data pipeline with ``PYTHONPATH=src python -m arabic_sentiment``."""

import argparse
from pathlib import Path

from .data import load_raw_data
from .preprocessing import prepare_reviews
from .split import (
    DEFAULT_SEED,
    SPLIT_FRACTIONS,
    SPLIT_NAMES,
    split_reviews,
    verify_no_review_leakage,
)


def run_pipeline(
    raw_path: Path,
    output_dir: Path,
    seed: int = DEFAULT_SEED,
    split_ratios: tuple[float, float, float] = tuple(SPLIT_FRACTIONS),
) -> dict:
    raw = load_raw_data(raw_path)
    prepared, preprocessing_stats = prepare_reviews(raw)
    splits = split_reviews(prepared, seed=seed, ratios=split_ratios)
    if not verify_no_review_leakage(splits):
        raise RuntimeError("Duplicate review text crossed split boundaries")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name in SPLIT_NAMES:
        splits[name].to_csv(output_dir / f"{name}.csv", index=False, encoding="utf-8")

    summary = {
        "raw_rows": len(raw),
        "raw_columns": raw.columns.tolist(),
        "raw_shape": list(raw.shape),
        "pandas_dtypes": {column: str(dtype) for column, dtype in raw.dtypes.items()},
        "missing_by_column": {
            column: int(value) for column, value in raw.isna().sum().items()
        },
        "rating_distribution": {
            str(key): int(value)
            for key, value in raw["rating"]
            .value_counts(dropna=False)
            .sort_index()
            .items()
        },
        "duplicate_full_rows": int(raw.duplicated().sum()),
        "missing_hotel_names": int(raw["Hotel name"].isna().sum()),
        "preprocessing": preprocessing_stats,
        "usable_rows": len(prepared),
        "class_distribution": {
            "negative": int(prepared["sentiment"].eq("negative").sum()),
            "positive": int(prepared["sentiment"].eq("positive").sum()),
        },
        "seed": seed,
        "splits": {
            name: {
                "rows": len(frame),
                "proportion": len(frame) / len(prepared),
                "class_distribution": {
                    label: int(frame["sentiment"].eq(label).sum())
                    for label in ("negative", "positive")
                },
            }
            for name, frame in splits.items()
        },
        "duplicate_leakage_free": verify_no_review_leakage(splits),
        "output_dir": str(output_dir),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare deterministic HARD sentiment data splits"
    )
    parser.add_argument(
        "--raw", type=Path, default=Path("data/raw/balanced-reviews.txt")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    summary = run_pipeline(args.raw, args.output_dir, args.seed)
    print(
        f"Raw dataset: {summary['raw_shape'][0]} rows x {summary['raw_shape'][1]} columns"
    )
    print(f"Usable binary sentiment rows: {summary['usable_rows']}")
    print(f"Excluded rows: {summary['preprocessing']['excluded_rows']}")
    print(
        f"Duplicate review rows retained: {summary['preprocessing']['duplicate_review_rows']}"
    )
    print(f"Sentiment counts: {summary['class_distribution']}")
    for name, values in summary["splits"].items():
        print(
            f"{name}: {values['rows']} rows ({values['proportion']:.2%}), {values['class_distribution']}"
        )
    print(
        f"Duplicate leakage check: {'PASS' if summary['duplicate_leakage_free'] else 'FAIL'}"
    )
    print(f"Wrote UTF-8 CSV files to {summary['output_dir']}")


if __name__ == "__main__":
    main()

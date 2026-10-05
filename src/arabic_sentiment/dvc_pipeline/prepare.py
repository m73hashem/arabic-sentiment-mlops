"""Prepare deterministic DVC datasets using the established Phase 1 pipeline."""

import argparse
from pathlib import Path

from arabic_sentiment import __main__ as data_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--train-ratio", type=float, required=True)
    parser.add_argument("--validation-ratio", type=float, required=True)
    parser.add_argument("--test-ratio", type=float, required=True)
    args = parser.parse_args()
    summary = data_pipeline.run_pipeline(
        args.raw,
        args.output_dir,
        seed=args.seed,
        split_ratios=(args.train_ratio, args.validation_ratio, args.test_ratio),
    )
    print(
        "Prepared "
        f"{summary['usable_rows']} usable rows into {args.output_dir} "
        f"with seed {args.seed}; leakage-free={summary['duplicate_leakage_free']}"
    )


if __name__ == "__main__":
    main()

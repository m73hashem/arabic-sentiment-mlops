"""Command-line entry point for the Phase 2 AraBERT baseline."""

import argparse
import json

from .training import TrainingConfig, load_config, train_and_evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the AraBERT baseline")
    parser.add_argument("--config", default="configs/baseline.json")
    args = parser.parse_args()
    config = load_config(args.config)
    results = train_and_evaluate(config)
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

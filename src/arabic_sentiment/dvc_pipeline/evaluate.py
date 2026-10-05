"""Evaluate the lightweight DVC reference model on validation and test splits."""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def evaluate_split(model, path: Path) -> dict[str, float]:
    frame = pd.read_csv(path, encoding="utf-8", usecols=["review", "sentiment"])
    if frame.empty or frame[["review", "sentiment"]].isna().any().any():
        raise ValueError(f"Evaluation data must contain rows with review and sentiment: {path}")
    expected = frame["sentiment"].astype(str)
    predicted = model.predict(frame["review"].astype(str))
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "precision": float(precision_score(expected, predicted, pos_label="positive", zero_division=0)),
        "recall": float(recall_score(expected, predicted, pos_label="positive", zero_division=0)),
        "f1": float(f1_score(expected, predicted, pos_label="positive", zero_division=0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model = joblib.load(args.model)
    results = {
        "validation": evaluate_split(model, args.validation),
        "test": evaluate_split(model, args.test),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote validation/test metrics to {args.output}")


if __name__ == "__main__":
    main()

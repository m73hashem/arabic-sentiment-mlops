"""Train the CPU-feasible TF-IDF/logistic-regression DVC reference model."""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_model(
    *,
    model_type: str,
    max_features: int,
    ngram_min: int,
    ngram_max: int,
    c_value: float,
    max_iter: int,
    solver: str,
    seed: int,
) -> Pipeline:
    if model_type != "tfidf_logistic_regression":
        raise ValueError(f"Unsupported DVC model type: {model_type}")
    if ngram_min < 1 or ngram_max < ngram_min:
        raise ValueError("N-gram bounds must satisfy 1 <= ngram_min <= ngram_max")
    if max_features < 1 or c_value <= 0 or max_iter < 1:
        raise ValueError("max_features, C, and max_iter must be positive")
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=(ngram_min, ngram_max),
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    max_iter=max_iter,
                    random_state=seed,
                    solver=solver,
                ),
            ),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--model-type", required=True)
    parser.add_argument("--max-features", type=int, required=True)
    parser.add_argument("--ngram-min", type=int, required=True)
    parser.add_argument("--ngram-max", type=int, required=True)
    parser.add_argument("--c", type=float, required=True)
    parser.add_argument("--max-iter", type=int, required=True)
    parser.add_argument("--solver", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.train, encoding="utf-8", usecols=["review", "sentiment"])
    if frame.empty or frame[["review", "sentiment"]].isna().any().any():
        raise ValueError(
            "Training data must contain nonempty review and sentiment values"
        )
    model = build_model(
        model_type=args.model_type,
        max_features=args.max_features,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
        c_value=args.c,
        max_iter=args.max_iter,
        solver=args.solver,
        seed=args.seed,
    )
    model.fit(frame["review"].astype(str), frame["sentiment"].astype(str))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(f"Trained {args.model_type} on {len(frame)} rows; wrote {args.output}")


if __name__ == "__main__":
    main()

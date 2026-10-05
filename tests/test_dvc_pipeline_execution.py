import json
import sys

import joblib
import pandas as pd
import pytest

from arabic_sentiment import __main__ as data_pipeline
from arabic_sentiment.dvc_pipeline import evaluate, prepare, train


def _raw_reviews(count=40):
    ratings = [1, 5] * (count // 2)
    return pd.DataFrame(
        {
            "no": [str(index) for index in range(count)],
            "Hotel name": ["Hotel"] * count,
            "rating": [str(rating) for rating in ratings],
            "user type": ["Guest"] * count,
            "room type": ["Room"] * count,
            "nights": ["1"] * count,
            "review": [
                f"{'سيء' if rating == 1 else 'ممتاز'} تجربة {index}"
                for index, rating in enumerate(ratings)
            ],
        }
    )


def _labeled_reviews(label, count=12):
    token = "سيء" if label == "negative" else "ممتاز"
    return pd.DataFrame(
        {
            "review": [f"{token} تجربة مشتركة {index}" for index in range(count)],
            "sentiment": [label] * count,
        }
    )


def test_data_pipeline_cli_writes_deterministic_schema_and_summary(
    tmp_path, monkeypatch, capsys
):
    raw_path = tmp_path / "raw.tsv"
    output_dir = tmp_path / "processed"
    _raw_reviews().to_csv(
        raw_path, sep="\t", index=False, encoding="utf-16", lineterminator="\r\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["arabic-sentiment", "--raw", str(raw_path), "--output-dir", str(output_dir)],
    )

    data_pipeline.main()

    printed = capsys.readouterr().out
    assert "Duplicate leakage check: PASS" in printed
    frames = {
        name: pd.read_csv(output_dir / f"{name}.csv")
        for name in ("train", "validation", "test")
    }
    assert sum(map(len, frames.values())) == len(_raw_reviews())
    assert all(
        frame.columns.tolist() == ["review", "rating", "sentiment"]
        for frame in frames.values()
    )
    assert all(
        set(frame.sentiment) <= {"negative", "positive"} for frame in frames.values()
    )


def test_dvc_prepare_cli_reuses_project_split_pipeline(tmp_path, monkeypatch, capsys):
    raw_path = tmp_path / "raw.tsv"
    output_dir = tmp_path / "dvc-data"
    _raw_reviews().to_csv(
        raw_path, sep="\t", index=False, encoding="utf-16", lineterminator="\r\n"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare",
            "--raw",
            str(raw_path),
            "--output-dir",
            str(output_dir),
            "--seed",
            "42",
            "--train-ratio",
            "0.8",
            "--validation-ratio",
            "0.1",
            "--test-ratio",
            "0.1",
        ],
    )

    prepare.main()

    assert "leakage-free=True" in capsys.readouterr().out
    assert all(
        (output_dir / f"{name}.csv").is_file()
        for name in ("train", "validation", "test")
    )


def test_dvc_model_builder_validates_configuration_and_fits_classifier():
    model = train.build_model(
        model_type="tfidf_logistic_regression",
        max_features=100,
        ngram_min=1,
        ngram_max=2,
        c_value=1.0,
        max_iter=100,
        solver="liblinear",
        seed=42,
    )
    frame = pd.concat([_labeled_reviews("negative"), _labeled_reviews("positive")])
    model.fit(frame.review, frame.sentiment)
    predictions = model.predict(["سيء تجربة", "ممتاز تجربة"])
    assert predictions.tolist() == ["negative", "positive"]

    with pytest.raises(ValueError, match="Unsupported DVC model type"):
        train.build_model(
            model_type="arabert",
            max_features=10,
            ngram_min=1,
            ngram_max=1,
            c_value=1,
            max_iter=10,
            solver="liblinear",
            seed=42,
        )
    with pytest.raises(ValueError, match="N-gram bounds"):
        train.build_model(
            model_type="tfidf_logistic_regression",
            max_features=10,
            ngram_min=2,
            ngram_max=1,
            c_value=1,
            max_iter=10,
            solver="liblinear",
            seed=42,
        )
    with pytest.raises(ValueError, match="must be positive"):
        train.build_model(
            model_type="tfidf_logistic_regression",
            max_features=0,
            ngram_min=1,
            ngram_max=1,
            c_value=1,
            max_iter=10,
            solver="liblinear",
            seed=42,
        )


def test_dvc_train_and_evaluate_commands_create_real_model_and_metrics(
    tmp_path, monkeypatch, capsys
):
    train_path = tmp_path / "train.csv"
    validation_path = tmp_path / "validation.csv"
    test_path = tmp_path / "test.csv"
    model_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    pd.concat([_labeled_reviews("negative"), _labeled_reviews("positive")]).to_csv(
        train_path, index=False, encoding="utf-8"
    )
    _labeled_reviews("negative", 4).to_csv(
        validation_path, index=False, encoding="utf-8"
    )
    _labeled_reviews("positive", 4).to_csv(test_path, index=False, encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--train",
            str(train_path),
            "--output",
            str(model_path),
            "--seed",
            "42",
            "--model-type",
            "tfidf_logistic_regression",
            "--max-features",
            "100",
            "--ngram-min",
            "1",
            "--ngram-max",
            "1",
            "--c",
            "1.0",
            "--max-iter",
            "100",
            "--solver",
            "liblinear",
        ],
    )
    train.main()
    assert model_path.is_file()
    model = joblib.load(model_path)
    assert model.predict(["سيء تجربة"])[0] == "negative"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate",
            "--model",
            str(model_path),
            "--validation",
            str(validation_path),
            "--test",
            str(test_path),
            "--output",
            str(metrics_path),
        ],
    )
    evaluate.main()
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(metrics) == {"validation", "test"}
    assert all(
        set(split) == {"accuracy", "precision", "recall", "f1"}
        for split in metrics.values()
    )
    assert all(0 <= split["accuracy"] <= 1 for split in metrics.values())
    assert "Wrote validation/test metrics" in capsys.readouterr().out


def test_dvc_evaluation_rejects_empty_or_incomplete_split(tmp_path):
    model = train.build_model(
        model_type="tfidf_logistic_regression",
        max_features=20,
        ngram_min=1,
        ngram_max=1,
        c_value=1,
        max_iter=20,
        solver="liblinear",
        seed=42,
    )
    empty = tmp_path / "empty.csv"
    pd.DataFrame(columns=["review", "sentiment"]).to_csv(empty, index=False)
    with pytest.raises(ValueError, match="Evaluation data must contain rows"):
        evaluate.evaluate_split(model, empty)

    missing_column = tmp_path / "missing.csv"
    pd.DataFrame({"review": ["سيء"]}).to_csv(missing_column, index=False)
    with pytest.raises(ValueError, match="Usecols do not match columns"):
        evaluate.evaluate_split(model, missing_column)


def test_dvc_training_rejects_empty_or_null_training_rows(tmp_path, monkeypatch):
    train_path = tmp_path / "invalid.csv"
    output_path = tmp_path / "model.joblib"
    for frame in (
        pd.DataFrame(columns=["review", "sentiment"]),
        pd.DataFrame({"review": [None], "sentiment": ["negative"]}),
    ):
        frame.to_csv(train_path, index=False, encoding="utf-8")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "train",
                "--train",
                str(train_path),
                "--output",
                str(output_path),
                "--seed",
                "42",
                "--model-type",
                "tfidf_logistic_regression",
                "--max-features",
                "100",
                "--ngram-min",
                "1",
                "--ngram-max",
                "1",
                "--c",
                "1.0",
                "--max-iter",
                "100",
                "--solver",
                "liblinear",
            ],
        )
        with pytest.raises(ValueError, match="must contain nonempty review"):
            train.main()
    assert not output_path.exists()

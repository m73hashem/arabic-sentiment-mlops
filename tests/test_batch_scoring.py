import csv
import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "score_reviews.py"
_SPEC = importlib.util.spec_from_file_location("score_reviews", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_SCORING = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SCORING)


def _write_reviews(path: Path, rows: int = 1_000) -> Path:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source_row_id", "review"))
        writer.writeheader()
        writer.writerows(
            {"source_row_id": f"review-{index}", "review": f"رأي عربي {index}"}
            for index in range(rows)
        )
    return path


def test_batch_scoring_validates_schema_minimum_size_and_nonblank_reviews(tmp_path):
    no_review_column = tmp_path / "no-review.csv"
    no_review_column.write_text("id\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="review column"):
        _SCORING.score_reviews(no_review_column, tmp_path / "out.csv")

    too_small = _write_reviews(tmp_path / "small.csv", rows=999)
    with pytest.raises(ValueError, match="at least 1000"):
        _SCORING.score_reviews(too_small, tmp_path / "out.csv")

    blank_review = tmp_path / "blank.csv"
    blank_review.write_text("source_row_id,review\nrow-1,   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="blank review.*row-1"):
        _SCORING.score_reviews(blank_review, tmp_path / "out.csv")


def test_batch_scoring_writes_ordered_predictions_in_batches(tmp_path):
    source = _write_reviews(tmp_path / "reviews.csv", rows=1_003)
    destination = tmp_path / "data/scoring/output/predictions.csv"
    calls = []

    class Predictor:
        model_version = "test-model-version"

        def predict_many(self, reviews):
            calls.append(list(reviews))
            return [
                {
                    "label": "positive" if index % 2 else "negative",
                    "confidence": 0.75,
                }
                for index, _ in enumerate(reviews)
            ]

    assert (
        _SCORING.score_reviews(
            source, destination, batch_size=400, service_factory=Predictor
        )
        == 1_003
    )
    assert [len(batch) for batch in calls] == [400, 400, 203]

    with destination.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
    assert reader.fieldnames == [
        "source_row_id",
        "label",
        "confidence_score",
        "model_version",
    ]
    assert len(rows) == 1_003
    assert [rows[index]["source_row_id"] for index in (0, 400, 1_002)] == [
        "review-0",
        "review-400",
        "review-1002",
    ]
    assert rows[0] == {
        "source_row_id": "review-0",
        "label": "negative",
        "confidence_score": "0.75",
        "model_version": "test-model-version",
    }


def test_batch_scoring_reports_failed_batch_and_does_not_publish_partial_output(
    tmp_path,
):
    source = _write_reviews(tmp_path / "reviews.csv", rows=1_000)
    destination = tmp_path / "predictions.csv"

    class FailingPredictor:
        model_version = "test-model-version"

        def predict_many(self, reviews):
            raise OSError("predictor unavailable")

    with pytest.raises(RuntimeError, match="review-0.*review-9.*predictor unavailable"):
        _SCORING.score_reviews(
            source, destination, batch_size=10, service_factory=FailingPredictor
        )
    assert not destination.exists()
    assert list(tmp_path.glob(".predictions.csv.*.tmp")) == []


def test_batch_scoring_rejects_same_input_and_output_path(tmp_path):
    source = _write_reviews(tmp_path / "reviews.csv")
    with pytest.raises(ValueError, match="must differ"):
        _SCORING.score_reviews(source, source)

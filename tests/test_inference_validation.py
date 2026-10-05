import json
from pathlib import Path

import pytest

from arabic_sentiment import inference
from arabic_sentiment.inference import SentimentPredictor, predict_review


def test_predict_many_validates_batch_size_and_each_review_without_loading_model():
    predictor = SentimentPredictor.__new__(SentimentPredictor)
    with pytest.raises(ValueError, match="batch_size must be positive"):
        predictor.predict_many([], batch_size=0)
    with pytest.raises(TypeError, match="review must be a string"):
        predictor.predict_many([None])
    with pytest.raises(ValueError, match="non-whitespace text"):
        predictor.predict_many([" \n "])


@pytest.mark.parametrize(
    ("training_config", "explicit_max_length", "expected"),
    [
        ({"max_sequence_length": 128}, None, 128),
        ({"max_sequence_length": 128}, 64, 64),
        (None, None, 96),
    ],
)
def test_predictor_loads_local_artifact_configuration_without_downloading(
    tmp_path, monkeypatch, training_config, explicit_max_length, expected
):
    calls = []

    class FakeTokenizer:
        pass

    class FakeModel:
        def to(self, device):
            calls.append(("device", str(device)))
            return self

        def eval(self):
            calls.append(("eval",))

    monkeypatch.setattr(
        inference.AutoTokenizer,
        "from_pretrained",
        lambda path, **kwargs: calls.append(("tokenizer", path, kwargs)) or FakeTokenizer(),
    )
    monkeypatch.setattr(
        inference.AutoModelForSequenceClassification,
        "from_pretrained",
        lambda path, **kwargs: calls.append(("model", path, kwargs)) or FakeModel(),
    )
    if training_config is not None:
        (tmp_path / "training_config.json").write_text(
            json.dumps(training_config), encoding="utf-8"
        )

    predictor = SentimentPredictor(tmp_path, device="cpu", max_sequence_length=explicit_max_length)

    assert predictor.max_sequence_length == expected
    assert predictor.device.type == "cpu"
    assert ("tokenizer", tmp_path, {"local_files_only": True}) in calls
    assert ("model", tmp_path, {"local_files_only": True}) in calls
    assert ("eval",) in calls


@pytest.mark.parametrize(
    ("official_exists", "expected"),
    [(True, "models/full-gpu-arabert-inference"), (False, "models/baseline-arabert")],
)
def test_predict_review_prefers_official_artifact_then_uses_legacy_fallback(
    tmp_path, monkeypatch, official_exists, expected
):
    monkeypatch.chdir(tmp_path)
    official = Path("models/full-gpu-arabert-inference")
    if official_exists:
        official.mkdir(parents=True)
    calls = []

    class Predictor:
        def __init__(self, artifact_dir):
            calls.append(Path(artifact_dir))

        def predict(self, review):
            return {"label": "positive", "confidence": 0.8}

    monkeypatch.setattr("arabic_sentiment.inference.SentimentPredictor", Predictor)
    assert predict_review("جيد") == {"label": "positive", "confidence": 0.8}
    assert calls == [Path(expected)]

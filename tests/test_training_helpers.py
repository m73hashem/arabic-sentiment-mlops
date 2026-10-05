import json

import pandas as pd
import pytest
import torch

from arabic_sentiment import train as training_cli
from arabic_sentiment.modeling import MODEL_CHECKPOINT
from arabic_sentiment.training import (
    ReviewDataset,
    _collate,
    _metrics,
    _sample_training_rows,
    load_config,
    seed_everything,
)


def test_review_dataset_rejects_missing_fields_missing_values_and_unknown_labels():
    with pytest.raises(ValueError, match="missing required columns"):
        ReviewDataset(pd.DataFrame({"review": ["سيء"]}))
    with pytest.raises(ValueError, match="cannot contain missing values"):
        ReviewDataset(pd.DataFrame({"review": [None], "sentiment": ["negative"]}))
    with pytest.raises(ValueError, match="Unexpected sentiment labels"):
        ReviewDataset(pd.DataFrame({"review": ["محايد"], "sentiment": ["neutral"]}))


def test_training_config_rejects_unrelated_checkpoint_and_invalid_dimensions(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"model_checkpoint": "unrelated/model"}), encoding="utf-8")
    with pytest.raises(ValueError, match="fixed to"):
        load_config(config_path)

    config_path.write_text(
        json.dumps({"model_checkpoint": MODEL_CHECKPOINT, "epochs": 0}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="must be positive"):
        load_config(config_path)


def test_training_dataset_collation_and_metrics_preserve_label_contract():
    dataset = ReviewDataset(
        pd.DataFrame({"review": ["سيء", "ممتاز"], "sentiment": ["negative", "positive"]})
    )
    assert len(dataset) == 2
    assert dataset[1] == ("ممتاز", 1)

    class Tokenizer:
        def __call__(self, texts, **kwargs):
            return {"input_ids": torch.tensor([[len(text)] for text in texts])}

    batch = _collate(Tokenizer(), max_length=16)([dataset[0], dataset[1]])
    assert batch["input_ids"].tolist() == [[3], [5]]
    assert batch["labels"].tolist() == [0, 1]
    assert _metrics([0, 1, 1], [0, 0, 1], loss_sum=0.6) == {
        "loss": pytest.approx(0.2),
        "accuracy": pytest.approx(2 / 3),
        "precision": pytest.approx(1.0),
        "recall": pytest.approx(0.5),
        "f1": pytest.approx(2 / 3),
    }
    assert _metrics([], [], 0.0)["accuracy"] == 0.0


def test_seeded_sampling_rejects_limits_that_cannot_represent_each_class():
    frame = pd.DataFrame({"review": ["a", "b"], "sentiment": ["negative", "positive"]})
    with pytest.raises(ValueError, match="at least one row per class"):
        _sample_training_rows(frame, 1, seed=42)
    seed_everything(42)
    first = torch.rand(3)
    seed_everything(42)
    assert torch.equal(first, torch.rand(3))


def test_training_cli_passes_config_to_training_and_prints_result(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "baseline.json"
    config_path.write_text("{}", encoding="utf-8")
    captured = {}

    def fake_train(config):
        captured["config"] = config
        return {"device": "cpu", "test_evaluations": 0}

    monkeypatch.setattr("arabic_sentiment.train.load_config", lambda path: {"path": path})
    monkeypatch.setattr("arabic_sentiment.train.train_and_evaluate", fake_train)
    monkeypatch.setattr("sys.argv", ["train", "--config", str(config_path)])

    training_cli.main()

    assert captured["config"] == {"path": str(config_path)}
    assert json.loads(capsys.readouterr().out) == {"device": "cpu", "test_evaluations": 0}

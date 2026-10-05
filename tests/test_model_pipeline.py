import json
import os
import unittest
from pathlib import Path

import pandas as pd
import pytest

from arabic_sentiment.inference import SentimentPredictor
from arabic_sentiment.modeling import ID_TO_LABEL, LABEL_TO_ID, MODEL_CHECKPOINT
from arabic_sentiment.training import ReviewDataset, _sample_training_rows, load_config

ARTIFACT_DIR = Path(
    os.environ.get("ARABIC_SENTIMENT_TEST_ARTIFACT_DIR", "models/baseline-arabert")
)
ARTIFACT_AVAILABLE = all(
    (ARTIFACT_DIR / filename).is_file()
    for filename in ("model.safetensors", "training_config.json")
)


class ModelPipelineTests(unittest.TestCase):
    def test_label_mapping_is_deterministic(self):
        self.assertEqual(LABEL_TO_ID, {"negative": 0, "positive": 1})
        self.assertEqual(ID_TO_LABEL, {0: "negative", 1: "positive"})
        dataset = ReviewDataset(
            pd.DataFrame(
                {"review": ["سيئ", "جيد"], "sentiment": ["negative", "positive"]}
            )
        )
        self.assertEqual(dataset.labels, [0, 1])

    def test_config_records_the_official_arabert_checkpoint(self):
        config = load_config("configs/baseline.json")
        self.assertEqual(config.model_checkpoint, MODEL_CHECKPOINT)
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.best_model_metric, "validation_f1")

    def test_training_sample_is_seeded_balanced_and_stays_within_train_rows(self):
        frame = pd.DataFrame(
            {
                "review": [f"review-{i}" for i in range(100)],
                "sentiment": ["negative"] * 60 + ["positive"] * 40,
            }
        )
        first = _sample_training_rows(frame, 20, seed=42)
        second = _sample_training_rows(frame, 20, seed=42)
        self.assertEqual(first.to_dict("records"), second.to_dict("records"))
        self.assertEqual(
            first.sentiment.value_counts().to_dict(), {"negative": 12, "positive": 8}
        )
        self.assertTrue(set(first.review).issubset(frame.review))


@pytest.mark.skipif(
    not ARTIFACT_AVAILABLE,
    reason="local AraBERT artifact unavailable; saved-artifact tests require model.safetensors and training_config.json",
)
class SavedArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ARTIFACT_DIR / "training_config.json").open(
            encoding="utf-8"
        ) as config_file:
            cls.config = json.load(config_file)
        cls.predictor = SentimentPredictor(ARTIFACT_DIR)

    def test_saved_model_and_tokenizer_load_locally(self):
        self.assertEqual(self.predictor.model.config.num_labels, 2)
        self.assertEqual(self.predictor.tokenizer.vocab_size > 0, True)
        self.assertEqual(self.config["model_checkpoint"], MODEL_CHECKPOINT)

    def test_prediction_label_confidence_and_repeatability(self):
        review = "الخدمة ممتازة والغرفة نظيفة ومريحة"
        first = self.predictor.predict(review)
        second = self.predictor.predict(review)
        self.assertIn(first["label"], {"negative", "positive"})
        self.assertGreaterEqual(first["confidence"], 0.0)
        self.assertLessEqual(first["confidence"], 1.0)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()

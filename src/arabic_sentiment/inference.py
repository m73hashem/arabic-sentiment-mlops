"""Local model-level prediction interface for a saved sentiment artifact."""

import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .modeling import ID_TO_LABEL


class SentimentPredictor:
    """Load a saved model once and predict labels and confidence locally."""

    def __init__(
        self,
        artifact_dir: str | Path = "models/baseline-arabert",
        device: str | None = None,
        max_sequence_length: int | None = None,
    ):
        self.artifact_dir = Path(artifact_dir)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.artifact_dir, local_files_only=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.artifact_dir, local_files_only=True
        ).to(self.device)
        self.model.eval()
        training_config = self.artifact_dir / "training_config.json"
        if max_sequence_length is not None:
            self.max_sequence_length = int(max_sequence_length)
        elif training_config.is_file():
            with training_config.open(encoding="utf-8") as config_file:
                self.max_sequence_length = int(
                    json.load(config_file)["max_sequence_length"]
                )
        else:
            # Official inference-only exports omit training state; the project baseline
            # used 96 tokens and keeps that setting as its local inference default.
            self.max_sequence_length = 96

    @torch.inference_mode()
    def predict_many(
        self,
        reviews: list[str],
        batch_size: int = 8,
        max_sequence_length: int | None = None,
    ) -> list[dict[str, str | float]]:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        for review in reviews:
            if not isinstance(review, str):
                raise TypeError("review must be a string")
            if not review.strip():
                raise ValueError("review must contain non-whitespace text")

        predictions = []
        max_length = int(max_sequence_length or self.max_sequence_length)
        for offset in range(0, len(reviews), batch_size):
            batch = reviews[offset : offset + batch_size]
            inputs = self.tokenizer(
                batch,
                truncation=True,
                max_length=max_length,
                padding=True,
                return_tensors="pt",
            )
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            probabilities = torch.softmax(self.model(**inputs).logits, dim=-1)
            for row in probabilities:
                predicted_id = int(row.argmax().item())
                predictions.append(
                    {
                        "label": ID_TO_LABEL[predicted_id],
                        "confidence": float(row[predicted_id].item()),
                    }
                )
        return predictions

    @torch.inference_mode()
    def predict(self, review: str) -> dict[str, str | float]:
        return self.predict_many([review])[0]


def predict_review(
    review: str, artifact_dir: str | Path | None = None
) -> dict[str, str | float]:
    """Convenience helper that loads the local artifact and predicts once."""
    if artifact_dir is None:
        official_artifact = Path("models/full-gpu-arabert-inference")
        artifact_dir = (
            official_artifact
            if official_artifact.is_dir()
            else Path("models/baseline-arabert")
        )
    return SentimentPredictor(artifact_dir).predict(review)

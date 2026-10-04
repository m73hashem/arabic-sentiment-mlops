"""Local model-level prediction interface for a saved sentiment artifact."""

from pathlib import Path
import json

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .modeling import ID_TO_LABEL


class SentimentPredictor:
    """Load a saved model once and predict labels and confidence locally."""

    def __init__(self, artifact_dir: str | Path = "models/baseline-arabert", device: str | None = None):
        self.artifact_dir = Path(artifact_dir)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained(self.artifact_dir, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.artifact_dir, local_files_only=True
        ).to(self.device)
        self.model.eval()
        with (self.artifact_dir / "training_config.json").open(encoding="utf-8") as config_file:
            self.max_sequence_length = int(json.load(config_file)["max_sequence_length"])

    @torch.inference_mode()
    def predict(self, review: str) -> dict[str, str | float]:
        if not isinstance(review, str):
            raise TypeError("review must be a string")
        if not review.strip():
            raise ValueError("review must contain non-whitespace text")

        inputs = self.tokenizer(
            review,
            truncation=True,
            max_length=self.max_sequence_length,
            padding=True,
            return_tensors="pt",
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        logits = self.model(**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_id = int(probabilities.argmax().item())
        return {
            "label": ID_TO_LABEL[predicted_id],
            "confidence": float(probabilities[predicted_id].item()),
        }


def predict_review(
    review: str, artifact_dir: str | Path = "models/baseline-arabert"
) -> dict[str, str | float]:
    """Convenience helper that loads the local artifact and predicts once."""
    return SentimentPredictor(artifact_dir).predict(review)

"""Adapter between the HTTP API and the local model-level predictor."""

import json
from pathlib import Path
from typing import Any

from .inference import SentimentPredictor


class LocalAraBERTService:
    """Load and serve the saved local AraBERT artifact on first prediction."""

    def __init__(self, artifact_dir: str | Path = "models/baseline-arabert"):
        self.artifact_dir = Path(artifact_dir)
        self._predictor: SentimentPredictor | None = None
        self._model_version = self._read_model_version()

    def _read_model_version(self) -> str:
        config_path = self.artifact_dir / "training_config.json"
        if not config_path.is_file():
            raise FileNotFoundError(f"Model artifact metadata not found: {config_path}")
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
        checkpoint = str(config["model_checkpoint"])
        seed = int(config["seed"])
        return f"{checkpoint}-finetuned-seed-{seed}"

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(self, review: str) -> dict[str, Any]:
        if self._predictor is None:
            self._predictor = SentimentPredictor(self.artifact_dir, device="cpu")
        return self._predictor.predict(review)

"""Adapter between the HTTP API and the local model-level predictor."""

import json
import os
from pathlib import Path
from typing import Any

from .inference import SentimentPredictor


class LocalAraBERTService:
    """Load and serve the saved local AraBERT artifact on first prediction."""

    def __init__(self, artifact_dir: str | Path | None = None):
        default_dir = (
            "models/full-gpu-arabert-inference"
            if Path("models/full-gpu-arabert-inference").is_dir()
            else "models/baseline-arabert"
        )
        configured_dir = artifact_dir or os.environ.get("SENTIMENT_MODEL_DIR", default_dir)
        self.artifact_dir = Path(configured_dir)
        self._predictor: SentimentPredictor | None = None
        self._model_version = self._read_model_version()

    def _read_model_version(self) -> str:
        config_path = self.artifact_dir / "training_config.json"
        if not config_path.is_file():
            if (self.artifact_dir / "config.json").is_file():
                from .modeling import MODEL_CHECKPOINT

                return f"{MODEL_CHECKPOINT}-official-full-gpu-seed-42"
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


class MlflowModelService:
    """Serve an MLflow model URI without coupling model selection to FastAPI."""

    def __init__(self, model_uri: str):
        if not model_uri:
            raise ValueError("model_uri must not be empty")
        self.model_uri = model_uri
        self._model = None
        self._model_version = self._resolve_version()

    def _resolve_version(self) -> str:
        if self.model_uri.startswith("models:/") and "@" in self.model_uri:
            from .mlflow_tracking import registered_alias_version

            alias = self.model_uri.rsplit("@", maxsplit=1)[1]
            return registered_alias_version(alias)
        if self.model_uri.startswith("models:/"):
            return self.model_uri.rsplit("/", maxsplit=1)[1]
        return self.model_uri

    @property
    def model_version(self) -> str:
        return self._model_version

    def predict(self, review: str) -> dict[str, Any]:
        if self._model is None:
            from .mlflow_tracking import load_selected_pyfunc

            self._model = load_selected_pyfunc(self.model_uri)
        import pandas as pd

        result = self._model.predict(pd.DataFrame({"review": [review]}))
        row = result.iloc[0]
        return {"label": str(row["label"]), "confidence": float(row["confidence"])}


def create_prediction_service(model_uri: str | None = None):
    """Choose a registry/URI-backed provider, falling back to the local artifact."""
    selected_uri = model_uri or os.environ.get("SENTIMENT_MODEL_URI")
    if selected_uri:
        return MlflowModelService(selected_uri)
    return LocalAraBERTService()

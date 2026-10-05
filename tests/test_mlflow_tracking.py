import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlflow
from mlflow import MlflowClient

from arabic_sentiment.mlflow_tracking import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TRACKING_DB,
    EXPERIMENT_NAME,
    OFFICIAL_BASELINE_METRICS,
    REGISTERED_MODEL_NAME,
    configure_tracking,
    verify_project_history,
)
from arabic_sentiment.serving import LocalAraBERTService, MlflowModelService, create_prediction_service


class MlflowTrackingTests(unittest.TestCase):
    def test_local_tracking_logs_parameters_metrics_and_artifact(self):
        self.addCleanup(
            configure_tracking,
            f"sqlite:///{DEFAULT_TRACKING_DB.as_posix()}",
            DEFAULT_ARTIFACT_ROOT,
        )
        with tempfile.TemporaryDirectory(prefix="arabic-sentiment-test-mlflow-") as temp_dir:
            root = Path(temp_dir)
            tracking_uri = f"sqlite:///{(root / 'tracking.db').as_posix()}"
            configure_tracking(tracking_uri, artifact_root=root / "artifacts")
            with mlflow.start_run(run_name="tracking-smoke-test") as run:
                mlflow.log_params({"model_checkpoint": "unit-test", "seed": 42})
                mlflow.log_metrics(
                    {
                        "validation_loss": 0.25,
                        "validation_accuracy": 0.9,
                        "validation_precision": 0.9,
                        "validation_recall": 0.9,
                        "validation_f1": 0.9,
                    }
                )
                mlflow.log_text("actual test artifact", "metadata/result.txt")

            experiment = MlflowClient(tracking_uri).get_experiment_by_name(EXPERIMENT_NAME)
            logged = MlflowClient(tracking_uri).get_run(run.info.run_id)
            self.assertEqual(logged.info.status, "FINISHED")
            self.assertEqual(logged.data.params["model_checkpoint"], "unit-test")
            self.assertEqual(logged.data.params["seed"], "42")
            self.assertEqual(logged.data.metrics["validation_f1"], 0.9)
            self.assertTrue(
                (root / "artifacts" / EXPERIMENT_NAME / run.info.run_id / "artifacts" / "metadata" / "result.txt").is_file()
            )
            self.assertEqual(experiment.name, "arabic-sentiment")

    def test_official_baseline_contract_and_real_history(self):
        self.assertEqual(
            set(OFFICIAL_BASELINE_METRICS),
            {
                "validation_loss",
                "validation_accuracy",
                "validation_precision",
                "validation_recall",
                "validation_f1",
                "test_loss",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
            },
        )
        if not Path("mlflow.db").is_file():
            self.skipTest("local project MLflow history is generated locally, not checked into Git")

        audit = verify_project_history(minimum_runs=5)
        self.assertGreaterEqual(audit["valid_run_count"], 5)
        self.assertEqual(audit["candidate_alias_version"], audit["official_model_version"])
        self.assertTrue(audit["production_alias_set"])
        self.assertEqual(audit["production_alias_version"], audit["official_model_version"])
        configure_tracking()
        client = MlflowClient()
        model_version = client.get_model_version(
            REGISTERED_MODEL_NAME, audit["official_model_version"]
        )
        self.assertEqual(model_version.run_id, audit["official_run_id"])
        gate = client.get_run(audit["quality_gate_run_id"])
        self.assertEqual(gate.info.status, "FINISHED")
        self.assertEqual(gate.data.tags["promotion_decision"], "passed")
        self.assertTrue(gate.data.metrics["validation_f1_margin"] > 0)
        self.assertTrue(gate.data.metrics["test_f1_margin"] > 0)
        official = client.get_run(audit["official_run_id"])
        self.assertEqual(official.info.status, "FINISHED")
        self.assertEqual(official.data.tags["source"], "colab_gpu")
        self.assertEqual(official.data.tags["training_performed_by_mlflow"], "false")
        for name in (
            "model_name",
            "model_checkpoint",
            "seed",
            "epochs",
            "max_length",
            "batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "weight_decay",
            "optimizer",
            "fp16",
            "train_rows",
            "validation_rows",
            "test_rows",
            "training_device",
            "training_environment",
            "source_artifact_path",
        ):
            self.assertIn(name, official.data.params)
        self.assertEqual(
            set(OFFICIAL_BASELINE_METRICS), set(official.data.metrics) & set(OFFICIAL_BASELINE_METRICS)
        )
        self.assertTrue(client.list_artifacts(audit["official_run_id"]))

        evaluation_runs = [
            client.get_run(run_id) for run_id in audit["valid_run_ids"]
            if client.get_run(run_id).data.tags.get("experiment_type")
            == "deterministic_local_evaluation"
        ]
        self.assertGreaterEqual(len(evaluation_runs), 2)
        for run in evaluation_runs:
            self.assertEqual(run.data.params["training_performed"], "False")
            self.assertEqual(len(run.data.tags["sample_sha256"]), 64)

    def test_serving_uses_configured_mlflow_uri_or_local_fallback(self):
        with patch.dict(os.environ, {"SENTIMENT_MODEL_URI": "models:/arabic-sentiment@candidate"}):
            with patch.object(MlflowModelService, "_resolve_version", return_value="7"):
                service = create_prediction_service()
            self.assertIsInstance(service, MlflowModelService)
            self.assertEqual(service.model_uri, "models:/arabic-sentiment@candidate")
            self.assertEqual(service.model_version, "7")

        with patch.dict(os.environ, {}, clear=True):
            service = create_prediction_service()
            self.assertIsInstance(service, LocalAraBERTService)
            if Path("models/full-gpu-arabert-inference").is_dir():
                self.assertEqual(
                    service.artifact_dir, Path("models/full-gpu-arabert-inference")
                )

        with patch.dict(os.environ, {"SENTIMENT_MODEL_URI": "models:/arabic-sentiment/4"}):
            service = create_prediction_service()
            self.assertIsInstance(service, MlflowModelService)
            self.assertEqual(service.model_version, "4")


if __name__ == "__main__":
    unittest.main()

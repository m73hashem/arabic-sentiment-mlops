import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import mlflow
from mlflow import MlflowClient

import arabic_sentiment.mlflow_tracking as tracking
from arabic_sentiment.mlflow_tracking import (
    EXPERIMENT_NAME,
    OFFICIAL_BASELINE_METRICS,
    REGISTERED_MODEL_NAME,
    configure_tracking,
    verify_project_history,
)
from arabic_sentiment.serving import (
    MlflowModelService,
    create_prediction_service,
)


class MlflowTrackingTests(unittest.TestCase):
    def test_local_tracking_logs_parameters_metrics_and_artifact(self):
        with tempfile.TemporaryDirectory(
            prefix="arabic-sentiment-test-mlflow-"
        ) as temp_dir:
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

            experiment = MlflowClient(tracking_uri).get_experiment_by_name(
                EXPERIMENT_NAME
            )
            logged = MlflowClient(tracking_uri).get_run(run.info.run_id)
            self.assertEqual(logged.info.status, "FINISHED")
            self.assertEqual(logged.data.params["model_checkpoint"], "unit-test")
            self.assertEqual(logged.data.params["seed"], "42")
            self.assertEqual(logged.data.metrics["validation_f1"], 0.9)
            self.assertTrue(
                (
                    root
                    / "artifacts"
                    / EXPERIMENT_NAME
                    / run.info.run_id
                    / "artifacts"
                    / "metadata"
                    / "result.txt"
                ).is_file()
            )
            self.assertEqual(experiment.name, "arabic-sentiment")

    def test_official_baseline_contract_and_history_audit(self):
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
        with tempfile.TemporaryDirectory(
            prefix="arabic-sentiment-history-"
        ) as directory:
            root = Path(directory)
            tracking_uri = f"sqlite:///{(root / 'tracking.db').as_posix()}"
            artifact_root = root / "artifacts"
            with (
                patch.dict(os.environ, {"MLFLOW_TRACKING_URI": tracking_uri}),
                patch.object(tracking, "DEFAULT_ARTIFACT_ROOT", artifact_root),
            ):
                self._populate_valid_project_history()
                audit = verify_project_history(minimum_runs=5)
                self._assert_history_audit(audit, tracking_uri)

    def _populate_valid_project_history(self):
        """Write coherent MLflow metadata in an isolated temporary tracking store."""
        configure_tracking()
        client = MlflowClient()
        official_params = {
            "model_name": "AraBERTv0.2-base",
            "model_checkpoint": "aubmindlab/bert-base-arabertv02",
            "seed": 42,
            "epochs": 1,
            "max_length": 96,
            "batch_size": 16,
            "gradient_accumulation_steps": 1,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "optimizer": "AdamW",
            "fp16": True,
            "train_rows": 84558,
            "validation_rows": 10570,
            "test_rows": 10570,
            "training_device": "Tesla T4",
            "training_environment": "Google Colab",
            "source_artifact_path": "test-fixture-existing-artifact-reference",
        }

        def log_run(name, params, metrics, tags):
            with mlflow.start_run(run_name=name) as run:
                mlflow.log_params(params)
                mlflow.log_metrics(metrics)
                mlflow.set_tags(tags)
                mlflow.log_text("isolated test evidence", f"metadata/{name}.txt")
            return run

        official = log_run(
            "official-baseline-fixture",
            official_params,
            OFFICIAL_BASELINE_METRICS,
            {
                "run_kind": "official_gpu_baseline",
                "model_version": "1",
                "source": "colab_gpu",
                "training_performed_by_mlflow": "false",
            },
        )
        for index in range(2):
            log_run(
                f"evaluation-fixture-{index}",
                {"training_performed": False, "sample_count": 8},
                {"validation_f1": 0.9},
                {
                    "experiment_type": "deterministic_local_evaluation",
                    "sample_sha256": f"{index:064x}",
                },
            )
        gate = log_run(
            "quality-gate-fixture",
            {"model_version": "1"},
            {"validation_f1_margin": 0.001, "test_f1_margin": 0.001},
            {
                "run_kind": "production_promotion_quality_gate",
                "promotion_decision": "passed",
                "model_version": "1",
            },
        )
        log_run(
            "additional-evaluation-fixture",
            {"training_performed": False, "sample_count": 4},
            {"accuracy": 0.75},
            {"run_kind": "test_evaluation"},
        )

        client.create_registered_model(REGISTERED_MODEL_NAME)
        client.create_model_version(
            REGISTERED_MODEL_NAME,
            source=official.info.artifact_uri,
            run_id=official.info.run_id,
        )
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "candidate", "1")
        client.set_registered_model_alias(REGISTERED_MODEL_NAME, "production", "1")
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME, "1", "promotion_quality_gate", "passed"
        )
        client.set_model_version_tag(
            REGISTERED_MODEL_NAME,
            "1",
            "promotion_gate_run_id",
            gate.info.run_id,
        )

    def _assert_history_audit(self, audit, tracking_uri):
        self.assertGreaterEqual(audit["valid_run_count"], 5)
        self.assertEqual(
            audit["candidate_alias_version"], audit["official_model_version"]
        )
        self.assertTrue(audit["production_alias_set"])
        self.assertEqual(
            audit["production_alias_version"], audit["official_model_version"]
        )
        client = MlflowClient(tracking_uri)
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
            set(OFFICIAL_BASELINE_METRICS),
            set(official.data.metrics) & set(OFFICIAL_BASELINE_METRICS),
        )
        self.assertTrue(client.list_artifacts(audit["official_run_id"]))

        evaluation_runs = [
            client.get_run(run_id)
            for run_id in audit["valid_run_ids"]
            if client.get_run(run_id).data.tags.get("experiment_type")
            == "deterministic_local_evaluation"
        ]
        self.assertGreaterEqual(len(evaluation_runs), 2)
        for run in evaluation_runs:
            self.assertEqual(run.data.params["training_performed"], "False")
            self.assertEqual(len(run.data.tags["sample_sha256"]), 64)

    def test_serving_uses_configured_mlflow_uri_or_local_fallback(self):
        with patch.dict(
            os.environ, {"SENTIMENT_MODEL_URI": "models:/arabic-sentiment@candidate"}
        ):
            with patch.object(MlflowModelService, "_resolve_version", return_value="7"):
                service = create_prediction_service()
            self.assertIsInstance(service, MlflowModelService)
            self.assertEqual(service.model_uri, "models:/arabic-sentiment@candidate")
            self.assertEqual(service.model_version, "7")

        with patch.dict(os.environ, {}, clear=True):
            local_service = object()
            with patch(
                "arabic_sentiment.serving.LocalAraBERTService",
                return_value=local_service,
            ) as local_factory:
                service = create_prediction_service()
            self.assertIs(service, local_service)
            local_factory.assert_called_once_with()

        with patch.dict(
            os.environ, {"SENTIMENT_MODEL_URI": "models:/arabic-sentiment/4"}
        ):
            service = create_prediction_service()
            self.assertIsInstance(service, MlflowModelService)
            self.assertEqual(service.model_version, "4")


if __name__ == "__main__":
    unittest.main()

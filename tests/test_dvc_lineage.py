import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mlflow import MlflowClient

from arabic_sentiment.dvc_lineage import (
    DVC_DATASET_HASH_TAG,
    current_git_revision,
    get_dvc_dataset_hash,
    mlflow_lineage_tags,
)
from arabic_sentiment.mlflow_tracking import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_TRACKING_DB,
    configure_tracking,
    start_project_run,
)


class DvcLineageTests(unittest.TestCase):
    HASH = "0123456789abcdef0123456789abcdef"

    def _write_pointer(self, path: Path, content: str | None = None) -> Path:
        path.write_text(
            content
            or f"outs:\n- md5: {self.HASH}\n  size: 123\n  hash: md5\n  path: balanced-reviews.txt\n",
            encoding="utf-8",
        )
        return path

    def test_extracts_dvc_dataset_hash_from_pointer(self):
        with tempfile.TemporaryDirectory() as directory:
            pointer = self._write_pointer(Path(directory) / "dataset.dvc")
            self.assertEqual(get_dvc_dataset_hash(pointer), self.HASH)

    def test_generates_dvc_and_git_lineage_tags(self):
        with tempfile.TemporaryDirectory() as directory:
            pointer = self._write_pointer(Path(directory) / "dataset.dvc")
            self.assertEqual(
                mlflow_lineage_tags(pointer, git_revision="abc123"),
                {DVC_DATASET_HASH_TAG: self.HASH, "git_commit": "abc123"},
            )

    def test_missing_or_malformed_pointer_fails_clearly(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.dvc"
            with self.assertRaises(FileNotFoundError):
                get_dvc_dataset_hash(missing)

            malformed = self._write_pointer(
                Path(directory) / "malformed.dvc", "outs: []\n"
            )
            with self.assertRaisesRegex(ValueError, "exactly one file output"):
                get_dvc_dataset_hash(malformed)

            invalid_hash = self._write_pointer(
                Path(directory) / "invalid-hash.dvc", "outs:\n- md5: not-a-hash\n"
            )
            with self.assertRaisesRegex(ValueError, "malformed MD5"):
                get_dvc_dataset_hash(invalid_hash)

            invalid_yaml = self._write_pointer(
                Path(directory) / "invalid-yaml.dvc", "outs: [\n"
            )
            with self.assertRaisesRegex(ValueError, "Cannot parse DVC pointer"):
                get_dvc_dataset_hash(invalid_yaml)

            non_mapping = self._write_pointer(
                Path(directory) / "non-mapping.dvc", "a plain scalar\n"
            )
            with self.assertRaisesRegex(ValueError, "must contain a mapping"):
                get_dvc_dataset_hash(non_mapping)

            invalid_hash_type = self._write_pointer(
                Path(directory) / "invalid-hash-type.dvc",
                "outs:\n- md5: 123\n  hash: sha256\n",
            )
            with self.assertRaisesRegex(ValueError, "does not contain an MD5"):
                get_dvc_dataset_hash(invalid_hash_type)

    def test_git_revision_falls_back_when_git_is_unavailable(self):
        with patch(
            "arabic_sentiment.dvc_lineage.subprocess.check_output",
            side_effect=FileNotFoundError("git executable unavailable"),
        ):
            self.assertEqual(current_git_revision(), "unknown")

    def test_project_mlflow_run_records_dvc_hash_tag(self):
        self.addCleanup(
            configure_tracking,
            f"sqlite:///{DEFAULT_TRACKING_DB.as_posix()}",
            DEFAULT_ARTIFACT_ROOT,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pointer = self._write_pointer(root / "dataset.dvc")
            configure_tracking(
                f"sqlite:///{(root / 'tracking.db').as_posix()}",
                artifact_root=root / "artifacts",
            )
            with start_project_run("lineage-test", pointer_path=pointer) as run:
                pass

            tracked = MlflowClient().get_run(run.info.run_id)
            self.assertEqual(tracked.data.tags[DVC_DATASET_HASH_TAG], self.HASH)
            self.assertTrue(tracked.data.tags["git_commit"])


if __name__ == "__main__":
    unittest.main()

"""Helpers for resolving DVC dataset provenance for experiment tracking."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DVC_POINTER = Path("data/raw/balanced-reviews.txt.dvc")
DVC_DATASET_HASH_TAG = "dvc.dataset_hash"
GIT_REVISION_TAG = "git_commit"


def get_dvc_dataset_hash(pointer_path: str | Path = DEFAULT_DVC_POINTER) -> str:
    """Read and validate the raw dataset's content hash from its DVC pointer."""
    pointer = Path(pointer_path)
    try:
        document: Any = yaml.safe_load(pointer.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot parse DVC pointer {pointer}: {exc}") from exc

    if not isinstance(document, dict):
        raise ValueError(  # noqa: TRY004 - malformed file content is a value error.
            f"DVC pointer must contain a mapping: {pointer}"
        )
    outputs = document.get("outs")
    if (
        not isinstance(outputs, list)
        or len(outputs) != 1
        or not isinstance(outputs[0], dict)
    ):
        raise ValueError(f"DVC pointer must contain exactly one file output: {pointer}")
    output = outputs[0]
    digest = output.get("md5")
    if output.get("hash", "md5") != "md5" or not isinstance(digest, str):
        raise ValueError(f"DVC pointer does not contain an MD5 content hash: {pointer}")
    if re.fullmatch(r"[0-9a-f]{32}", digest) is None:
        raise ValueError(
            f"DVC pointer contains a malformed MD5 content hash: {pointer}"
        )
    return digest


def current_git_revision() -> str:
    """Return the current Git HEAD revision, or 'unknown' outside a Git checkout."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def mlflow_lineage_tags(
    pointer_path: str | Path = DEFAULT_DVC_POINTER,
    *,
    git_revision: str | None = None,
) -> dict[str, str]:
    """Build MLflow lineage tags from the current DVC pointer and Git revision."""
    return {
        DVC_DATASET_HASH_TAG: get_dvc_dataset_hash(pointer_path),
        GIT_REVISION_TAG: (
            git_revision if git_revision is not None else current_git_revision()
        ),
    }

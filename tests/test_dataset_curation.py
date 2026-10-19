import hashlib
import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from arabic_sentiment.data import RAW_COLUMNS, load_raw_data
from arabic_sentiment.preprocessing import clean_review_text, prepare_reviews

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "curate_conflicting_reviews.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "curate_conflicting_reviews", _SCRIPT_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_CURATION = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CURATION)
curate = _CURATION.curate


@pytest.fixture
def raw_reviews():
    return pd.DataFrame(
        [
            ["1", "Hotel A", "1", "Guest", "Room", "1", "  نص   متعارض "],
            ["2", "Hotel B", "4", "Guest", "Room", "2", "نص متعارض"],
            ["3", "Hotel A", "1", "Guest", "Room", "1", "نص مكرر ثابت"],
            ["4", "Hotel B", "2", "Guest", "Room", "2", " نص   مكرر ثابت "],
            ["5", "Hotel C", "5", "Guest", "Suite", "3", "إقامة رائعة"],
        ],
        columns=RAW_COLUMNS,
        dtype="string",
    )


def _write_source(path, frame):
    frame.to_csv(path, sep="\t", index=False, encoding="utf-16", lineterminator="\r\n")
    return path


def test_curation_excludes_every_row_in_a_normalized_conflict_group(
    tmp_path, raw_reviews
):
    source = _write_source(tmp_path / "source.txt", raw_reviews)
    destination = tmp_path / "curated.txt"

    curate(source, destination)

    curated = load_raw_data(destination)
    normalized = clean_review_text(curated["review"])
    assert "نص متعارض" not in normalized.tolist()


def test_curation_retains_consistent_duplicate_reviews_in_source_order(
    tmp_path, raw_reviews
):
    source = _write_source(tmp_path / "source.txt", raw_reviews)
    destination = tmp_path / "curated.txt"

    curate(source, destination)

    curated = load_raw_data(destination)
    assert curated["no"].tolist() == ["3", "4", "5"]
    assert clean_review_text(curated.loc[:1, "review"]).tolist() == [
        "نص مكرر ثابت",
        "نص مكرر ثابت",
    ]
    prepared, _ = prepare_reviews(curated)
    assert prepared.loc[:1, "sentiment"].tolist() == ["negative", "negative"]


def test_curation_does_not_modify_source_bytes_or_hash(tmp_path, raw_reviews):
    source = _write_source(tmp_path / "source.txt", raw_reviews)
    original_bytes = source.read_bytes()
    original_sha256 = hashlib.sha256(original_bytes).hexdigest()

    curate(source, tmp_path / "curated.txt")

    resulting_bytes = source.read_bytes()
    assert resulting_bytes == original_bytes
    assert hashlib.sha256(resulting_bytes).hexdigest() == original_sha256


def test_curated_output_loads_with_original_schema_and_row_order(tmp_path, raw_reviews):
    source = _write_source(tmp_path / "source.txt", raw_reviews)
    destination = tmp_path / "curated.txt"

    curate(source, destination)

    curated = load_raw_data(destination)
    assert curated.columns.tolist() == RAW_COLUMNS
    assert curated["no"].tolist() == ["3", "4", "5"]
    assert curated["rating"].tolist() == ["1", "2", "5"]


def test_curation_rejects_same_source_and_destination_without_changing_source(
    tmp_path, raw_reviews
):
    source = _write_source(tmp_path / "source.txt", raw_reviews)
    original_bytes = source.read_bytes()

    with pytest.raises(ValueError, match="different paths"):
        curate(source, source)

    assert source.read_bytes() == original_bytes


def test_curation_summary_reports_exact_conflict_and_label_counts(
    tmp_path, raw_reviews
):
    source = _write_source(tmp_path / "source.txt", raw_reviews)
    destination = tmp_path / "curated.txt"

    summary = curate(source, destination)

    assert summary["source_rows"] == 5
    assert summary["conflicting_normalized_review_groups"] == 1
    assert summary["excluded_rows"] == 2
    assert summary["excluded_mapped_label_counts"] == {"negative": 1, "positive": 1}
    assert summary["output_rows"] == 3
    assert summary["output_mapped_label_counts"] == {"negative": 2, "positive": 1}
    assert summary["remaining_conflicting_groups"] == 0

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

from arabic_sentiment.data import RAW_COLUMNS, load_raw_data
from arabic_sentiment.preprocessing import clean_review_text

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "curate_three_class_reviews.py"
)
_SPEC = importlib.util.spec_from_file_location("curate_three_class_reviews", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_CURATION = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CURATION)
curate_three_class = _CURATION.curate_three_class


def _write_source(path: Path, rows: list[list[str]]) -> Path:
    frame = pd.DataFrame(rows, columns=RAW_COLUMNS, dtype="string")
    frame.to_csv(path, sep="\t", index=False, encoding="utf-16", lineterminator="\r\n")
    return path


def _inputs(tmp_path: Path) -> tuple[Path, Path, bytes]:
    source = _write_source(
        tmp_path / "unbalanced.txt",
        [
            ["1", "H", "1", "U", "R", "1", "source conflict"],
            ["2", "H", "4", "U", "R", "1", " source   conflict "],
            ["3", "H", "1", "U", "R", "1", "consistent duplicate"],
            ["4", "H", "2", "U", "R", "1", " consistent   duplicate "],
            ["5", "H", "3", "U", "R", "1", "neutral unique"],
            ["6", "H", "4", "U", "R", "1", "same V2 positive"],
            ["7", "H", "3", "U", "R", "1", "same V2 negative"],
            ["8", "H", "", "U", "R", "1", "missing rating"],
            ["9", "H", "invalid", "U", "R", "1", "malformed rating"],
            ["10", "H", "3", "U", "R", "1", "   "],
            ["11", "H", "5", "U", "R", "1", "positive unique"],
        ],
    )
    existing_v2 = _write_source(
        tmp_path / "version2.txt",
        [
            ["101", "H", "5", "U", "R", "1", "same V2 positive"],
            ["102", "H", "1", "U", "R", "1", "same V2 negative"],
        ],
    )
    return source, existing_v2, source.read_bytes()


def _run(tmp_path: Path, source: Path, existing_v2: Path) -> dict:
    return curate_three_class(
        source,
        existing_v2,
        tmp_path / "candidate.tsv",
        tmp_path / "curation-report.json",
        tmp_path / "excluded-rows.tsv",
    )


def test_three_class_mapping_conflicts_v2_policy_and_first_occurrence(tmp_path):
    source, existing_v2, _ = _inputs(tmp_path)

    summary = _run(tmp_path, source, existing_v2)

    candidate = load_raw_data(tmp_path / "candidate.tsv")
    assert candidate.columns.tolist() == [*RAW_COLUMNS, "sentiment"]
    assert candidate["no"].tolist() == ["3", "5", "6", "11"]
    assert candidate["sentiment"].tolist() == [
        "negative",
        "neutral",
        "positive",
        "positive",
    ]
    assert summary["source"]["rating_counts"] == {
        "1": 2,
        "2": 1,
        "3": 3,
        "4": 2,
        "5": 1,
    }
    assert summary["curation"]["output_label_counts"] == {
        "negative": 1,
        "neutral": 1,
        "positive": 2,
    }
    assert summary["curation"]["remaining_normalized_label_conflicts"] == 0
    assert summary["version_2_reference"]["candidate_conflict_groups"] == 1
    assert summary["version_2_reference"]["candidate_rows_in_conflicting_groups"] == 1
    assert summary["curation"]["excluded_rows_by_reason"] == {
        "duplicate_normalized_label": 1,
        "empty_review": 1,
        "existing_v2_label_conflict": 1,
        "malformed_rating": 1,
        "missing_rating": 1,
        "source_label_conflict": 2,
    }


def test_candidate_is_deterministic_and_exclusion_audit_preserves_source_ids(
    tmp_path,
):
    source, existing_v2, original_source = _inputs(tmp_path)

    first = _run(tmp_path, source, existing_v2)
    first_candidate = (tmp_path / "candidate.tsv").read_bytes()
    first_report = (tmp_path / "curation-report.json").read_bytes()
    first_audit = (tmp_path / "excluded-rows.tsv").read_bytes()
    second = _run(tmp_path, source, existing_v2)

    assert (tmp_path / "candidate.tsv").read_bytes() == first_candidate
    assert (tmp_path / "curation-report.json").read_bytes() == first_report
    assert (tmp_path / "excluded-rows.tsv").read_bytes() == first_audit
    assert source.read_bytes() == original_source
    assert first == second
    assert first["source"]["sha256"] == hashlib.sha256(original_source).hexdigest()
    assert first["curation"]["output_sha256"] == hashlib.sha256(first_candidate).hexdigest()

    audit = pd.read_csv(tmp_path / "excluded-rows.tsv", sep="\t", dtype="string")
    assert set(audit["source_row_id"]) == {"1", "2", "4", "7", "8", "9", "10"}
    assert {"source_row_number", "source_row_id", "rating", "reason"}.issubset(
        audit.columns
    )
    assert "review" not in audit.columns
    report = json.loads((tmp_path / "curation-report.json").read_text(encoding="utf-8"))
    assert report["source"]["row_count_matches_previous_citation"] is False
    assert report["curation"]["source_row_id_preserved_in_output"] is True


def test_final_candidate_has_one_label_per_normalized_review(tmp_path):
    source, existing_v2, _ = _inputs(tmp_path)
    _run(tmp_path, source, existing_v2)

    candidate = load_raw_data(tmp_path / "candidate.tsv")
    normalized = clean_review_text(candidate["review"])
    conflicts = (
        pd.DataFrame({"normalized": normalized, "sentiment": candidate["sentiment"]})
        .groupby("normalized", sort=False)["sentiment"]
        .nunique()
    )
    assert not conflicts.gt(1).any()

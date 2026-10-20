"""Curate a three-class HARD review candidate without changing source data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from arabic_sentiment.data import load_raw_data
from arabic_sentiment.preprocessing import clean_review_text, prepare_reviews

RATING_TO_SENTIMENT = {
    1: "negative",
    2: "negative",
    3: "neutral",
    4: "positive",
    5: "positive",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_counts(values: pd.Series) -> dict[str, int]:
    counts = values.value_counts().sort_index()
    return {str(label): int(count) for label, count in counts.items()}


def curate_three_class(
    source: Path,
    existing_binary: Path,
    destination: Path,
    report_path: Path,
    exclusions_path: Path,
) -> dict[str, object]:
    """Write a deterministic three-class candidate and a text-free audit.

    The first source row is retained for each eligible normalized text/label.
    Whole normalized-text groups are excluded when source labels conflict or
    when the candidate label conflicts with the existing binary dataset.
    """
    paths = [
        Path(p).resolve()
        for p in (source, existing_binary, destination, report_path, exclusions_path)
    ]
    if len(set(paths)) != len(paths):
        raise ValueError("Source, reference, output, and audit paths must differ")
    source, existing_binary, destination, report_path, exclusions_path = paths

    raw = load_raw_data(source)
    reference_raw = load_raw_data(existing_binary)
    source_sha256 = _sha256(source)
    source_size = source.stat().st_size

    original_review = raw["review"].astype("string")
    normalized = clean_review_text(original_review)
    rating_text = raw["rating"].astype("string").str.strip()
    numeric_rating = pd.to_numeric(rating_text, errors="coerce")
    rating_is_valid = numeric_rating.isin(RATING_TO_SENTIMENT) & numeric_rating.eq(
        numeric_rating.round()
    )
    missing_rating = rating_text.isna() | rating_text.eq("")
    malformed_rating = ~missing_rating & ~rating_is_valid
    mapped = numeric_rating.map(RATING_TO_SENTIMENT).astype("string")
    nonempty_review = normalized.ne("")
    source_eligible = rating_is_valid & nonempty_review

    source_group_labels = (
        pd.DataFrame(
            {
                "review": normalized.loc[source_eligible],
                "sentiment": mapped.loc[source_eligible],
            }
        )
        .groupby("review", sort=False)["sentiment"]
        .agg(lambda values: frozenset(values))
    )
    source_conflicting_groups = set(
        source_group_labels[source_group_labels.map(len).gt(1)].index
    )

    reference, _ = prepare_reviews(reference_raw)
    reference_labels = reference.groupby("review", sort=False)["sentiment"].agg(
        lambda values: frozenset(values)
    )
    candidate_group_labels = source_group_labels
    reference_conflicting_groups = {
        review
        for review, candidate_labels in candidate_group_labels.items()
        if review in reference_labels and candidate_labels != reference_labels[review]
    }
    v2_conflict_rows = source_eligible & normalized.isin(reference_conflicting_groups)
    source_conflict_rows = source_eligible & normalized.isin(source_conflicting_groups)
    source_and_v2_conflict_rows = v2_conflict_rows & source_conflict_rows

    reasons = pd.Series("", index=raw.index, dtype="string")
    reasons.loc[missing_rating] = "missing_rating"
    reasons.loc[malformed_rating] = "malformed_rating"
    reasons.loc[~missing_rating & ~malformed_rating & ~nonempty_review] = "empty_review"
    valid_nonempty = source_eligible
    reasons.loc[valid_nonempty & normalized.isin(source_conflicting_groups)] = (
        "source_label_conflict"
    )
    reasons.loc[
        valid_nonempty
        & ~normalized.isin(source_conflicting_groups)
        & normalized.isin(reference_conflicting_groups)
    ] = "existing_v2_label_conflict"

    eligible = reasons.eq("")
    eligible_frame = pd.DataFrame(
        {
            "normalized": normalized.loc[eligible],
            "sentiment": mapped.loc[eligible],
        }
    )
    duplicate_rows = eligible_frame.duplicated(
        subset=["normalized", "sentiment"], keep="first"
    )
    duplicate_indexes = eligible_frame.index[duplicate_rows]
    reasons.loc[duplicate_indexes] = "duplicate_normalized_label"
    retained = reasons.eq("")

    curated = raw.loc[retained].copy()
    curated["sentiment"] = mapped.loc[retained].astype("string")
    final_groups = (
        pd.DataFrame(
            {
                "normalized": normalized.loc[retained],
                "sentiment": mapped.loc[retained],
            }
        )
        .groupby("normalized", sort=False)["sentiment"]
        .nunique()
    )
    remaining_conflicts = int(final_groups.gt(1).sum())
    if remaining_conflicts:
        raise RuntimeError("Curated output contains conflicting normalized labels")
    final_candidate_labels = (
        pd.DataFrame(
            {
                "review": normalized.loc[retained],
                "sentiment": mapped.loc[retained],
            }
        )
        .groupby("review", sort=False)["sentiment"]
        .agg(lambda values: frozenset(values))
    )
    remaining_v2_conflicts = {
        review
        for review, candidate_labels in final_candidate_labels.items()
        if review in reference_labels and candidate_labels != reference_labels[review]
    }
    if remaining_v2_conflicts:
        raise RuntimeError("Curated output conflicts with existing Version 2 labels")

    destination.parent.mkdir(parents=True, exist_ok=True)
    curated.to_csv(
        destination,
        sep="\t",
        encoding="utf-16",
        index=False,
        lineterminator="\r\n",
    )

    normalized_hashes = normalized.map(
        lambda text: hashlib.sha256(str(text).encode("utf-8")).hexdigest()
    )
    excluded = reasons.ne("")
    audit = pd.DataFrame(
        {
            "source_row_number": raw.index[excluded] + 1,
            "source_row_id": raw.loc[excluded, "no"].astype("string"),
            "rating": rating_text.loc[excluded],
            "reason": reasons.loc[excluded],
            "normalized_review_sha256": normalized_hashes.loc[excluded],
        }
    )
    exclusions_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(exclusions_path, sep="\t", index=False, lineterminator="\n")

    exact_counts = original_review.dropna().value_counts()
    normalized_counts = normalized.loc[nonempty_review].value_counts()
    rating_counts = {
        str(value): int(numeric_rating.eq(value).sum()) for value in range(1, 6)
    }
    excluded_by_reason = {
        str(reason): int(count)
        for reason, count in reasons.loc[excluded].value_counts().sort_index().items()
    }
    source_id_text = raw["no"].astype("string").str.strip()
    present_ids = source_id_text.notna() & source_id_text.ne("")
    id_duplicate_rows = int(source_id_text.loc[present_ids].duplicated().sum())

    summary: dict[str, object] = {
        "policy": {
            "rating_to_sentiment": {str(k): v for k, v in RATING_TO_SENTIMENT.items()},
            "normalization": "arabic_sentiment.preprocessing.clean_review_text",
            "source_conflicts": "exclude every row in normalized groups with multiple mapped labels",
            "existing_v2_conflicts": "exclude every candidate row in a normalized group whose label set differs from Version 2",
            "deduplication": "retain first eligible source occurrence per normalized text and mapped label",
        },
        "source": {
            "path": source.name,
            "bytes": source_size,
            "sha256": source_sha256,
            "encoding": "UTF-16 little-endian",
            "delimiter": "tab",
            "rows": len(raw),
            "previously_cited_rows": 373750,
            "row_count_matches_previous_citation": len(raw) == 373750,
            "rating_counts": rating_counts,
            "missing_rating_rows": int(missing_rating.sum()),
            "malformed_or_out_of_range_rating_rows": int(malformed_rating.sum()),
            "empty_review_rows": int((~nonempty_review).sum()),
            "source_id_column": "no",
            "source_id_missing_rows": int((~present_ids).sum()),
            "source_id_duplicate_rows": id_duplicate_rows,
            "exact_duplicate_groups": int(exact_counts.gt(1).sum()),
            "exact_duplicate_excess_rows": int(exact_counts.sub(1).clip(lower=0).sum()),
            "normalized_duplicate_groups": int(normalized_counts.gt(1).sum()),
            "normalized_duplicate_excess_rows": int(
                normalized_counts.sub(1).clip(lower=0).sum()
            ),
        },
        "version_2_reference": {
            "path": existing_binary.name,
            "source_rows": len(reference_raw),
            "normalized_text_groups": int(reference["review"].nunique()),
            "candidate_conflict_groups": len(reference_conflicting_groups),
            "candidate_rows_in_conflicting_groups": int(v2_conflict_rows.sum()),
            "candidate_conflict_groups_already_excluded_by_source_rule": len(
                reference_conflicting_groups & source_conflicting_groups
            ),
            "candidate_rows_already_excluded_by_source_rule": int(
                source_and_v2_conflict_rows.sum()
            ),
            "candidate_rows_excluded_only_by_v2_rule": int(
                reasons.eq("existing_v2_label_conflict").sum()
            ),
            "remaining_candidate_conflict_groups": len(remaining_v2_conflicts),
        },
        "curation": {
            "source_conflict_groups": len(source_conflicting_groups),
            "source_conflict_rows": int(
                (source_eligible & normalized.isin(source_conflicting_groups)).sum()
            ),
            "excluded_rows_by_reason": excluded_by_reason,
            "excluded_rows_total": int(excluded.sum()),
            "output_rows": len(curated),
            "output_label_counts": _label_counts(curated["sentiment"]),
            "remaining_normalized_label_conflicts": remaining_conflicts,
            "output_bytes": destination.stat().st_size,
            "output_sha256": _sha256(destination),
            "exclusion_audit_path": exclusions_path.name,
            "exclusion_audit_rows": len(audit),
            "exclusion_audit_sha256": _sha256(exclusions_path),
            "source_row_id_preserved_in_output": "no",
        },
    }
    # Keep original identifier and all source metadata in the output schema.
    summary["curation"]["source_row_id_preserved_in_output"] = "no" in curated.columns

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("existing_binary", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("report", type=Path)
    parser.add_argument("exclusions", type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            curate_three_class(
                args.source,
                args.existing_binary,
                args.destination,
                args.report,
                args.exclusions,
            ),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

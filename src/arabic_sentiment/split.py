"""Deterministic row-proportioned splitting grouped by cleaned review text."""

import numpy as np
import pandas as pd

DEFAULT_SEED = 42
SPLIT_NAMES = ("train", "validation", "test")
SPLIT_FRACTIONS = np.array([0.8, 0.1, 0.1], dtype=float)


def split_reviews(
    data: pd.DataFrame, seed: int = DEFAULT_SEED
) -> dict[str, pd.DataFrame]:
    """Assign every cleaned review text group to one split, balancing class counts.

    Groups are assigned largest-first. A seeded tie order and a greedy minimum
    class-count deviation make the assignment deterministic and near the target
    80/10/10 proportions without separating duplicate review text.
    """
    if data.empty:
        raise ValueError("Cannot split an empty dataset")
    if data["review"].isna().any() or data["review"].eq("").any():
        raise ValueError("Reviews must be nonempty before splitting")
    if not set(data["sentiment"].unique()).issubset({"negative", "positive"}):
        raise ValueError("Sentiment must contain only 'negative' and 'positive'")

    group_ids, _ = pd.factorize(data["review"], sort=True)
    group_count = int(group_ids.max()) + 1
    labels = data["sentiment"].to_numpy()
    class_codes = (labels == "positive").astype(np.int8)
    group_class_counts = np.zeros((group_count, 2), dtype=np.int64)
    np.add.at(group_class_counts, (group_ids, class_codes), 1)

    class_totals = group_class_counts.sum(axis=0)
    targets = SPLIT_FRACTIONS[:, None] * class_totals[None, :]
    assignments = np.zeros((3, 2), dtype=np.int64)

    rng = np.random.default_rng(seed)
    tie_order = rng.permutation(group_count)
    order = tie_order[np.argsort(-group_class_counts[tie_order].sum(axis=1), kind="stable")]
    group_split = np.empty(group_count, dtype=np.int8)

    for group_id in order:
        addition = group_class_counts[group_id]
        before = assignments - targets
        after = assignments + addition - targets
        scores = (((after**2) - (before**2)) / np.maximum(targets, 1)).sum(axis=1)
        chosen = int(np.argmin(scores))
        group_split[group_id] = chosen
        assignments[chosen] += addition

    row_splits = group_split[group_ids]
    return {
        name: data.loc[row_splits == i].reset_index(drop=True)
        for i, name in enumerate(SPLIT_NAMES)
    }


def verify_no_review_leakage(splits: dict[str, pd.DataFrame]) -> bool:
    """Return true when no cleaned review text occurs in multiple splits."""
    seen: set[str] = set()
    for name in SPLIT_NAMES:
        reviews = set(splits[name]["review"])
        if seen.intersection(reviews):
            return False
        seen.update(reviews)
    return True

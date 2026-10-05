"""Conservative review cleaning and binary sentiment labeling."""

import pandas as pd

RATING_TO_SENTIMENT = {1: "negative", 2: "negative", 4: "positive", 5: "positive"}


def clean_review_text(values: pd.Series) -> pd.Series:
    """Replace missing text with empty text, trim it, and collapse whitespace."""
    return (
        values.astype("string")
        .fillna("")
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def prepare_reviews(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    """Return usable review/rating/sentiment rows and exclusion/duplicate counts."""
    original_review = raw["review"].astype("string")
    review = clean_review_text(original_review)
    rating_text = raw["rating"].astype("string").str.strip()
    rating = pd.to_numeric(rating_text, errors="coerce")
    mapped = rating.map(RATING_TO_SENTIMENT)

    missing_rating = rating_text.isna() | rating_text.eq("")
    invalid_rating = ~missing_rating & (rating.isna() | ~rating.isin([1, 2, 3, 4, 5]))
    neutral_rating = rating.eq(3)
    empty_review = review.eq("")
    usable = mapped.notna() & ~empty_review

    result = pd.DataFrame(
        {
            "review": review.loc[usable],
            "rating": rating.loc[usable].astype("int64"),
            "sentiment": mapped.loc[usable].astype("string"),
        }
    ).reset_index(drop=True)

    cleaned_review_counts = review.loc[~empty_review].value_counts()
    stats = {
        "missing_ratings": int(missing_rating.sum()),
        "invalid_ratings": int(invalid_rating.sum()),
        "neutral_rating_rows": int(neutral_rating.sum()),
        "missing_reviews": int(original_review.isna().sum()),
        "empty_reviews": int(empty_review.sum()),
        "whitespace_only_reviews": int(
            (original_review.notna() & original_review.ne("") & review.eq("")).sum()
        ),
        "duplicate_review_rows": int(cleaned_review_counts.sub(1).clip(lower=0).sum()),
        "duplicate_review_groups": int(cleaned_review_counts.gt(1).sum()),
        "conflicting_sentiment_duplicate_groups": int(
            result.groupby("review", sort=False)["sentiment"].nunique().gt(1).sum()
        ),
        "conflicting_sentiment_duplicate_rows": int(
            result["review"]
            .isin(
                result.groupby("review", sort=False)["sentiment"]
                .nunique()
                .loc[lambda x: x.gt(1)]
                .index
            )
            .sum()
        ),
        "excluded_rows": int((~usable).sum()),
    }
    return result, stats

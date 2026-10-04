"""Loading and schema validation for the HARD reviews dataset."""

from pathlib import Path

import pandas as pd

RAW_COLUMNS = [
    "no",
    "Hotel name",
    "rating",
    "user type",
    "room type",
    "nights",
    "review",
]


def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Read the UTF-16 tab-separated source without changing the source file."""
    path = Path(path)
    with path.open("rb") as source:
        if source.read(2) != b"\xff\xfe":
            raise ValueError("Expected a UTF-16 little-endian BOM in the raw dataset")

    frame = pd.read_csv(
        path,
        sep="\t",
        encoding="utf-16",
        dtype="string",
        keep_default_na=True,
    )
    validate_raw_schema(frame)
    return frame


def validate_raw_schema(frame: pd.DataFrame) -> None:
    """Require the documented source fields while allowing additional metadata."""
    missing = [column for column in RAW_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Raw dataset is missing required columns: {missing}")
    if frame.columns.duplicated().any():
        raise ValueError("Raw dataset contains duplicate column names")

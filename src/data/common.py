"""Shared utilities for MovieLens data loading and schema normalization."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd  # type: ignore[import-not-found]

RATINGS_COLUMNS = ["UserID", "MovieID", "Rating", "Timestamp"]
MOVIES_COLUMNS = ["MovieID", "Title", "Genres"]
USERS_COLUMNS = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]

COLUMN_ALIASES = {
    "userid": "user_id",
    "movieid": "movie_id",
    "zipcode": "zip_code",
}


def standardize_column_name(column_name: str) -> str:
    """Convert a column name to lowercase snake_case."""
    cleaned_name = column_name.strip().lower()
    cleaned_name = re.sub(r"[^a-z0-9]+", "_", cleaned_name)
    return re.sub(r"_+", "_", cleaned_name).strip("_")


def standardize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply standardized naming to all DataFrame columns."""
    standardized_df = df.copy()
    standardized_df.columns = [standardize_column_name(col) for col in standardized_df.columns]
    return standardized_df


def canonicalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize standardized columns into canonical project names."""
    standardized_df = standardize_dataframe_columns(df)
    canonical_df = standardized_df.rename(columns={col: COLUMN_ALIASES.get(col, col) for col in standardized_df.columns})

    if canonical_df.columns.duplicated().any():
        canonical_df = canonical_df.loc[:, ~canonical_df.columns.duplicated()]

    return canonical_df


def load_table(file_path: Path, columns: list[str]) -> pd.DataFrame:
    """Load a MovieLens DAT file that uses double-colon separators."""
    return pd.read_csv(
        file_path,
        sep="::",
        engine="python",
        names=columns,
        encoding="latin-1",
    )


def validate_input_files(raw_dir: Path) -> dict[str, Path]:
    """Check that all required MovieLens source files exist before loading."""
    required_files = {
        "ratings": raw_dir / "ratings.dat",
        "movies": raw_dir / "movies.dat",
        "users": raw_dir / "users.dat",
    }

    missing_files = [str(path) for path in required_files.values() if not path.exists()]
    if missing_files:
        missing_text = "\n".join(missing_files)
        raise FileNotFoundError(f"Missing required raw data files:\n{missing_text}")

    return required_files

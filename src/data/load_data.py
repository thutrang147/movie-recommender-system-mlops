"""Load and lightly clean MovieLens raw data tables."""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict

import pandas as pd  # type: ignore[import-not-found]

RATINGS_COLUMNS = ["UserID", "MovieID", "Rating", "Timestamp"]
MOVIES_COLUMNS = ["MovieID", "Title", "Genres"]
USERS_COLUMNS = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]

# Map common legacy or compact names to canonical snake_case names.
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

    # Keep the first column when multiple aliases collapse to the same canonical name.
    if canonical_df.columns.duplicated().any():
        canonical_df = canonical_df.loc[:, ~canonical_df.columns.duplicated()]

    return canonical_df


def load_table(file_path: Path, columns: list[str]) -> pd.DataFrame:
    """Load a MovieLens file using the original double-colon separator."""
    return pd.read_csv(
        file_path,
        sep="::",
        engine="python",
        names=columns,
        encoding="latin-1",
    )


def resolve_data_dirs(raw_dir: str | None, output_dir: str | None) -> tuple[Path, Path]:
    """Resolve input/output directories from args or project defaults."""
    project_root = Path(__file__).resolve().parents[2]
    resolved_raw_dir = Path(raw_dir) if raw_dir else project_root / "data" / "raw"
    resolved_output_dir = Path(output_dir) if output_dir else project_root / "data" / "interim"
    return resolved_raw_dir, resolved_output_dir


def validate_input_files(raw_dir: Path) -> Dict[str, Path]:
    """Check that all required raw files exist before loading."""
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


def save_cleaned_tables(tables: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Persist preliminary cleaned tables for downstream steps."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for table_name, table_df in tables.items():
        output_path = output_dir / f"{table_name}_cleaned.csv"
        table_df.to_csv(output_path, index=False)
        print(f"Saved {table_name} cleaned data to: {output_path}")


def main() -> Dict[str, pd.DataFrame]:
    """Entrypoint for loading, standardizing, and optionally saving data."""
    parser = argparse.ArgumentParser(
        description="Load MovieLens ratings, movies, and users tables with basic cleaning."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Path to raw data directory (default: project_root/data/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to cleaned output directory (default: project_root/data/interim).",
    )
    parser.add_argument(
        "--save-cleaned",
        dest="save_cleaned",
        action="store_true",
        help="Save cleaned tables to CSV files.",
    )
    parser.add_argument(
        "--no-save-cleaned",
        dest="save_cleaned",
        action="store_false",
        help="Skip writing cleaned tables to disk.",
    )
    parser.set_defaults(save_cleaned=True)

    args = parser.parse_args()

    raw_dir, output_dir = resolve_data_dirs(args.raw_dir, args.output_dir)
    file_paths = validate_input_files(raw_dir)

    ratings_df = canonicalize_dataframe_columns(load_table(file_paths["ratings"], RATINGS_COLUMNS))
    movies_df = canonicalize_dataframe_columns(load_table(file_paths["movies"], MOVIES_COLUMNS))
    users_df = canonicalize_dataframe_columns(load_table(file_paths["users"], USERS_COLUMNS))

    tables = {
        "ratings": ratings_df,
        "movies": movies_df,
        "users": users_df,
    }

    print("Loaded table shapes:")
    for table_name, table_df in tables.items():
        print(f"- {table_name}: {table_df.shape}")

    if args.save_cleaned:
        save_cleaned_tables(tables, output_dir)
    else:
        print("Skipped saving cleaned tables.")

    return tables


if __name__ == "__main__":
    main()

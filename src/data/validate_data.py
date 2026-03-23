"""Validate data quality for MovieLens ratings, movies, and users tables."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd  # type: ignore[import-not-found]

try:
    from src.data.common import (
        MOVIES_COLUMNS,
        RATINGS_COLUMNS,
        USERS_COLUMNS,
        canonicalize_dataframe_columns,
        load_table,
        validate_input_files,
    )
except ModuleNotFoundError:
    from common import (  # type: ignore[no-redef]
        MOVIES_COLUMNS,
        RATINGS_COLUMNS,
        USERS_COLUMNS,
        canonicalize_dataframe_columns,
        load_table,
        validate_input_files,
    )


def resolve_paths(
    raw_dir: str | None,
    interim_dir: str | None,
    report_path: str | None,
) -> tuple[Path, Path, Path]:
    """Resolve project-relative defaults for raw/interim data and report output."""
    project_root = Path(__file__).resolve().parents[2]
    resolved_raw_dir = Path(raw_dir) if raw_dir else project_root / "data" / "raw"
    resolved_interim_dir = Path(interim_dir) if interim_dir else project_root / "data" / "interim"
    resolved_report_path = Path(report_path) if report_path else project_root / "docs" / "data_quality_report.md"
    return resolved_raw_dir, resolved_interim_dir, resolved_report_path


def load_datasets(raw_dir: Path, interim_dir: Path, force_raw: bool) -> tuple[Dict[str, pd.DataFrame], str]:
    """Load cleaned CSV files when available, otherwise load raw DAT files."""
    cleaned_paths = {
        "ratings": interim_dir / "ratings_cleaned.csv",
        "movies": interim_dir / "movies_cleaned.csv",
        "users": interim_dir / "users_cleaned.csv",
    }

    use_cleaned = not force_raw and all(path.exists() for path in cleaned_paths.values())
    if use_cleaned:
        tables = {
            "ratings": pd.read_csv(cleaned_paths["ratings"]),
            "movies": pd.read_csv(cleaned_paths["movies"]),
            "users": pd.read_csv(cleaned_paths["users"]),
        }
        source_label = "cleaned CSV files from data/interim"
    else:
        raw_paths = validate_input_files(raw_dir)

        tables = {
            "ratings": load_table(raw_paths["ratings"], RATINGS_COLUMNS),
            "movies": load_table(raw_paths["movies"], MOVIES_COLUMNS),
            "users": load_table(raw_paths["users"], USERS_COLUMNS),
        }
        source_label = "raw DAT files from data/raw"

    canonical_tables = {name: canonicalize_dataframe_columns(df) for name, df in tables.items()}
    return canonical_tables, source_label


def ensure_required_columns(tables: Dict[str, pd.DataFrame]) -> None:
    """Validate required columns to provide a clear error before detailed checks."""
    required_columns = {
        "ratings": {"user_id", "movie_id", "rating", "timestamp"},
        "movies": {"movie_id"},
        "users": {"user_id"},
    }

    missing_details = []
    for table_name, required in required_columns.items():
        available = set(tables[table_name].columns)
        missing = sorted(required - available)
        if missing:
            missing_details.append(f"{table_name}: {', '.join(missing)}")

    if missing_details:
        details_text = "\n".join(missing_details)
        raise ValueError(f"Missing required columns after normalization:\n{details_text}")


def get_id_quality_masks(series: pd.Series) -> Dict[str, pd.Series]:
    """Build masks for null, non-numeric, non-integer, and non-positive ID values."""
    numeric_series = pd.to_numeric(series, errors="coerce")

    null_mask = series.isna()
    non_numeric_mask = series.notna() & numeric_series.isna()
    non_integer_mask = numeric_series.notna() & (numeric_series % 1 != 0)
    non_positive_mask = numeric_series.notna() & (numeric_series <= 0)

    valid_mask = ~(null_mask | non_numeric_mask | non_integer_mask | non_positive_mask)

    return {
        "numeric": numeric_series,
        "null": null_mask,
        "non_numeric": non_numeric_mask,
        "non_integer": non_integer_mask,
        "non_positive": non_positive_mask,
        "valid": valid_mask,
    }


def id_set_from_series(series: pd.Series) -> set[int]:
    """Extract a set of positive integer IDs from a series."""
    masks = get_id_quality_masks(series)
    valid_numeric = masks["numeric"][masks["valid"]]
    return set(valid_numeric.astype(int).tolist())


def check_reference_ids(series: pd.Series, valid_ids: set[int]) -> pd.Series:
    """Return a boolean mask for IDs that are syntactically valid but missing in reference IDs."""
    masks = get_id_quality_masks(series)
    valid_numeric = masks["numeric"][masks["valid"]].astype(int)

    unknown_mask = pd.Series(False, index=series.index)
    if valid_ids:
        unknown_indices = valid_numeric[~valid_numeric.isin(valid_ids)].index
        unknown_mask.loc[unknown_indices] = True

    return unknown_mask


def count_missing_values(df: pd.DataFrame) -> Dict[str, int]:
    """Count missing values per column."""
    return {column: int(count) for column, count in df.isna().sum().to_dict().items()}


def count_duplicate_records(df: pd.DataFrame) -> int:
    """Count exact duplicate rows."""
    return int(df.duplicated().sum())


def build_data_quality_report(
    tables: Dict[str, pd.DataFrame],
    source_label: str,
) -> tuple[str, Dict[str, int]]:
    """Generate a markdown report and a compact issue summary."""
    ratings_df = tables["ratings"]
    movies_df = tables["movies"]
    users_df = tables["users"]

    missing_by_table = {table_name: count_missing_values(df) for table_name, df in tables.items()}
    duplicates_by_table = {table_name: count_duplicate_records(df) for table_name, df in tables.items()}

    ratings_user_masks = get_id_quality_masks(ratings_df["user_id"])
    ratings_movie_masks = get_id_quality_masks(ratings_df["movie_id"])
    users_user_id_set = id_set_from_series(users_df["user_id"])
    movies_movie_id_set = id_set_from_series(movies_df["movie_id"])

    unknown_user_in_ratings_mask = check_reference_ids(ratings_df["user_id"], users_user_id_set)
    unknown_movie_in_ratings_mask = check_reference_ids(ratings_df["movie_id"], movies_movie_id_set)

    rating_numeric = pd.to_numeric(ratings_df["rating"], errors="coerce")
    rating_null_mask = ratings_df["rating"].isna()
    rating_non_numeric_mask = ratings_df["rating"].notna() & rating_numeric.isna()
    rating_out_of_range_mask = rating_numeric.notna() & ((rating_numeric < 1) | (rating_numeric > 5))

    timestamp_numeric = pd.to_numeric(ratings_df["timestamp"], errors="coerce")
    timestamp_null_mask = ratings_df["timestamp"].isna()
    timestamp_non_numeric_mask = ratings_df["timestamp"].notna() & timestamp_numeric.isna()
    timestamp_non_positive_mask = timestamp_numeric.notna() & (timestamp_numeric <= 0)

    timestamp_candidate_mask = timestamp_numeric.notna() & (timestamp_numeric > 0)
    timestamp_parsed = pd.to_datetime(timestamp_numeric[timestamp_candidate_mask], unit="s", errors="coerce", utc=True)

    timestamp_unparsable_mask = pd.Series(False, index=ratings_df.index)
    timestamp_unparsable_indices = timestamp_parsed[timestamp_parsed.isna()].index
    timestamp_unparsable_mask.loc[timestamp_unparsable_indices] = True

    timestamp_future_mask = pd.Series(False, index=ratings_df.index)
    valid_timestamp_series = timestamp_parsed[timestamp_parsed.notna()]
    timestamp_future_indices = valid_timestamp_series[valid_timestamp_series > pd.Timestamp.now(tz="UTC")].index
    timestamp_future_mask.loc[timestamp_future_indices] = True

    duplicate_user_movie_count = int(ratings_df.duplicated(subset=["user_id", "movie_id"]).sum())

    summary = {
        "ratings_missing_cells": int(sum(missing_by_table["ratings"].values())),
        "movies_missing_cells": int(sum(missing_by_table["movies"].values())),
        "users_missing_cells": int(sum(missing_by_table["users"].values())),
        "ratings_exact_duplicates": duplicates_by_table["ratings"],
        "movies_exact_duplicates": duplicates_by_table["movies"],
        "users_exact_duplicates": duplicates_by_table["users"],
        "invalid_user_id_rows": int((~ratings_user_masks["valid"]).sum()),
        "unknown_user_id_rows": int(unknown_user_in_ratings_mask.sum()),
        "invalid_movie_id_rows": int((~ratings_movie_masks["valid"]).sum()),
        "unknown_movie_id_rows": int(unknown_movie_in_ratings_mask.sum()),
        "rating_out_of_range_rows": int(rating_out_of_range_mask.sum()),
        "rating_non_numeric_rows": int(rating_non_numeric_mask.sum()),
        "rating_null_rows": int(rating_null_mask.sum()),
        "timestamp_null_rows": int(timestamp_null_mask.sum()),
        "timestamp_non_numeric_rows": int(timestamp_non_numeric_mask.sum()),
        "timestamp_non_positive_rows": int(timestamp_non_positive_mask.sum()),
        "timestamp_unparsable_rows": int(timestamp_unparsable_mask.sum()),
        "timestamp_future_rows": int(timestamp_future_mask.sum()),
        "duplicate_user_movie_rows": duplicate_user_movie_count,
    }

    report_lines = [
        "# Data Quality Report",
        "",
        f"- Generated at (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Data source: {source_label}",
        "",
        "## Dataset Shapes",
        f"- ratings: {ratings_df.shape}",
        f"- movies: {movies_df.shape}",
        f"- users: {users_df.shape}",
        "",
        "## Missing Values",
    ]

    for table_name, counts in missing_by_table.items():
        total_missing = int(sum(counts.values()))
        report_lines.append(f"### {table_name}")
        report_lines.append(f"- Total missing cells: {total_missing}")

        missing_columns = [f"{column}: {count}" for column, count in counts.items() if count > 0]
        if missing_columns:
            report_lines.append("- Missing by column:")
            for item in missing_columns:
                report_lines.append(f"  - {item}")
        else:
            report_lines.append("- Missing by column: none")

    report_lines.extend(
        [
            "",
            "## Duplicate Records",
            f"- ratings exact duplicate rows: {duplicates_by_table['ratings']}",
            f"- movies exact duplicate rows: {duplicates_by_table['movies']}",
            f"- users exact duplicate rows: {duplicates_by_table['users']}",
            f"- ratings duplicate (user_id, movie_id) rows: {duplicate_user_movie_count}",
            "",
            "## User ID Checks (ratings.user_id)",
            f"- null: {int(ratings_user_masks['null'].sum())}",
            f"- non-numeric (wrong format): {int(ratings_user_masks['non_numeric'].sum())}",
            f"- non-integer (wrong format): {int(ratings_user_masks['non_integer'].sum())}",
            f"- non-positive: {int(ratings_user_masks['non_positive'].sum())}",
            f"- unknown user_id (not found in users table): {int(unknown_user_in_ratings_mask.sum())}",
            "",
            "## Movie ID Checks (ratings.movie_id)",
            f"- null: {int(ratings_movie_masks['null'].sum())}",
            f"- non-numeric (wrong format): {int(ratings_movie_masks['non_numeric'].sum())}",
            f"- non-integer (wrong format): {int(ratings_movie_masks['non_integer'].sum())}",
            f"- non-positive: {int(ratings_movie_masks['non_positive'].sum())}",
            f"- unknown movie_id (not found in movies table): {int(unknown_movie_in_ratings_mask.sum())}",
            "",
            "## Rating Checks (ratings.rating)",
            f"- null: {int(rating_null_mask.sum())}",
            f"- non-numeric: {int(rating_non_numeric_mask.sum())}",
            f"- out of range [1, 5]: {int(rating_out_of_range_mask.sum())}",
            "",
            "## Timestamp Checks (ratings.timestamp)",
            f"- null: {int(timestamp_null_mask.sum())}",
            f"- non-numeric: {int(timestamp_non_numeric_mask.sum())}",
            f"- non-positive: {int(timestamp_non_positive_mask.sum())}",
            f"- unparsable as Unix timestamp: {int(timestamp_unparsable_mask.sum())}",
            f"- future timestamps: {int(timestamp_future_mask.sum())}",
            "",
            "## Quick Summary",
            "- A value greater than 0 in any check indicates a data quality issue to investigate.",
        ]
    )

    return "\n".join(report_lines), summary


def print_summary_to_terminal(summary: Dict[str, int]) -> None:
    """Print a concise terminal-friendly summary of validation findings."""
    print("Validation summary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")


def main() -> Tuple[str, Dict[str, int]]:
    """Run data validation and optionally write a markdown report."""
    parser = argparse.ArgumentParser(
        description="Validate missing values, duplicates, IDs, rating range, and timestamps."
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Path to raw data directory (default: project_root/data/raw).",
    )
    parser.add_argument(
        "--interim-dir",
        type=str,
        default=None,
        help="Path to interim cleaned data directory (default: project_root/data/interim).",
    )
    parser.add_argument(
        "--save-report",
        action="store_true",
        help="Save markdown report to docs/data_quality_report.md by default.",
    )
    parser.add_argument(
        "--report-path",
        type=str,
        default=None,
        help="Custom report output path (default: project_root/docs/data_quality_report.md).",
    )
    parser.add_argument(
        "--force-raw",
        action="store_true",
        help="Force validation from raw DAT files even if cleaned CSV files exist.",
    )

    args = parser.parse_args()

    raw_dir, interim_dir, report_path = resolve_paths(args.raw_dir, args.interim_dir, args.report_path)
    tables, source_label = load_datasets(raw_dir=raw_dir, interim_dir=interim_dir, force_raw=args.force_raw)
    ensure_required_columns(tables)

    report_text, summary = build_data_quality_report(tables=tables, source_label=source_label)

    print(report_text)
    print()
    print_summary_to_terminal(summary)

    if args.save_report:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report_text, encoding="utf-8")
        print(f"Saved markdown report to: {report_path}")

    return report_text, summary


if __name__ == "__main__":
    main()

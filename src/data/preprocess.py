"""Preprocess MovieLens tables into training-ready artifacts."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
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


@dataclass
class PreprocessConfig:
    """Config values for preprocessing and optional filtering."""

    min_user_interactions: int = 0
    min_movie_interactions: int = 0
    merge_metadata: bool = False
    encode_ids: bool = True
    keep_duplicate_strategy: str = "latest"


def resolve_paths(
    raw_dir: str | None,
    interim_dir: str | None,
    output_dir: str | None,
) -> tuple[Path, Path, Path]:
    """Resolve project-relative defaults for raw, interim, and output folders."""
    project_root = Path(__file__).resolve().parents[2]
    resolved_raw_dir = Path(raw_dir) if raw_dir else project_root / "data" / "raw"
    resolved_interim_dir = Path(interim_dir) if interim_dir else project_root / "data" / "interim"
    resolved_output_dir = Path(output_dir) if output_dir else project_root / "data" / "processed"
    return resolved_raw_dir, resolved_interim_dir, resolved_output_dir


def load_input_tables(raw_dir: Path, interim_dir: Path, from_raw: bool) -> Dict[str, pd.DataFrame]:
    """Load tables from either raw DAT files or cleaned interim CSV files."""
    if from_raw:
        file_paths = validate_input_files(raw_dir)
        tables = {
            "ratings": load_table(file_paths["ratings"], RATINGS_COLUMNS),
            "movies": load_table(file_paths["movies"], MOVIES_COLUMNS),
            "users": load_table(file_paths["users"], USERS_COLUMNS),
        }
    else:
        cleaned_paths = {
            "ratings": interim_dir / "ratings_cleaned.csv",
            "movies": interim_dir / "movies_cleaned.csv",
            "users": interim_dir / "users_cleaned.csv",
        }
        missing = [str(path) for path in cleaned_paths.values() if not path.exists()]
        if missing:
            missing_text = "\n".join(missing)
            raise FileNotFoundError(
                "Missing cleaned CSV files in data/interim. Run load_data.py first or use --from-raw.\n"
                f"{missing_text}"
            )

        tables = {
            "ratings": pd.read_csv(cleaned_paths["ratings"]),
            "movies": pd.read_csv(cleaned_paths["movies"]),
            "users": pd.read_csv(cleaned_paths["users"]),
        }

    return {name: canonicalize_dataframe_columns(df) for name, df in tables.items()}


def cast_tables(tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """Cast columns to stable dtypes before filtering and joining."""
    ratings_df = tables["ratings"].copy()
    movies_df = tables["movies"].copy()
    users_df = tables["users"].copy()

    ratings_df["user_id"] = pd.to_numeric(ratings_df["user_id"], errors="coerce")
    ratings_df["movie_id"] = pd.to_numeric(ratings_df["movie_id"], errors="coerce")
    ratings_df["rating"] = pd.to_numeric(ratings_df["rating"], errors="coerce")
    ratings_df["timestamp"] = pd.to_numeric(ratings_df["timestamp"], errors="coerce")
    ratings_df["timestamp_dt"] = pd.to_datetime(ratings_df["timestamp"], unit="s", errors="coerce", utc=True)

    movies_df["movie_id"] = pd.to_numeric(movies_df["movie_id"], errors="coerce")
    movies_df["title"] = movies_df["title"].astype("string")
    movies_df["genres"] = movies_df["genres"].astype("string")

    users_df["user_id"] = pd.to_numeric(users_df["user_id"], errors="coerce")
    if "gender" in users_df.columns:
        users_df["gender"] = users_df["gender"].astype("string")
    if "age" in users_df.columns:
        users_df["age"] = pd.to_numeric(users_df["age"], errors="coerce")
    if "occupation" in users_df.columns:
        users_df["occupation"] = pd.to_numeric(users_df["occupation"], errors="coerce")
    if "zip_code" in users_df.columns:
        users_df["zip_code"] = users_df["zip_code"].astype("string")

    return {
        "ratings": ratings_df,
        "movies": movies_df,
        "users": users_df,
    }


def filter_faulty_rows(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    users_df: pd.DataFrame,
    duplicate_strategy: str = "latest",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Drop malformed rows and keep the audit counts needed for reporting."""
    original_row_count = len(ratings_df)

    if duplicate_strategy not in {"latest", "first"}:
        raise ValueError(f"Unsupported duplicate strategy: {duplicate_strategy}")

    ratings_df = ratings_df.dropna(subset=["user_id", "movie_id", "rating", "timestamp", "timestamp_dt"])
    rows_after_null_drop = len(ratings_df)
    ratings_df = ratings_df[
        ratings_df["user_id"].gt(0)
        & ratings_df["movie_id"].gt(0)
        & ratings_df["rating"].between(1, 5)
        & ratings_df["timestamp"].gt(0)
    ]
    rows_after_basic_validation = len(ratings_df)

    movie_id_set = set(movies_df["movie_id"].dropna().astype(int).tolist())
    user_id_set = set(users_df["user_id"].dropna().astype(int).tolist())

    ratings_df["user_id"] = ratings_df["user_id"].astype(int)
    ratings_df["movie_id"] = ratings_df["movie_id"].astype(int)
    ratings_df["timestamp"] = ratings_df["timestamp"].astype("int64")
    ratings_df["rating"] = ratings_df["rating"].astype("float32")

    known_reference_mask = ratings_df["user_id"].isin(user_id_set) & ratings_df["movie_id"].isin(movie_id_set)
    ratings_df = ratings_df.loc[known_reference_mask].copy()
    rows_after_reference_check = len(ratings_df)

    ratings_df = ratings_df.sort_values(["user_id", "movie_id", "timestamp"])
    duplicate_keep = "last" if duplicate_strategy == "latest" else "first"
    duplicate_mask = ratings_df.duplicated(subset=["user_id", "movie_id"], keep=duplicate_keep)
    duplicate_rows = int(duplicate_mask.sum())
    ratings_df = ratings_df.loc[~duplicate_mask].copy()

    summary = {
        "ratings_rows_before_filter": original_row_count,
        "ratings_rows_after_filter": len(ratings_df),
        "ratings_rows_removed_null": original_row_count - rows_after_null_drop,
        "ratings_rows_removed_invalid_values": rows_after_null_drop - rows_after_basic_validation,
        "ratings_rows_removed_unknown_reference": rows_after_basic_validation - rows_after_reference_check,
        "ratings_duplicate_user_movie_rows_removed": duplicate_rows,
        "duplicate_strategy": duplicate_strategy,
    }
    return ratings_df, summary


def iterative_frequency_filter(
    ratings_df: pd.DataFrame,
    min_user_interactions: int,
    min_movie_interactions: int,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Apply optional user/movie frequency filters until the result stabilizes."""
    if min_user_interactions <= 0 and min_movie_interactions <= 0:
        return ratings_df.copy(), {
            "min_user_interactions": min_user_interactions,
            "min_movie_interactions": min_movie_interactions,
            "ratings_rows_removed_by_frequency_filter": 0,
            "users_removed_by_frequency_filter": 0,
            "movies_removed_by_frequency_filter": 0,
            "frequency_filter_iterations": 0,
        }

    working_df = ratings_df.copy()
    removed_users_total = 0
    removed_movies_total = 0
    iterations = 0

    while True:
        iterations += 1
        start_rows = len(working_df)

        if min_user_interactions > 0:
            users_before = set(working_df["user_id"].unique().tolist())
            user_counts = working_df.groupby("user_id").size()
            keep_users = user_counts[user_counts >= min_user_interactions].index
            users_after = set(working_df.loc[working_df["user_id"].isin(keep_users), "user_id"].unique().tolist())
            removed_users_total += len(users_before - users_after)
            working_df = working_df[working_df["user_id"].isin(keep_users)].copy()

        if min_movie_interactions > 0:
            movies_before = set(working_df["movie_id"].unique().tolist())
            movie_counts = working_df.groupby("movie_id").size()
            keep_movies = movie_counts[movie_counts >= min_movie_interactions].index
            movies_after = set(working_df.loc[working_df["movie_id"].isin(keep_movies), "movie_id"].unique().tolist())
            removed_movies_total += len(movies_before - movies_after)
            working_df = working_df[working_df["movie_id"].isin(keep_movies)].copy()

        if len(working_df) == start_rows:
            break

    return working_df, {
        "min_user_interactions": min_user_interactions,
        "min_movie_interactions": min_movie_interactions,
        "ratings_rows_removed_by_frequency_filter": len(ratings_df) - len(working_df),
        "users_removed_by_frequency_filter": removed_users_total,
        "movies_removed_by_frequency_filter": removed_movies_total,
        "frequency_filter_iterations": iterations,
    }


def build_id_mappings(ratings_df: pd.DataFrame) -> tuple[Dict[int, int], Dict[int, int]]:
    """Create contiguous integer mappings for user and movie ids."""
    user_ids = sorted(ratings_df["user_id"].unique().tolist())
    movie_ids = sorted(ratings_df["movie_id"].unique().tolist())

    user_map = {int(user_id): index for index, user_id in enumerate(user_ids)}
    movie_map = {int(movie_id): index for index, movie_id in enumerate(movie_ids)}
    return user_map, movie_map


def merge_metadata(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    users_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge movie and user metadata into the ratings table when requested."""
    merged_df = ratings_df.merge(movies_df, on="movie_id", how="left", validate="m:1")
    merged_df = merged_df.merge(users_df, on="user_id", how="left", validate="m:1", suffixes=("", "_user"))
    return merged_df


def add_encoded_ids(
    ratings_df: pd.DataFrame,
    user_map: Dict[int, int],
    movie_map: Dict[int, int],
) -> pd.DataFrame:
    """Add zero-based encoded ids for matrix-based models."""
    encoded_df = ratings_df.copy()
    encoded_df["user_idx"] = encoded_df["user_id"].map(user_map).astype("int64")
    encoded_df["movie_idx"] = encoded_df["movie_id"].map(movie_map).astype("int64")
    return encoded_df


def save_tables(tables: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Persist preprocessing outputs as parquet artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for table_name, table_df in tables.items():
        output_path = output_dir / f"{table_name}.parquet"
        table_df.to_parquet(output_path, index=False)
        print(f"Saved {table_name} to: {output_path}")


def save_summary(summary: Dict[str, int | float | str], output_dir: Path) -> None:
    """Write a compact JSON summary for auditability."""
    output_path = output_dir / "preprocess_summary.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f"Saved preprocessing summary to: {output_path}")


def save_mappings(user_map: Dict[int, int], movie_map: Dict[int, int], output_dir: Path) -> None:
    """Persist id mappings as parquet tables for safer reloads."""
    user_mapping_df = pd.DataFrame({"user_id": list(user_map.keys()), "user_idx": list(user_map.values())})
    movie_mapping_df = pd.DataFrame({"movie_id": list(movie_map.keys()), "movie_idx": list(movie_map.values())})

    user_mapping_path = output_dir / "user_mapping.parquet"
    movie_mapping_path = output_dir / "movie_mapping.parquet"

    user_mapping_df.to_parquet(user_mapping_path, index=False)
    movie_mapping_df.to_parquet(movie_mapping_path, index=False)

    print(f"Saved user mappings to: {user_mapping_path}")
    print(f"Saved movie mappings to: {movie_mapping_path}")


def save_report(summary: Dict[str, int | float | str], output_dir: Path) -> None:
    """Write a human-readable markdown report with filter thresholds and impact."""
    project_root = Path(__file__).resolve().parents[2]
    report_dir = project_root / "reports" / "preprocessing"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "preprocess_report.md"
    report_lines = [
        "# Preprocessing Report",
        "",
        "## Cleaning Rules",
        "- Cast id/rating/timestamp fields to numeric types.",
        "- Drop rows with null, non-positive, or out-of-range values.",
        "- Remove ratings whose user/movie ids are missing from the reference tables.",
        f"- Duplicate strategy: {summary['duplicate_strategy']}.",
        (
            "- Keep the latest record when the same user rates the same movie multiple times."
            if summary["duplicate_strategy"] == "latest"
            else "- Keep the first record when the same user rates the same movie multiple times."
        ),
        "",
        "## Optional Frequency Filters",
        f"- min_user_interactions: {summary['min_user_interactions']}",
        f"- min_movie_interactions: {summary['min_movie_interactions']}",
        "- Why: reduce extreme sparsity and stabilize matrix factorization for training.",
        "- Impact: fewer cold-start edges, better signal density, but lower catalog coverage and a stronger popularity bias.",
        "",
        "## Output Impact",
        f"- ratings rows before filter: {summary['ratings_rows_before_filter']}",
        f"- ratings rows after filter: {summary['ratings_rows_after_filter']}",
        f"- ratings rows removed by frequency filter: {summary['ratings_rows_removed_by_frequency_filter']}",
        f"- users removed by frequency filter: {summary['users_removed_by_frequency_filter']}",
        f"- movies removed by frequency filter: {summary['movies_removed_by_frequency_filter']}",
        f"- output ratings rows: {summary['output_rows_ratings']}",
        f"- output users: {summary['output_rows_users']}",
        f"- output movies: {summary['output_rows_movies']}",
        "",
        "## Notes",
        str(summary["filter_note"]),
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Saved preprocessing report to: {report_path}")


def preprocess(
    raw_dir: Path,
    interim_dir: Path,
    output_dir: Path,
    config: PreprocessConfig,
    from_raw: bool,
) -> Dict[str, pd.DataFrame]:
    """Run the full preprocessing workflow and return the generated tables."""
    loaded_tables = load_input_tables(raw_dir, interim_dir, from_raw=from_raw)
    casted_tables = cast_tables(loaded_tables)

    ratings_df, faulty_summary = filter_faulty_rows(
        casted_tables["ratings"],
        casted_tables["movies"],
        casted_tables["users"],
        duplicate_strategy=config.keep_duplicate_strategy,
    )

    filtered_ratings_df, frequency_summary = iterative_frequency_filter(
        ratings_df,
        min_user_interactions=config.min_user_interactions,
        min_movie_interactions=config.min_movie_interactions,
    )

    filtered_movie_ids = set(filtered_ratings_df["movie_id"].unique().tolist())
    filtered_user_ids = set(filtered_ratings_df["user_id"].unique().tolist())

    movies_df = casted_tables["movies"][casted_tables["movies"]["movie_id"].isin(filtered_movie_ids)].copy()
    users_df = casted_tables["users"][casted_tables["users"]["user_id"].isin(filtered_user_ids)].copy()

    processed_ratings_df = filtered_ratings_df.copy()
    if config.merge_metadata:
        processed_ratings_df = merge_metadata(processed_ratings_df, movies_df, users_df)

    user_map, movie_map = build_id_mappings(filtered_ratings_df)
    if config.encode_ids:
        processed_ratings_df = add_encoded_ids(processed_ratings_df, user_map, movie_map)
        movies_df = movies_df.assign(movie_idx=movies_df["movie_id"].map(movie_map).astype("int64"))
        users_df = users_df.assign(user_idx=users_df["user_id"].map(user_map).astype("int64"))

    save_tables(
        {
            "ratings_preprocessed": processed_ratings_df,
            "movies_preprocessed": movies_df,
            "users_preprocessed": users_df,
        },
        output_dir,
    )

    summary = {
        **faulty_summary,
        **frequency_summary,
        "merge_metadata": str(config.merge_metadata),
        "encode_ids": str(config.encode_ids),
        "duplicate_strategy": config.keep_duplicate_strategy,
        "output_rows_ratings": len(processed_ratings_df),
        "output_rows_movies": len(movies_df),
        "output_rows_users": len(users_df),
        "user_count": len(user_map),
        "movie_count": len(movie_map),
        "filter_note": (
            "Frequency filtering is optional and removes sparse users/items to reduce extreme sparsity, "
            "but it can shrink catalog coverage and bias the dataset toward active users and popular movies."
        ),
    }
    save_summary(summary, output_dir)
    save_report(summary, output_dir)

    if config.encode_ids:
        save_mappings(user_map, movie_map, output_dir)

    return {
        "ratings_preprocessed": processed_ratings_df,
        "movies_preprocessed": movies_df,
        "users_preprocessed": users_df,
    }


def main() -> Dict[str, pd.DataFrame]:
    """CLI entrypoint for preprocessing MovieLens data."""
    parser = argparse.ArgumentParser(
        description="Preprocess MovieLens data into training-ready parquet artifacts."
    )
    parser.add_argument("--raw-dir", type=str, default=None, help="Path to raw MovieLens data directory.")
    parser.add_argument(
        "--interim-dir",
        type=str,
        default=None,
        help="Path to cleaned CSV directory (default: project_root/data/interim).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to preprocessing output directory (default: project_root/data/processed).",
    )
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="Read directly from raw DAT files instead of cleaned CSV files.",
    )
    parser.add_argument(
        "--min-user-interactions",
        type=int,
        default=0,
        help="Drop users with fewer than N interactions. Default 0 disables this filter.",
    )
    parser.add_argument(
        "--min-movie-interactions",
        type=int,
        default=0,
        help="Drop movies with fewer than N interactions. Default 0 disables this filter.",
    )
    parser.add_argument(
        "--no-merge-metadata",
        dest="merge_metadata",
        action="store_false",
        help="Do not merge movie/user metadata into the ratings table.",
    )
    parser.add_argument(
        "--no-encode-ids",
        dest="encode_ids",
        action="store_false",
        help="Skip creating contiguous user/movie id mappings.",
    )
    parser.add_argument(
        "--keep-duplicate-strategy",
        choices=["latest", "first"],
        default="latest",
        help="Strategy for duplicate user/movie interactions.",
    )

    parser.set_defaults(merge_metadata=False, encode_ids=True)
    args = parser.parse_args()

    raw_dir, interim_dir, output_dir = resolve_paths(args.raw_dir, args.interim_dir, args.output_dir)
    config = PreprocessConfig(
        min_user_interactions=args.min_user_interactions,
        min_movie_interactions=args.min_movie_interactions,
        merge_metadata=args.merge_metadata,
        encode_ids=args.encode_ids,
        keep_duplicate_strategy=args.keep_duplicate_strategy,
    )

    print("Starting preprocessing pipeline...")
    print(f"- from_raw: {args.from_raw}")
    print(f"- min_user_interactions: {config.min_user_interactions}")
    print(f"- min_movie_interactions: {config.min_movie_interactions}")
    print(f"- merge_metadata: {config.merge_metadata}")
    print(f"- encode_ids: {config.encode_ids}")

    tables = preprocess(
        raw_dir=raw_dir,
        interim_dir=interim_dir,
        output_dir=output_dir,
        config=config,
        from_raw=args.from_raw,
    )

    print("Preprocessed table shapes:")
    for table_name, table_df in tables.items():
        print(f"- {table_name}: {table_df.shape}")

    return tables


if __name__ == "__main__":
    main()

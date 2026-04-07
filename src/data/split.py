"""Split preprocessed MovieLens ratings into train, validation, and test sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd  # type: ignore[import-not-found]


def resolve_paths(
    input_path: str | None,
    output_dir: str | None,
    report_dir: str | None,
) -> tuple[Path, Path, Path]:
    """Resolve project-relative defaults for split input, output, and report locations."""
    project_root = Path(__file__).resolve().parents[2]
    resolved_input_path = Path(input_path) if input_path else project_root / "data" / "processed" / "ratings_preprocessed.parquet"
    resolved_output_dir = Path(output_dir) if output_dir else project_root / "data" / "split"
    resolved_report_dir = Path(report_dir) if report_dir else project_root / "reports" / "splitting"
    return resolved_input_path, resolved_output_dir, resolved_report_dir


def load_ratings(input_path: Path) -> pd.DataFrame:
    """Load the preprocessed ratings table."""
    if not input_path.exists():
        raise FileNotFoundError(f"Missing preprocessed ratings file: {input_path}")
    return pd.read_parquet(input_path)


def choose_time_column(df: pd.DataFrame) -> str:
    """Pick the best available time column for temporal splitting."""
    if "timestamp_dt" in df.columns:
        return "timestamp_dt"
    if "timestamp" in df.columns:
        return "timestamp"
    raise ValueError("Expected either timestamp_dt or timestamp in the ratings table.")


def split_user_history(
    user_df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    time_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split one user's history into train/validation/test rows."""
    user_df = user_df.sort_values([time_column, "movie_id"], kind="mergesort")
    n_rows = len(user_df)

    if n_rows < 3:
        if n_rows == 2:
            train_df = user_df.iloc[:1].copy()
            test_df = user_df.iloc[1:].copy()
            return train_df, pd.DataFrame(columns=user_df.columns), test_df
        return user_df.copy(), pd.DataFrame(columns=user_df.columns), pd.DataFrame(columns=user_df.columns)

    test_count = max(1, int(round(n_rows * test_ratio)))
    val_count = max(1, int(round(n_rows * val_ratio)))

    if test_count + val_count >= n_rows:
        test_count = 1
        val_count = 1

    train_count = n_rows - test_count - val_count
    if train_count < 1:
        train_count = 1
        if test_count > val_count:
            test_count = max(1, test_count - 1)
        else:
            val_count = max(1, val_count - 1)

    train_end = train_count
    val_end = train_count + val_count

    train_df = user_df.iloc[:train_end].copy()
    val_df = user_df.iloc[train_end:val_end].copy()
    test_df = user_df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def split_ratings(
    ratings_df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    min_user_interactions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, int | float | str]]:
    """Split ratings by user and time, keeping at least one train row per eligible user."""
    time_column = choose_time_column(ratings_df)
    if "user_id" not in ratings_df.columns:
        raise ValueError("Expected a user_id column in the ratings table.")

    train_parts = []
    val_parts = []
    test_parts = []

    eligible_users = 0
    low_history_users = 0

    for user_id, user_df in ratings_df.groupby("user_id", sort=False):
        if len(user_df) < min_user_interactions:
            low_history_users += 1
            train_parts.append(user_df.copy())
            continue

        eligible_users += 1
        train_df, val_df, test_df = split_user_history(user_df, val_ratio=val_ratio, test_ratio=test_ratio, time_column=time_column)
        train_parts.append(train_df)
        if not val_df.empty:
            val_parts.append(val_df)
        if not test_df.empty:
            test_parts.append(test_df)

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=ratings_df.columns)
    val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame(columns=ratings_df.columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=ratings_df.columns)

    summary = {
        "time_column": time_column,
        "val_ratio": val_ratio,
        "test_ratio": test_ratio,
        "min_user_interactions": min_user_interactions,
        "eligible_users": eligible_users,
        "low_history_users_kept_in_train": low_history_users,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "total_rows": len(ratings_df),
        "split_method": "per-user temporal holdout",
        "split_note": (
            "Users with fewer than the minimum interaction threshold are kept entirely in train to avoid empty validation/test slices."
        ),
    }

    return train_df, val_df, test_df, summary


def save_outputs(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    """Persist split artifacts as parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    print(f"Saved train split to: {train_path}")
    print(f"Saved val split to: {val_path}")
    print(f"Saved test split to: {test_path}")


def save_summary(summary: Dict[str, int | float | str], output_dir: Path) -> None:
    """Write a compact JSON split summary."""
    output_path = output_dir / "split_summary.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f"Saved split summary to: {output_path}")


def save_report(summary: Dict[str, int | float | str], report_dir: Path) -> None:
    """Write a human-readable split report."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "split_report.md"
    report_lines = [
        "# Split Report",
        "",
        "## Method",
        f"- split_method: {summary['split_method']}",
        f"- time_column: {summary['time_column']}",
        f"- val_ratio: {summary['val_ratio']}",
        f"- test_ratio: {summary['test_ratio']}",
        f"- min_user_interactions: {summary['min_user_interactions']}",
        "- Why: temporal splitting reduces leakage for recommender evaluation.",
        "",
        "## Dataset Size",
        f"- total rows: {summary['total_rows']}",
        f"- train rows: {summary['train_rows']}",
        f"- val rows: {summary['val_rows']}",
        f"- test rows: {summary['test_rows']}",
        f"- eligible users: {summary['eligible_users']}",
        f"- low-history users kept in train: {summary['low_history_users_kept_in_train']}",
        "",
        "## Notes",
        str(summary["split_note"]),
    ]
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"Saved split report to: {report_path}")


def run_split(
    input_path: Path,
    output_dir: Path,
    report_dir: Path,
    val_ratio: float,
    test_ratio: float,
    min_user_interactions: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, split, and persist the dataset."""
    ratings_df = load_ratings(input_path)
    train_df, val_df, test_df, summary = split_ratings(
        ratings_df=ratings_df,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        min_user_interactions=min_user_interactions,
    )

    save_outputs(train_df, val_df, test_df, output_dir)
    save_summary(summary, output_dir)
    save_report(summary, report_dir)

    return train_df, val_df, test_df


def main() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """CLI entrypoint for temporal user-wise splitting."""
    parser = argparse.ArgumentParser(description="Split preprocessed MovieLens ratings into train/val/test sets.")
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="Path to preprocessed ratings parquet (default: data/processed/ratings_preprocessed.parquet).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for split parquet outputs (default: data/split).",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory for split report (default: reports/splitting).",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation ratio per user for eligible users.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test ratio per user for eligible users.",
    )
    parser.add_argument(
        "--min-user-interactions",
        type=int,
        default=3,
        help="Users below this threshold stay entirely in train.",
    )

    args = parser.parse_args()
    input_path, output_dir, report_dir = resolve_paths(args.input_path, args.output_dir, args.report_dir)

    print("Starting split pipeline...")
    print(f"- input_path: {input_path}")
    print(f"- val_ratio: {args.val_ratio}")
    print(f"- test_ratio: {args.test_ratio}")
    print(f"- min_user_interactions: {args.min_user_interactions}")

    return run_split(
        input_path=input_path,
        output_dir=output_dir,
        report_dir=report_dir,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        min_user_interactions=args.min_user_interactions,
    )


if __name__ == "__main__":
    main()
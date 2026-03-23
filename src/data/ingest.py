"""Build processed parquet datasets from cleaned CSVs or raw MovieLens DAT files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

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


def resolve_dirs(interim_dir: str | None, raw_dir: str | None, output_dir: str | None) -> tuple[Path, Path, Path]:
    """Resolve default project paths and allow CLI overrides."""
    project_root = Path(__file__).resolve().parents[2]
    resolved_interim_dir = Path(interim_dir) if interim_dir else project_root / "data" / "interim"
    resolved_raw_dir = Path(raw_dir) if raw_dir else project_root / "data" / "raw"
    resolved_output_dir = Path(output_dir) if output_dir else project_root / "data" / "processed"
    return resolved_interim_dir, resolved_raw_dir, resolved_output_dir


def load_from_cleaned_csv(interim_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load cleaned CSV tables when available."""
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

    return {
        "ratings": canonicalize_dataframe_columns(pd.read_csv(cleaned_paths["ratings"])),
        "movies": canonicalize_dataframe_columns(pd.read_csv(cleaned_paths["movies"])),
        "users": canonicalize_dataframe_columns(pd.read_csv(cleaned_paths["users"])),
    }


def load_from_raw_dat(raw_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load and normalize MovieLens DAT files directly from data/raw."""
    paths = validate_input_files(raw_dir)

    return {
        "ratings": canonicalize_dataframe_columns(load_table(paths["ratings"], RATINGS_COLUMNS)),
        "movies": canonicalize_dataframe_columns(load_table(paths["movies"], MOVIES_COLUMNS)),
        "users": canonicalize_dataframe_columns(load_table(paths["users"], USERS_COLUMNS)),
    }


def save_parquet_tables(tables: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Write normalized tables to parquet for downstream training and DVC tracking."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for table_name, table_df in tables.items():
        output_path = output_dir / f"{table_name}.parquet"
        table_df.to_parquet(output_path, index=False)
        print(f"Saved {table_name} parquet to: {output_path}")


def main() -> Dict[str, pd.DataFrame]:
    """Entrypoint for creating processed parquet artifacts."""
    parser = argparse.ArgumentParser(
        description="Convert MovieLens cleaned CSVs (or raw DAT files) into processed parquet tables."
    )
    parser.add_argument(
        "--interim-dir",
        type=str,
        default=None,
        help="Path to cleaned CSV directory (default: project_root/data/interim).",
    )
    parser.add_argument(
        "--raw-dir",
        type=str,
        default=None,
        help="Path to raw DAT directory (default: project_root/data/raw).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Path to processed parquet directory (default: project_root/data/processed).",
    )
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="Read directly from raw DAT files instead of cleaned CSV files.",
    )

    args = parser.parse_args()
    interim_dir, raw_dir, output_dir = resolve_dirs(args.interim_dir, args.raw_dir, args.output_dir)

    if args.from_raw:
        print("Loading from raw DAT files...")
        tables = load_from_raw_dat(raw_dir)
    else:
        print("Loading from cleaned CSV files...")
        tables = load_from_cleaned_csv(interim_dir)

    save_parquet_tables(tables, output_dir)

    print("Processed table shapes:")
    for name, df in tables.items():
        print(f"- {name}: {df.shape}")

    return tables


if __name__ == "__main__":
    main()
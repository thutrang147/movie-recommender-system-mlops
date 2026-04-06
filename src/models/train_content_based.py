"""Train a content-based recommender using movie metadata and user profiles."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict

import pandas as pd  # type: ignore[import-not-found]

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.content_based import ContentBasedConfig, ContentBasedRecommender


def resolve_paths(
    train_path: str | None,
    movies_path: str | None,
    model_path: str | None,
    report_dir: str | None,
) -> tuple[Path, Path, Path, Path]:
    """Resolve project-relative defaults for split input and model outputs."""
    project_root = Path(__file__).resolve().parents[2]
    resolved_train_path = Path(train_path) if train_path else project_root / "data" / "split" / "train.parquet"
    resolved_movies_path = Path(movies_path) if movies_path else project_root / "data" / "processed" / "movies_preprocessed.parquet"
    resolved_model_path = Path(model_path) if model_path else project_root / "models" / "personalized" / "content_based_model.pkl"
    resolved_report_dir = Path(report_dir) if report_dir else project_root / "reports" / "personalized"
    return resolved_train_path, resolved_movies_path, resolved_model_path, resolved_report_dir


def load_table(path: Path) -> pd.DataFrame:
    """Load one parquet file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_parquet(path)


def save_bundle(bundle: Dict[str, object], model_path: Path) -> None:
    """Persist the content-based bundle as a pickle artifact."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump(bundle, file)
    print(f"Saved content-based model bundle to: {model_path}")


def save_summary(summary: Dict[str, object], model_path: Path) -> Path:
    """Persist JSON summary alongside the model artifact."""
    summary_path = model_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f"Saved content-based training summary to: {summary_path}")
    return summary_path


def save_report(summary: Dict[str, object], report_dir: Path) -> Path:
    """Write a markdown report for the content-based run."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "content_based_train_report.md"

    lines = [
        "# Content-Based Personalized Model Training Report",
        "",
        "## Config",
        f"- top_k: {summary['config']['top_k']}",
        f"- relevance_threshold: {summary['config']['relevance_threshold']}",
        f"- min_df: {summary['config']['min_df']}",
        f"- max_features: {summary['config']['max_features']}",
        f"- ngram_range: {summary['config']['ngram_range']}",
        "",
        "## Validation Metrics",
        f"- users_evaluated: {summary['validation_metrics']['users_evaluated']}",
        f"- recall_at_k: {summary['validation_metrics']['recall_at_k']:.4f}",
        f"- map_at_k: {summary['validation_metrics']['map_at_k']:.4f}",
        f"- hit_rate_at_k: {summary['validation_metrics']['hit_rate_at_k']:.4f}",
        f"- coverage: {summary['validation_metrics']['coverage']:.4f}",
        "",
        "## Notes",
        str(summary["notes"]),
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved content-based training report to: {report_path}")
    return report_path


def run_training(
    train_path: Path,
    movies_path: Path,
    model_path: Path,
    report_dir: Path,
    config: ContentBasedConfig,
) -> Dict[str, object]:
    """Fit the content-based model and persist artifacts."""
    train_df = load_table(train_path)
    movies_df = load_table(movies_path)

    model = ContentBasedRecommender(config=config).fit(train_df=train_df, movies_df=movies_df)
    validation_metrics = model.ranking_metrics(train_df, top_k=config.top_k)

    bundle = model.to_bundle()
    save_bundle(bundle=bundle, model_path=model_path)

    summary: Dict[str, object] = {
        "algorithm": "content_based_tfidf",
        "config": bundle["config"],
        "validation_metrics": validation_metrics,
        "artifact_path": str(model_path),
        "notes": "Content-based ranking uses TF-IDF over movie title and genres with user preference profiles.",
    }

    save_summary(summary=summary, model_path=model_path)
    save_report(summary=summary, report_dir=report_dir)
    return summary


def main() -> Dict[str, object]:
    """CLI entrypoint for training the content-based personalized model."""
    parser = argparse.ArgumentParser(description="Train a content-based recommender on split data.")
    parser.add_argument("--train-path", type=str, default=None, help="Path to train parquet (default: data/split/train.parquet).")
    parser.add_argument(
        "--movies-path",
        type=str,
        default=None,
        help="Path to movie metadata parquet (default: data/processed/movies_preprocessed.parquet).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to save the pickled model bundle (default: models/personalized/content_based_model.pkl).",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory for training report (default: reports/personalized).",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K cutoff for ranking metrics.")
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=4.0,
        help="Minimum rating to treat an item as relevant.",
    )
    parser.add_argument("--min-df", type=int, default=2, help="Minimum document frequency for TF-IDF.")
    parser.add_argument("--max-features", type=int, default=12000, help="Maximum TF-IDF vocabulary size.")

    args = parser.parse_args()
    train_path, movies_path, model_path, report_dir = resolve_paths(args.train_path, args.movies_path, args.model_path, args.report_dir)

    config = ContentBasedConfig(
        top_k=args.top_k,
        relevance_threshold=args.relevance_threshold,
        min_df=args.min_df,
        max_features=args.max_features,
    )

    print("Starting content-based training pipeline...")
    print(f"- train_path: {train_path}")
    print(f"- movies_path: {movies_path}")
    print(f"- model_path: {model_path}")
    print(f"- config: {config}")

    return run_training(
        train_path=train_path,
        movies_path=movies_path,
        model_path=model_path,
        report_dir=report_dir,
        config=config,
    )


if __name__ == "__main__":
    main()

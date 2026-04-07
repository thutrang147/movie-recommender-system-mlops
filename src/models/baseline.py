"""Train and evaluate simple popularity-based recommenders."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd  # type: ignore[import-not-found]


@dataclass
class BaselineConfig:
    """Configuration for the baseline recommender run."""

    strategies: tuple[str, ...] = ("most_popular", "weighted_popularity")
    top_k: int = 10
    relevance_threshold: float = 4.0


def resolve_paths(
    train_path: str | None,
    val_path: str | None,
    test_path: str | None,
    model_dir: str | None,
    report_dir: str | None,
) -> tuple[Path, Path, Path, Path, Path]:
    """Resolve project-relative defaults for split inputs and outputs."""
    project_root = Path(__file__).resolve().parents[2]
    resolved_train_path = Path(train_path) if train_path else project_root / "data" / "split" / "train.parquet"
    resolved_val_path = Path(val_path) if val_path else project_root / "data" / "split" / "val.parquet"
    resolved_test_path = Path(test_path) if test_path else project_root / "data" / "split" / "test.parquet"
    resolved_model_dir = Path(model_dir) if model_dir else project_root / "models" / "baseline"
    resolved_report_dir = Path(report_dir) if report_dir else project_root / "reports" / "baseline"
    return resolved_train_path, resolved_val_path, resolved_test_path, resolved_model_dir, resolved_report_dir


def load_split(path: Path) -> pd.DataFrame:
    """Load one split parquet file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_parquet(path)


def choose_time_column(df: pd.DataFrame) -> str:
    """Pick the best available time column for sorting and auditing."""
    if "timestamp_dt" in df.columns:
        return "timestamp_dt"
    if "timestamp" in df.columns:
        return "timestamp"
    raise ValueError("Expected either timestamp_dt or timestamp in the split table.")


def average_precision_at_k(recommended_items: Sequence[int], relevant_items: set[int], top_k: int) -> float:
    """Compute Average Precision at K."""
    if not relevant_items:
        return 0.0

    hits = 0
    precision_sum = 0.0

    for rank, item_id in enumerate(recommended_items[:top_k], start=1):
        if item_id in relevant_items:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / min(len(relevant_items), top_k)


def recall_at_k(recommended_items: Sequence[int], relevant_items: set[int], top_k: int) -> float:
    """Compute Recall at K."""
    if not relevant_items:
        return 0.0

    hits = len(set(recommended_items[:top_k]) & relevant_items)
    return hits / len(relevant_items)


class PopularityBaseline:
    """Popularity-based ranking model."""

    def __init__(self, strategy: str):
        if strategy not in {"most_popular", "weighted_popularity"}:
            raise ValueError(f"Unsupported strategy: {strategy}")
        self.strategy = strategy
        self.item_stats: pd.DataFrame | None = None
        self.item_order: List[int] = []
        self.item_scores: Dict[int, float] = {}

    def fit(self, train_df: pd.DataFrame) -> "PopularityBaseline":
        """Compute item popularity scores from the training split."""
        if "movie_id" not in train_df.columns:
            raise ValueError("Expected a movie_id column in the training split.")
        if "rating" not in train_df.columns:
            raise ValueError("Expected a rating column in the training split.")

        time_column = choose_time_column(train_df)

        item_stats = (
            train_df.groupby("movie_id", as_index=False)
            .agg(
                interaction_count=("rating", "size"),
                mean_rating=("rating", "mean"),
                rating_sum=("rating", "sum"),
                last_seen=(time_column, "max"),
            )
            .copy()
        )

        if self.strategy == "most_popular":
            item_stats["score"] = item_stats["interaction_count"].astype(float)
        else:
            item_stats["score"] = item_stats["mean_rating"].astype(float) * item_stats["interaction_count"].map(math.log1p)

        item_stats = item_stats.sort_values(["score", "interaction_count", "movie_id"], ascending=[False, False, True])
        self.item_stats = item_stats.reset_index(drop=True)
        self.item_order = self.item_stats["movie_id"].astype(int).tolist()
        self.item_scores = dict(zip(self.item_stats["movie_id"].astype(int).tolist(), self.item_stats["score"].astype(float).tolist()))
        return self

    def recommend(self, seen_items: Iterable[int], top_k: int) -> List[int]:
        """Recommend the top-K unseen items."""
        seen_set = set(int(item_id) for item_id in seen_items)
        recommendations: List[int] = []

        for item_id in self.item_order:
            if item_id in seen_set:
                continue
            recommendations.append(int(item_id))
            if len(recommendations) >= top_k:
                break

        return recommendations


def evaluate_split(
    model: PopularityBaseline,
    eval_df: pd.DataFrame,
    train_df: pd.DataFrame,
    top_k: int,
    relevance_threshold: float,
) -> Dict[str, float | int]:
    """Evaluate a baseline on one holdout split."""
    if eval_df.empty:
        return {
            "users_evaluated": 0,
            "recall_at_k": 0.0,
            "map_at_k": 0.0,
            "coverage": 0.0,
            "hit_rate_at_k": 0.0,
        }

    train_user_items = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    relevant_df = eval_df[eval_df["rating"] >= relevance_threshold]
    relevant_user_items = relevant_df.groupby("user_id")["movie_id"].apply(set).to_dict()

    users = [user_id for user_id in relevant_user_items.keys() if user_id in train_user_items]
    if not users:
        return {
            "users_evaluated": 0,
            "recall_at_k": 0.0,
            "map_at_k": 0.0,
            "coverage": 0.0,
            "hit_rate_at_k": 0.0,
        }

    recalls: List[float] = []
    aps: List[float] = []
    hits: List[int] = []
    recommended_pool: set[int] = set()

    for user_id in users:
        seen_items = train_user_items.get(user_id, set())
        relevant_items = relevant_user_items[user_id]
        recommendations = model.recommend(seen_items=seen_items, top_k=top_k)
        recommended_pool.update(recommendations)

        recalls.append(recall_at_k(recommendations, relevant_items, top_k))
        aps.append(average_precision_at_k(recommendations, relevant_items, top_k))
        hits.append(int(len(set(recommendations[:top_k]) & relevant_items) > 0))

    coverage = len(recommended_pool) / max(len(model.item_order), 1)

    return {
        "users_evaluated": len(users),
        "recall_at_k": float(sum(recalls) / len(recalls)),
        "map_at_k": float(sum(aps) / len(aps)),
        "coverage": float(coverage),
        "hit_rate_at_k": float(sum(hits) / len(hits)),
    }


def save_strategy_artifact(model: PopularityBaseline, model_dir: Path) -> Path:
    """Persist a baseline item ranking table for inspection and reuse."""
    model_dir.mkdir(parents=True, exist_ok=True)
    if model.item_stats is None:
        raise RuntimeError("Model must be fit before saving artifacts.")

    artifact_path = model_dir / f"{model.strategy}_items.parquet"
    model.item_stats.to_parquet(artifact_path, index=False)
    return artifact_path


def save_summary(summary: Dict[str, object], model_dir: Path) -> Path:
    """Persist a JSON summary for the full baseline run."""
    model_dir.mkdir(parents=True, exist_ok=True)
    summary_path = model_dir / "baseline_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    return summary_path


def save_report(summary: Dict[str, object], report_dir: Path) -> Path:
    """Persist a markdown report of baseline results."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "baseline_report.md"

    lines = [
        "# Baseline Recommender Report",
        "",
        "## Config",
        f"- top_k: {summary['top_k']}",
        f"- relevance_threshold: {summary['relevance_threshold']}",
        f"- selected_strategy: {summary['selected_strategy']}",
        f"- selection_metric: {summary['selection_metric']}",
        "",
        "## Validation Metrics",
    ]

    for strategy_name, metrics in summary["validation_metrics"].items():
        lines.extend(
            [
                f"### {strategy_name}",
                f"- users_evaluated: {metrics['users_evaluated']}",
                f"- recall_at_k: {metrics['recall_at_k']:.4f}",
                f"- map_at_k: {metrics['map_at_k']:.4f}",
                f"- hit_rate_at_k: {metrics['hit_rate_at_k']:.4f}",
                f"- coverage: {metrics['coverage']:.4f}",
                "",
            ]
        )

    lines.extend(
        [
            "## Test Metrics",
        ]
    )

    for strategy_name, metrics in summary["test_metrics"].items():
        lines.extend(
            [
                f"### {strategy_name}",
                f"- users_evaluated: {metrics['users_evaluated']}",
                f"- recall_at_k: {metrics['recall_at_k']:.4f}",
                f"- map_at_k: {metrics['map_at_k']:.4f}",
                f"- hit_rate_at_k: {metrics['hit_rate_at_k']:.4f}",
                f"- coverage: {metrics['coverage']:.4f}",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            str(summary["notes"]),
        ]
    )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def run_baseline(
    train_path: Path,
    val_path: Path,
    test_path: Path,
    model_dir: Path,
    report_dir: Path,
    config: BaselineConfig,
) -> Dict[str, object]:
    """Train, evaluate, and persist baseline recommenders."""
    train_df = load_split(train_path)
    val_df = load_split(val_path)
    test_df = load_split(test_path)

    validation_metrics: Dict[str, Dict[str, float | int]] = {}
    test_metrics: Dict[str, Dict[str, float | int]] = {}
    saved_artifacts: Dict[str, str] = {}

    for strategy in config.strategies:
        model = PopularityBaseline(strategy=strategy).fit(train_df)
        validation_metrics[strategy] = evaluate_split(
            model=model,
            eval_df=val_df,
            train_df=train_df,
            top_k=config.top_k,
            relevance_threshold=config.relevance_threshold,
        )
        test_metrics[strategy] = evaluate_split(
            model=model,
            eval_df=test_df,
            train_df=train_df,
            top_k=config.top_k,
            relevance_threshold=config.relevance_threshold,
        )
        artifact_path = save_strategy_artifact(model, model_dir)
        saved_artifacts[strategy] = str(artifact_path)

    selection_metric = "map_at_k"
    selected_strategy = max(validation_metrics, key=lambda strategy: validation_metrics[strategy][selection_metric])

    summary: Dict[str, object] = {
        "top_k": config.top_k,
        "relevance_threshold": config.relevance_threshold,
        "selected_strategy": selected_strategy,
        "selection_metric": selection_metric,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "saved_artifacts": saved_artifacts,
        "notes": (
            "Most Popular ranks by item frequency; Weighted Popularity ranks by mean rating multiplied by log1p(count). "
            "Selection uses validation MAP@K and final reporting includes both validation and test metrics."
        ),
    }

    save_summary(summary, model_dir)
    save_report(summary, report_dir)
    return summary


def main() -> Dict[str, object]:
    """CLI entrypoint for the popularity baseline trainer."""
    parser = argparse.ArgumentParser(description="Train and evaluate popularity-based recommender baselines.")
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Path to train parquet (default: data/split/train.parquet).",
    )
    parser.add_argument(
        "--val-path",
        type=str,
        default=None,
        help="Path to validation parquet (default: data/split/val.parquet).",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Path to test parquet (default: data/split/test.parquet).",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory for baseline artifacts (default: models/baseline).",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory for baseline report (default: reports/baseline).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top-K cutoff for evaluation and recommendation generation.",
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=4.0,
        help="Minimum rating to count an item as relevant in ranking metrics.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=["most_popular", "weighted_popularity"],
        default=["most_popular", "weighted_popularity"],
        help="Which baseline strategies to train.",
    )

    args = parser.parse_args()
    train_path, val_path, test_path, model_dir, report_dir = resolve_paths(
        args.train_path,
        args.val_path,
        args.test_path,
        args.model_dir,
        args.report_dir,
    )

    config = BaselineConfig(
        strategies=tuple(args.strategies),
        top_k=args.top_k,
        relevance_threshold=args.relevance_threshold,
    )

    print("Starting baseline training pipeline...")
    print(f"- train_path: {train_path}")
    print(f"- val_path: {val_path}")
    print(f"- test_path: {test_path}")
    print(f"- strategies: {config.strategies}")
    print(f"- top_k: {config.top_k}")
    print(f"- relevance_threshold: {config.relevance_threshold}")

    return run_baseline(
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        model_dir=model_dir,
        report_dir=report_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
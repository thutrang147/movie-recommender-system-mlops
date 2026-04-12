"""Train a personalized SVD recommender on the training split."""

from __future__ import annotations

import argparse
import json
import pickle
from itertools import product
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd  # type: ignore[import-not-found]
from surprise import Dataset, Reader, SVD
from surprise import accuracy


@dataclass
class TrainConfig:
    """Configuration for the personalized SVD training run."""

    n_factors: int = 100
    n_epochs: int = 25
    lr_all: float = 0.005
    reg_all: float = 0.02
    random_state: int = 42
    top_k: int = 10
    relevance_threshold: float = 4.0


DEFAULT_SEARCH_SPACE = {
    "n_factors": (20, 50, 100, 150),
    "n_epochs": (20, 40, 60),
    "lr_all": (0.002, 0.003, 0.005, 0.007),
    "reg_all": (0.02, 0.05, 0.08),
}


def resolve_paths(
    train_path: str | None,
    val_path: str | None,
    model_path: str | None,
    report_dir: str | None,
) -> tuple[Path, Path, Path, Path]:
    """Resolve project-relative defaults for split input and model outputs."""
    project_root = Path(__file__).resolve().parents[2]
    resolved_train_path = Path(train_path) if train_path else project_root / "data" / "split" / "train.parquet"
    resolved_val_path = Path(val_path) if val_path else project_root / "data" / "split" / "val.parquet"
    resolved_model_path = Path(model_path) if model_path else project_root / "models" / "personalized" / "svd_model.pkl"
    resolved_report_dir = Path(report_dir) if report_dir else project_root / "reports" / "personalized"
    return resolved_train_path, resolved_val_path, resolved_model_path, resolved_report_dir


def load_split(path: Path) -> pd.DataFrame:
    """Load one split parquet file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_parquet(path)


def choose_time_column(df: pd.DataFrame) -> str:
    """Pick the best available time column for temporal sorting."""
    if "timestamp_dt" in df.columns:
        return "timestamp_dt"
    if "timestamp" in df.columns:
        return "timestamp"
    raise ValueError("Expected either timestamp_dt or timestamp in the split table.")


def build_surprise_dataset(train_df: pd.DataFrame) -> Dataset:
    """Convert the training DataFrame into a Surprise dataset."""
    required_columns = {"user_id", "movie_id", "rating"}
    missing = required_columns - set(train_df.columns)
    if missing:
        raise ValueError(f"Training data is missing required columns: {', '.join(sorted(missing))}")

    surprise_df = train_df.loc[:, ["user_id", "movie_id", "rating"]].copy()
    surprise_df["user_id"] = surprise_df["user_id"].astype(str)
    surprise_df["movie_id"] = surprise_df["movie_id"].astype(str)

    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(surprise_df, reader)


def build_item_stats(train_df: pd.DataFrame) -> pd.DataFrame:
    """Build item popularity statistics for recommendations and cold-start fallback."""
    time_column = choose_time_column(train_df)
    item_stats = (
        train_df.groupby("movie_id", as_index=False)
        .agg(
            interaction_count=("rating", "size"),
            mean_rating=("rating", "mean"),
            last_seen=(time_column, "max"),
        )
        .copy()
    )
    item_stats["popularity_score"] = item_stats["interaction_count"].astype(float)
    item_stats = item_stats.sort_values(["popularity_score", "mean_rating", "movie_id"], ascending=[False, False, True])
    return item_stats.reset_index(drop=True)


def build_user_seen_items(train_df: pd.DataFrame) -> Dict[int, set[int]]:
    """Map each user to the set of items already seen in training."""
    return {int(user_id): set(map(int, group["movie_id"].tolist())) for user_id, group in train_df.groupby("user_id")}


def build_candidate_configs(base_config: TrainConfig) -> List[TrainConfig]:
    """Create a compact validation search space around the default config."""
    candidates: List[TrainConfig] = []
    for n_factors, n_epochs, lr_all, reg_all in product(
        DEFAULT_SEARCH_SPACE["n_factors"],
        DEFAULT_SEARCH_SPACE["n_epochs"],
        DEFAULT_SEARCH_SPACE["lr_all"],
        DEFAULT_SEARCH_SPACE["reg_all"],
    ):
        candidates.append(
            TrainConfig(
                n_factors=n_factors,
                n_epochs=n_epochs,
                lr_all=lr_all,
                reg_all=reg_all,
                random_state=base_config.random_state,
                top_k=base_config.top_k,
                relevance_threshold=base_config.relevance_threshold,
            )
        )

    if base_config not in candidates:
        candidates.append(base_config)
    return candidates


def fit_and_score_candidate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    candidate: TrainConfig,
) -> tuple[SVD, Dict[str, float | int]]:
    """Fit one SVD candidate and score it on validation data."""
    train_data = build_surprise_dataset(train_df)
    trainset = train_data.build_full_trainset()

    algo = SVD(
        n_factors=candidate.n_factors,
        n_epochs=candidate.n_epochs,
        lr_all=candidate.lr_all,
        reg_all=candidate.reg_all,
        random_state=candidate.random_state,
    )
    algo.fit(trainset)

    validation_records = [
        (str(row.user_id), str(row.movie_id), float(row.rating))
        for row in val_df.itertuples(index=False)
        if pd.notna(row.rating)
    ]
    validation_predictions = algo.test(validation_records) if validation_records else []
    rmse = accuracy.rmse(validation_predictions, verbose=False) if validation_predictions else 0.0

    train_user_seen_items = build_user_seen_items(train_df)
    item_stats = build_item_stats(train_df)
    item_ids = item_stats["movie_id"].astype(int).tolist()

    ranking = ranking_metrics(
        algo=algo,
        eval_df=val_df,
        train_user_seen_items=train_user_seen_items,
        item_ids=item_ids,
        top_k=candidate.top_k,
        relevance_threshold=candidate.relevance_threshold,
    )

    metrics: Dict[str, float | int] = {"rmse": rmse, **ranking}
    return algo, metrics


def select_best_candidate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    base_config: TrainConfig,
) -> tuple[TrainConfig, Dict[str, Dict[str, float | int]]]:
    """Evaluate a compact SVD grid and pick the best validation MAP@K result."""
    candidates = build_candidate_configs(base_config)
    results: Dict[str, Dict[str, float | int]] = {}
    best_config = base_config
    best_metrics: Dict[str, float | int] | None = None

    for candidate in candidates:
        _, metrics = fit_and_score_candidate(train_df=train_df, val_df=val_df, candidate=candidate)
        candidate_key = (
            f"n_factors={candidate.n_factors},n_epochs={candidate.n_epochs},"
            f"lr_all={candidate.lr_all},reg_all={candidate.reg_all}"
        )
        results[candidate_key] = metrics

        if best_metrics is None:
            best_config = candidate
            best_metrics = metrics
            continue

        current_score = (
            float(metrics["map_at_k"]),
            float(metrics["recall_at_k"]),
            -float(metrics["rmse"]),
        )
        best_score = (
            float(best_metrics["map_at_k"]),
            float(best_metrics["recall_at_k"]),
            -float(best_metrics["rmse"]),
        )
        if current_score > best_score:
            best_config = candidate
            best_metrics = metrics

    return best_config, results


def recommend_for_user(
    algo: SVD,
    user_id: int,
    item_ids: Iterable[int],
    seen_items: set[int],
    top_k: int,
) -> pd.DataFrame:
    """Rank unseen items for one user."""
    rows = []
    user_raw_id = str(user_id)

    for movie_id in item_ids:
        movie_int = int(movie_id)
        if movie_int in seen_items:
            continue
        estimate = float(algo.predict(user_raw_id, str(movie_int)).est)
        rows.append({"movie_id": movie_int, "score": estimate})

    if not rows:
        return pd.DataFrame(columns=["movie_id", "score", "rank"])

    recommendations = pd.DataFrame(rows).sort_values(["score", "movie_id"], ascending=[False, True]).head(top_k).copy()
    recommendations["rank"] = range(1, len(recommendations) + 1)
    return recommendations.loc[:, ["rank", "movie_id", "score"]]


def recommend_with_bundle(bundle: Dict[str, object], user_id: int, top_k: int) -> List[Tuple[int, float]]:
    """Generate SVD recommendations directly from a serialized bundle."""
    algo: SVD = bundle["model"]  # type: ignore[assignment]
    train_user_seen_items: Dict[int, set[int]] = bundle["train_user_seen_items"]  # type: ignore[assignment]
    item_ids: List[int] = bundle["item_ids"]  # type: ignore[assignment]
    item_popularity_order: List[int] = bundle["item_popularity_order"]  # type: ignore[assignment]

    if top_k <= 0:
        return []

    seen_items = train_user_seen_items.get(user_id, set())
    
    # Get predictions for all unseen items
    predictions = []
    for movie_id in item_ids:
        if movie_id in seen_items:
            continue
        estimate = float(algo.predict(str(user_id), str(movie_id)).est)
        predictions.append((movie_id, estimate))
    
    # If no predictions (user not in training data), fall back to popularity
    if not predictions:
        fallback = item_popularity_order[:top_k]
        return [(movie_id, float(top_k - rank)) for rank, movie_id in enumerate(fallback)]
    
    # Sort by score descending and return top_k
    predictions.sort(key=lambda x: x[1], reverse=True)
    return predictions[:top_k]


def ranking_metrics(
    algo: SVD,
    eval_df: pd.DataFrame,
    train_user_seen_items: Dict[int, set[int]],
    item_ids: List[int],
    top_k: int,
    relevance_threshold: float,
) -> Dict[str, float | int]:
    """Compute top-K ranking metrics for a holdout split."""
    if eval_df.empty:
        return {
            "users_evaluated": 0,
            "recall_at_k": 0.0,
            "map_at_k": 0.0,
            "hit_rate_at_k": 0.0,
            "coverage": 0.0,
        }

    relevant_df = eval_df[eval_df["rating"] >= relevance_threshold]
    relevant_user_items = relevant_df.groupby("user_id")["movie_id"].apply(lambda values: set(map(int, values.tolist()))).to_dict()
    users = [int(user_id) for user_id in relevant_user_items.keys() if int(user_id) in train_user_seen_items]

    if not users:
        return {
            "users_evaluated": 0,
            "recall_at_k": 0.0,
            "map_at_k": 0.0,
            "hit_rate_at_k": 0.0,
            "coverage": 0.0,
        }

    recalls: List[float] = []
    aps: List[float] = []
    hits: List[int] = []
    recommended_pool: set[int] = set()

    for user_id in users:
        seen_items = train_user_seen_items.get(user_id, set())
        candidate_rows = []
        for movie_id in item_ids:
            movie_int = int(movie_id)
            if movie_int in seen_items:
                continue
            candidate_rows.append((movie_int, float(algo.predict(str(user_id), str(movie_int)).est)))

        candidate_rows.sort(key=lambda item: (-item[1], item[0]))
        top_items = [movie_id for movie_id, _ in candidate_rows[:top_k]]
        recommended_pool.update(top_items)

        relevant_items = relevant_user_items[user_id]
        hit_count = len(set(top_items) & relevant_items)
        recall = hit_count / len(relevant_items) if relevant_items else 0.0

        hits_so_far = 0
        precision_sum = 0.0
        for rank, movie_id in enumerate(top_items, start=1):
            if movie_id in relevant_items:
                hits_so_far += 1
                precision_sum += hits_so_far / rank

        ap = precision_sum / min(len(relevant_items), top_k) if relevant_items else 0.0

        recalls.append(recall)
        aps.append(ap)
        hits.append(int(hit_count > 0))

    coverage = len(recommended_pool) / max(len(item_ids), 1)
    return {
        "users_evaluated": len(users),
        "recall_at_k": float(sum(recalls) / len(recalls)),
        "map_at_k": float(sum(aps) / len(aps)),
        "hit_rate_at_k": float(sum(hits) / len(hits)),
        "coverage": float(coverage),
    }


def save_bundle(bundle: Dict[str, object], model_path: Path) -> None:
    """Persist the full model bundle as a pickle artifact."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump(bundle, file)
    print(f"Saved personalized model bundle to: {model_path}")


def save_summary(summary: Dict[str, object], model_path: Path) -> Path:
    """Persist a JSON summary alongside the model artifact."""
    summary_path = model_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f"Saved training summary to: {summary_path}")
    return summary_path


def save_report(summary: Dict[str, object], report_dir: Path) -> Path:
    """Write a markdown report for the training run."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "train_report.md"

    lines = [
        "# Personalized Model Training Report",
        "",
        "## Config",
        f"- n_factors: {summary['config']['n_factors']}",
        f"- n_epochs: {summary['config']['n_epochs']}",
        f"- lr_all: {summary['config']['lr_all']}",
        f"- reg_all: {summary['config']['reg_all']}",
        f"- top_k: {summary['config']['top_k']}",
        f"- relevance_threshold: {summary['config']['relevance_threshold']}",
        "",
        "## Tuning",
        f"- candidates_evaluated: {len(summary.get('candidate_metrics', {}))}",
        f"- selection_metric: map_at_k",
        "",
        "## Validation Metrics",
        f"- rmse: {summary['validation_metrics']['rmse']:.4f}",
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
    print(f"Saved training report to: {report_path}")
    return report_path


def run_training(
    train_path: Path,
    val_path: Path,
    model_path: Path,
    report_dir: Path,
    config: TrainConfig,
) -> Dict[str, object]:
    """Fit the SVD model and persist artifacts.

    Workflow:
    1) Select hyperparameters on validation split
    2) Refit final model on train+val with selected hyperparameters
    """
    train_df = load_split(train_path)
    val_df = load_split(val_path)
    selected_config, candidate_metrics = select_best_candidate(train_df=train_df, val_df=val_df, base_config=config)

    # Keep validation metrics from train-only fit for fair model selection reporting.
    _, validation_metrics = fit_and_score_candidate(train_df=train_df, val_df=val_df, candidate=selected_config)

    # Refit final model on train+val after selecting hyperparameters.
    final_train_df = pd.concat([train_df, val_df], ignore_index=True)
    final_train_data = build_surprise_dataset(final_train_df)
    final_trainset = final_train_data.build_full_trainset()

    algo = SVD(
        n_factors=selected_config.n_factors,
        n_epochs=selected_config.n_epochs,
        lr_all=selected_config.lr_all,
        reg_all=selected_config.reg_all,
        random_state=selected_config.random_state,
    )
    algo.fit(final_trainset)

    train_user_seen_items = build_user_seen_items(final_train_df)
    item_stats = build_item_stats(final_train_df)
    item_ids = item_stats["movie_id"].astype(int).tolist()

    item_popularity_order = item_stats["movie_id"].astype(int).tolist()
    item_popularity_scores = dict(zip(item_stats["movie_id"].astype(int).tolist(), item_stats["interaction_count"].astype(int).tolist()))

    bundle = {
        "model": algo,
        "config": asdict(selected_config),
        "train_user_seen_items": train_user_seen_items,
        "item_ids": item_ids,
        "item_popularity_order": item_popularity_order,
        "item_popularity_scores": item_popularity_scores,
        "validation_metrics": validation_metrics,
        "candidate_metrics": candidate_metrics,
    }
    save_bundle(bundle, model_path)

    summary: Dict[str, object] = {
        "config": asdict(selected_config),
        "validation_metrics": validation_metrics,
        "candidate_metrics": candidate_metrics,
        "final_fit_rows": int(len(final_train_df)),
        "artifact_path": str(model_path),
        "notes": (
            "The model was selected by validation MAP@K over an expanded SVD search grid, then refit on train+val with the winning hyperparameters."
        ),
    }
    save_summary(summary, model_path)
    save_report(summary, report_dir)
    return summary


def main() -> Dict[str, object]:
    """CLI entrypoint for training the personalized model."""
    parser = argparse.ArgumentParser(description="Train an SVD personalized recommender on the split data.")
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
        "--model-path",
        type=str,
        default=None,
        help="Path to save the pickled model bundle (default: models/personalized/svd_model.pkl).",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory for the training report (default: reports/personalized).",
    )
    parser.add_argument("--n-factors", type=int, default=100, help="Number of latent factors.")
    parser.add_argument("--n-epochs", type=int, default=25, help="Number of training epochs.")
    parser.add_argument("--lr-all", type=float, default=0.005, help="Learning rate for all parameters.")
    parser.add_argument("--reg-all", type=float, default=0.02, help="Regularization for all parameters.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for the SVD model.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K cutoff for ranking metrics.")
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=4.0,
        help="Minimum rating to treat an item as relevant for ranking metrics.",
    )

    args = parser.parse_args()
    train_path, val_path, model_path, report_dir = resolve_paths(args.train_path, args.val_path, args.model_path, args.report_dir)

    config = TrainConfig(
        n_factors=args.n_factors,
        n_epochs=args.n_epochs,
        lr_all=args.lr_all,
        reg_all=args.reg_all,
        random_state=args.random_state,
        top_k=args.top_k,
        relevance_threshold=args.relevance_threshold,
    )

    print("Starting personalized training pipeline...")
    print(f"- train_path: {train_path}")
    print(f"- val_path: {val_path}")
    print(f"- model_path: {model_path}")
    print(f"- config: {config}")

    return run_training(
        train_path=train_path,
        val_path=val_path,
        model_path=model_path,
        report_dir=report_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
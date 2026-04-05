"""Train a personalized item-based KNN recommender on the training split."""

from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List

import numpy as np  # type: ignore[import-not-found]
import pandas as pd  # type: ignore[import-not-found]
from surprise import Dataset, KNNBasic, Reader
from surprise import accuracy


@dataclass
class KNNConfig:
    """Configuration for personalized item-based KNN training."""

    k: int = 80
    min_k: int = 3
    sim_name: str = "cosine"
    top_k: int = 10
    relevance_threshold: float = 4.0
    sample_eval_users: int | None = 1000
    random_state: int = 42


DEFAULT_SEARCH_SPACE = {
    "k": (50, 100),
    "min_k": (1,),
    "sim_name": ("cosine", "pearson_baseline"),
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
    resolved_model_path = Path(model_path) if model_path else project_root / "models" / "personalized" / "knn_model.pkl"
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
    """Convert training rows into Surprise format."""
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
    """Build item popularity stats for fallback ordering and audits."""
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
    """Map each user to the set of items seen in training."""
    return {int(user_id): set(map(int, group["movie_id"].tolist())) for user_id, group in train_df.groupby("user_id")}


def ranking_metrics(
    model: KNNBasic,
    eval_df: pd.DataFrame,
    train_user_seen_items: Dict[int, set[int]],
    item_ids: List[int],
    top_k: int,
    relevance_threshold: float,
    sample_eval_users: int | None = None,
    random_state: int = 42,
) -> Dict[str, float | int]:
    """Compute top-K ranking metrics for one holdout split.
    
    If sample_eval_users is set and number of users exceeds it,
    randomly sample that many users for evaluation (for speed during tuning).
    """
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
    
    # Sample users if requested
    if sample_eval_users is not None and len(users) > sample_eval_users:
        rng = np.random.RandomState(random_state)
        users = rng.choice(users, size=sample_eval_users, replace=False).tolist()

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
            estimate = float(model.predict(str(user_id), str(movie_int)).est)
            candidate_rows.append((movie_int, estimate))

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


def build_candidate_configs(base_config: KNNConfig) -> List[KNNConfig]:
    """Build a compact search space for item-based KNN."""
    candidates: List[KNNConfig] = []
    for k_value, min_k_value, sim_name in product(
        DEFAULT_SEARCH_SPACE["k"],
        DEFAULT_SEARCH_SPACE["min_k"],
        DEFAULT_SEARCH_SPACE["sim_name"],
    ):
        candidates.append(
            KNNConfig(
                k=k_value,
                min_k=min_k_value,
                sim_name=str(sim_name),
                top_k=base_config.top_k,
                relevance_threshold=base_config.relevance_threshold,
                sample_eval_users=base_config.sample_eval_users,
                random_state=base_config.random_state,
            )
        )

    if base_config not in candidates:
        candidates.append(base_config)
    return candidates


def fit_and_score_candidate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    candidate: KNNConfig,
    item_ids: List[int],
    train_user_seen_items: Dict[int, set[int]],
    use_sampling: bool = True,
) -> tuple[KNNBasic, Dict[str, float | int]]:
    """Fit one KNN candidate and score it on validation.
    
    Args:
        item_ids: Precomputed list of all items from training set
        train_user_seen_items: Precomputed dict of users -> seen items
        use_sampling: If True, samples users for faster ranking evaluation
    """
    train_data = build_surprise_dataset(train_df)
    trainset = train_data.build_full_trainset()

    model = KNNBasic(
        k=candidate.k,
        min_k=candidate.min_k,
        sim_options={"name": candidate.sim_name, "user_based": False},
        verbose=False,
    )
    model.fit(trainset)

    # RMSE: only compute on known users (consistent with ranking eval)
    known_users = set(train_user_seen_items.keys())
    val_known = val_df[val_df["user_id"].isin(known_users)]
    validation_records = [
        (str(row.user_id), str(row.movie_id), float(row.rating))
        for row in val_known.itertuples(index=False)
        if pd.notna(row.rating)
    ]
    validation_predictions = model.test(validation_records) if validation_records else []
    rmse = accuracy.rmse(validation_predictions, verbose=False) if validation_predictions else 0.0

    # Use sampling during candidate search, full eval after selection
    sample_users = candidate.sample_eval_users if use_sampling else None
    ranking = ranking_metrics(
        model=model,
        eval_df=val_df,
        train_user_seen_items=train_user_seen_items,
        item_ids=item_ids,
        top_k=candidate.top_k,
        relevance_threshold=candidate.relevance_threshold,
        sample_eval_users=sample_users,
        random_state=candidate.random_state,
    )

    metrics: Dict[str, float | int] = {"rmse": rmse, **ranking}
    return model, metrics


def select_best_candidate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    base_config: KNNConfig,
    item_ids: List[int],
    train_user_seen_items: Dict[int, set[int]],
) -> tuple[KNNConfig, KNNBasic, Dict[str, Dict[str, float | int]], Dict[str, float | int]]:
    """Pick the best KNN candidate by validation MAP@K.
    
    Returns: (best_config, best_model, candidate_metrics_sampled, best_validation_metrics_full)
    
    1. Evaluate all candidates using sampling (if configured) for speed
    2. Select winner by (MAP@K, Recall@K, -RMSE) tuple
    3. Refit winner and recompute metrics on FULL validation set
    """
    candidates = build_candidate_configs(base_config)
    candidate_metrics_sampled: Dict[str, Dict[str, float | int]] = {}
    best_config = base_config
    best_metrics: Dict[str, float | int] | None = None

    total_candidates = len(candidates)
    for idx, candidate in enumerate(candidates, 1):
        candidate_key = f"k={candidate.k},min_k={candidate.min_k},sim_name={candidate.sim_name}"
        print(f"\n[{idx}/{total_candidates}] Evaluating {candidate_key}...")
        
        # Evaluate with sampling during search
        _, metrics = fit_and_score_candidate(
            train_df=train_df,
            val_df=val_df,
            candidate=candidate,
            item_ids=item_ids,
            train_user_seen_items=train_user_seen_items,
            use_sampling=True,
        )
        candidate_metrics_sampled[candidate_key] = metrics
        print(f"  MAP@K: {metrics['map_at_k']:.4f}, Recall@K: {metrics['recall_at_k']:.4f}, RMSE: {metrics['rmse']:.4f}")

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
            print(f"  ✓ New best!")

    # Refit best config on FULL validation set (no sampling)
    best_key = f"k={best_config.k},min_k={best_config.min_k},sim_name={best_config.sim_name}"
    print(f"\nRefitting best config on FULL validation set: {best_key}")
    best_config_full_eval = KNNConfig(
        k=best_config.k,
        min_k=best_config.min_k,
        sim_name=best_config.sim_name,
        top_k=best_config.top_k,
        relevance_threshold=best_config.relevance_threshold,
        sample_eval_users=None,  # Full evaluation
        random_state=best_config.random_state,
    )
    best_model, best_validation_metrics_full = fit_and_score_candidate(
        train_df=train_df,
        val_df=val_df,
        candidate=best_config_full_eval,
        item_ids=item_ids,
        train_user_seen_items=train_user_seen_items,
        use_sampling=False,
    )
    print(f"Full validation - MAP@K: {best_validation_metrics_full['map_at_k']:.4f}, Recall@K: {best_validation_metrics_full['recall_at_k']:.4f}")

    return best_config, best_model, candidate_metrics_sampled, best_validation_metrics_full


def save_bundle(bundle: Dict[str, object], model_path: Path) -> None:
    """Persist the KNN model bundle."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump(bundle, file)
    print(f"Saved KNN model bundle to: {model_path}")


def save_summary(summary: Dict[str, object], model_path: Path) -> Path:
    """Persist a JSON summary alongside the model artifact."""
    summary_path = model_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f"Saved KNN training summary to: {summary_path}")
    return summary_path


def save_report(summary: Dict[str, object], report_dir: Path) -> Path:
    """Write a markdown report for item-based KNN training."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "train_knn_report.md"

    lines = [
        "# Personalized KNN Training Report",
        "",
        "## Config",
        f"- k: {summary['config']['k']}",
        f"- min_k: {summary['config']['min_k']}",
        f"- sim_name: {summary['config']['sim_name']}",
        f"- top_k: {summary['config']['top_k']}",
        f"- relevance_threshold: {summary['config']['relevance_threshold']}",
        "",
        "## Tuning",
        f"- candidates_evaluated: {len(summary.get('candidate_metrics', {}))}",
        "- selection_metric: map_at_k",
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
    print(f"Saved KNN training report to: {report_path}")
    return report_path


def run_training(
    train_path: Path,
    val_path: Path,
    model_path: Path,
    report_dir: Path,
    config: KNNConfig,
) -> Dict[str, object]:
    """Fit the tuned item-based KNN model and persist artifacts."""
    train_df = load_split(train_path)
    val_df = load_split(val_path)

    # Precompute item stats and seen items once
    item_stats = build_item_stats(train_df)
    item_ids = item_stats["movie_id"].astype(int).tolist()
    train_user_seen_items = build_user_seen_items(train_df)

    # Select best config via sampling, get back best model + full validation metrics
    selected_config, model, candidate_metrics_sampled, validation_metrics_full = select_best_candidate(
        train_df=train_df,
        val_df=val_df,
        base_config=config,
        item_ids=item_ids,
        train_user_seen_items=train_user_seen_items,
    )

    bundle = {
        "model": model,
        "model_name": "knn_item",
        "config": asdict(selected_config),
        "train_user_seen_items": train_user_seen_items,
        "item_ids": item_ids,
        "validation_metrics": validation_metrics_full,  # Full validation metrics
        "candidate_metrics": candidate_metrics_sampled,  # For audit trail
    }
    save_bundle(bundle, model_path)

    summary: Dict[str, object] = {
        "config": asdict(selected_config),
        "validation_metrics": validation_metrics_full,  # Full validation metrics
        "candidate_metrics": candidate_metrics_sampled,  # For audit trail
        "artifact_path": str(model_path),
        "notes": "Item-based KNN model selected by validation MAP@K over a compact search grid (sampled during tuning, evaluated on full set).",
    }
    save_summary(summary, model_path)
    save_report(summary, report_dir)
    return summary


def main() -> Dict[str, object]:
    """CLI entrypoint for item-based KNN training."""
    parser = argparse.ArgumentParser(description="Train an item-based KNN personalized recommender on split data.")
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
        help="Path to save the pickled model bundle (default: models/personalized/knn_model.pkl).",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory for training report (default: reports/personalized).",
    )
    parser.add_argument("--k", type=int, default=80, help="Neighborhood size.")
    parser.add_argument("--min-k", type=int, default=3, help="Minimum neighbors for prediction.")
    parser.add_argument(
        "--sim-name",
        type=str,
        default="cosine",
        choices=["cosine", "pearson_baseline"],
        help="Similarity measure for item-based KNN.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Top-K cutoff for ranking metrics.")
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=4.0,
        help="Minimum rating to treat an item as relevant for ranking metrics.",
    )
    parser.add_argument(
        "--sample-eval-users",
        type=int,
        default=1000,
        help="Sample validation users during candidate search. Use 0 or negative to disable sampling (full eval). Default 1000 for speed.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for user sampling.",
    )

    args = parser.parse_args()
    train_path, val_path, model_path, report_dir = resolve_paths(args.train_path, args.val_path, args.model_path, args.report_dir)

    # Convert sample_eval_users: 0 or negative -> None (disable sampling)
    sample_eval_users = args.sample_eval_users if args.sample_eval_users > 0 else None

    config = KNNConfig(
        k=args.k,
        min_k=args.min_k,
        sim_name=args.sim_name,
        top_k=args.top_k,
        relevance_threshold=args.relevance_threshold,
        sample_eval_users=sample_eval_users,
        random_state=args.random_state,
    )

    print("Starting personalized KNN training pipeline...")
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

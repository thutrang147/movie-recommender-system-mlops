"""Evaluate the popularity baseline and personalized SVD recommender on the test split."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd  # type: ignore[import-not-found]


def resolve_paths(
    train_path: str | None,
    test_path: str | None,
    baseline_path: str | None,
    personalized_path: str | None,
    report_dir: str | None,
) -> tuple[Path, Path, Path, Path, Path]:
    """Resolve project-relative defaults for evaluation inputs and outputs."""
    project_root = Path(__file__).resolve().parents[2]
    resolved_train_path = Path(train_path) if train_path else project_root / "data" / "split" / "train.parquet"
    resolved_test_path = Path(test_path) if test_path else project_root / "data" / "split" / "test.parquet"
    resolved_baseline_path = Path(baseline_path) if baseline_path else project_root / "models" / "baseline" / "most_popular_items.parquet"
    resolved_personalized_path = Path(personalized_path) if personalized_path else project_root / "models" / "personalized" / "svd_model.pkl"
    resolved_report_dir = Path(report_dir) if report_dir else project_root / "reports" / "evaluation"
    return (
        resolved_train_path,
        resolved_test_path,
        resolved_baseline_path,
        resolved_personalized_path,
        resolved_report_dir,
    )


def load_split(path: Path) -> pd.DataFrame:
    """Load a split parquet file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_parquet(path)


def load_baseline_items(path: Path) -> pd.DataFrame:
    """Load the popularity baseline ranking table."""
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline artifact: {path}")
    return pd.read_parquet(path)


def load_personalized_bundle(path: Path) -> Dict[str, object]:
    """Load the persisted SVD bundle."""
    if not path.exists():
        raise FileNotFoundError(f"Missing personalized model artifact: {path}")
    with open(path, "rb") as file:
        bundle = pickle.load(file)
    if not isinstance(bundle, dict):
        raise ValueError("Personalized model artifact must contain a dictionary bundle.")
    return bundle


def choose_time_column(df: pd.DataFrame) -> str:
    """Pick the best available time column for sorting if needed."""
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


def evaluate_popularity_baseline(
    baseline_items: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    top_k: int,
    relevance_threshold: float,
) -> Dict[str, float | int]:
    """Evaluate the popularity baseline on the test split."""
    if test_df.empty:
        return {
            "users_evaluated": 0,
            "recall_at_k": 0.0,
            "map_at_k": 0.0,
            "coverage": 0.0,
        }

    item_order = baseline_items.sort_values(["score", "interaction_count", "movie_id"], ascending=[False, False, True])["movie_id"].astype(int).tolist()
    train_user_items = train_df.groupby("user_id")["movie_id"].apply(set).to_dict()
    relevant_user_items = (
        test_df[test_df["rating"] >= relevance_threshold]
        .groupby("user_id")["movie_id"]
        .apply(lambda values: set(map(int, values.tolist())))
        .to_dict()
    )

    users = [int(user_id) for user_id in relevant_user_items.keys() if int(user_id) in train_user_items]
    if not users:
        return {
            "users_evaluated": 0,
            "recall_at_k": 0.0,
            "map_at_k": 0.0,
            "coverage": 0.0,
        }

    recalls: List[float] = []
    aps: List[float] = []
    recommended_pool: set[int] = set()

    for user_id in users:
        seen_items = train_user_items.get(user_id, set())
        recommendations = [movie_id for movie_id in item_order if movie_id not in seen_items][:top_k]
        recommended_pool.update(recommendations)

        relevant_items = relevant_user_items[user_id]
        recalls.append(recall_at_k(recommendations, relevant_items, top_k))
        aps.append(average_precision_at_k(recommendations, relevant_items, top_k))

    coverage = len(recommended_pool) / max(len(item_order), 1)
    return {
        "users_evaluated": len(users),
        "recall_at_k": float(sum(recalls) / len(recalls)),
        "map_at_k": float(sum(aps) / len(aps)),
        "coverage": float(coverage),
    }


def evaluate_personalized_model(
    bundle: Dict[str, object],
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    top_k: int,
    relevance_threshold: float,
) -> Dict[str, float | int]:
    """Evaluate the personalized SVD model on the test split."""
    if test_df.empty:
        return {
            "users_evaluated": 0,
            "recall_at_k": 0.0,
            "map_at_k": 0.0,
            "coverage": 0.0,
        }

    model = bundle["model"]
    item_ids: List[int] = [int(item_id) for item_id in bundle["item_ids"]]
    train_user_seen_items: Dict[int, set[int]] = bundle["train_user_seen_items"]

    relevant_user_items = (
        test_df[test_df["rating"] >= relevance_threshold]
        .groupby("user_id")["movie_id"]
        .apply(lambda values: set(map(int, values.tolist())))
        .to_dict()
    )
    users = [int(user_id) for user_id in relevant_user_items.keys() if int(user_id) in train_user_seen_items]
    if not users:
        return {
            "users_evaluated": 0,
            "recall_at_k": 0.0,
            "map_at_k": 0.0,
            "coverage": 0.0,
        }

    recalls: List[float] = []
    aps: List[float] = []
    recommended_pool: set[int] = set()

    for user_id in users:
        seen_items = train_user_seen_items.get(user_id, set())
        candidate_rows = []

        for movie_id in item_ids:
            if movie_id in seen_items:
                continue
            estimate = float(model.predict(str(user_id), str(movie_id)).est)
            candidate_rows.append((movie_id, estimate))

        candidate_rows.sort(key=lambda item: (-item[1], item[0]))
        recommendations = [movie_id for movie_id, _ in candidate_rows[:top_k]]
        recommended_pool.update(recommendations)

        relevant_items = relevant_user_items[user_id]
        recalls.append(recall_at_k(recommendations, relevant_items, top_k))
        aps.append(average_precision_at_k(recommendations, relevant_items, top_k))

    coverage = len(recommended_pool) / max(len(item_ids), 1)
    return {
        "users_evaluated": len(users),
        "recall_at_k": float(sum(recalls) / len(recalls)),
        "map_at_k": float(sum(aps) / len(aps)),
        "coverage": float(coverage),
    }


def build_normalized_popularity_scores(baseline_items: pd.DataFrame) -> Dict[int, float]:
    """Build a 0-1 normalized popularity score map from baseline artifacts."""
    if baseline_items.empty:
        return {}

    score_df = baseline_items.loc[:, ["movie_id", "score"]].copy()
    score_df["movie_id"] = score_df["movie_id"].astype(int)
    score_df["score"] = score_df["score"].astype(float)

    min_score = float(score_df["score"].min())
    max_score = float(score_df["score"].max())
    if max_score <= min_score:
        score_df["score_norm"] = 0.0
    else:
        score_df["score_norm"] = (score_df["score"] - min_score) / (max_score - min_score)

    return dict(zip(score_df["movie_id"].tolist(), score_df["score_norm"].tolist()))


def evaluate_hybrid_model(
    bundle: Dict[str, object],
    baseline_items: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    top_k: int,
    relevance_threshold: float,
    alpha: float,
) -> Dict[str, float | int]:
    """Evaluate a weighted hybrid of SVD score and popularity score.

    final_score = alpha * svd_score_norm + (1 - alpha) * popularity_score_norm
    """
    if test_df.empty:
        return {
            "users_evaluated": 0,
            "recall_at_k": 0.0,
            "map_at_k": 0.0,
            "coverage": 0.0,
        }

    model = bundle["model"]
    item_ids: List[int] = [int(item_id) for item_id in bundle["item_ids"]]
    train_user_seen_items: Dict[int, set[int]] = bundle["train_user_seen_items"]
    popularity_scores = build_normalized_popularity_scores(baseline_items)

    relevant_user_items = (
        test_df[test_df["rating"] >= relevance_threshold]
        .groupby("user_id")["movie_id"]
        .apply(lambda values: set(map(int, values.tolist())))
        .to_dict()
    )
    users = [int(user_id) for user_id in relevant_user_items.keys() if int(user_id) in train_user_seen_items]
    if not users:
        return {
            "users_evaluated": 0,
            "recall_at_k": 0.0,
            "map_at_k": 0.0,
            "coverage": 0.0,
        }

    recalls: List[float] = []
    aps: List[float] = []
    recommended_pool: set[int] = set()

    for user_id in users:
        seen_items = train_user_seen_items.get(user_id, set())
        svd_rows: List[tuple[int, float]] = []
        for movie_id in item_ids:
            if movie_id in seen_items:
                continue
            svd_rows.append((movie_id, float(model.predict(str(user_id), str(movie_id)).est)))

        if not svd_rows:
            recalls.append(0.0)
            aps.append(0.0)
            continue

        svd_scores = [score for _, score in svd_rows]
        svd_min = min(svd_scores)
        svd_max = max(svd_scores)

        hybrid_rows: List[tuple[int, float]] = []
        for movie_id, svd_score in svd_rows:
            if svd_max <= svd_min:
                svd_norm = 0.0
            else:
                svd_norm = (svd_score - svd_min) / (svd_max - svd_min)
            pop_norm = float(popularity_scores.get(movie_id, 0.0))
            final_score = alpha * svd_norm + (1.0 - alpha) * pop_norm
            hybrid_rows.append((movie_id, final_score))

        hybrid_rows.sort(key=lambda item: (-item[1], item[0]))
        recommendations = [movie_id for movie_id, _ in hybrid_rows[:top_k]]
        recommended_pool.update(recommendations)

        relevant_items = relevant_user_items[user_id]
        recalls.append(recall_at_k(recommendations, relevant_items, top_k))
        aps.append(average_precision_at_k(recommendations, relevant_items, top_k))

    coverage = len(recommended_pool) / max(len(item_ids), 1)
    return {
        "users_evaluated": len(users),
        "recall_at_k": float(sum(recalls) / len(recalls)),
        "map_at_k": float(sum(aps) / len(aps)),
        "coverage": float(coverage),
    }


def select_best_hybrid_alpha(
    bundle: Dict[str, object],
    baseline_items: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    top_k: int,
    relevance_threshold: float,
    alphas: Sequence[float],
) -> Dict[str, object]:
    """Evaluate hybrid on a list of alpha values and return the best setting."""
    alpha_metrics: Dict[str, Dict[str, float | int]] = {}
    best_alpha = float(alphas[0])
    best_metrics: Dict[str, float | int] | None = None

    for alpha in alphas:
        metrics = evaluate_hybrid_model(
            bundle=bundle,
            baseline_items=baseline_items,
            train_df=train_df,
            test_df=test_df,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
            alpha=float(alpha),
        )
        key = f"{float(alpha):.2f}"
        alpha_metrics[key] = metrics

        if best_metrics is None:
            best_alpha = float(alpha)
            best_metrics = metrics
            continue

        current_score = (float(metrics["map_at_k"]), float(metrics["recall_at_k"]))
        best_score = (float(best_metrics["map_at_k"]), float(best_metrics["recall_at_k"]))
        if current_score > best_score:
            best_alpha = float(alpha)
            best_metrics = metrics

    return {
        "best_alpha": best_alpha,
        "metrics": best_metrics if best_metrics is not None else {},
        "alpha_metrics": alpha_metrics,
    }


def save_summary(summary: Dict[str, object], report_dir: Path) -> Path:
    """Persist a JSON evaluation summary."""
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f"Saved evaluation summary to: {summary_path}")
    return summary_path


def save_report(summary: Dict[str, object], report_dir: Path) -> Path:
    """Persist a markdown evaluation report."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "evaluation_report.md"

    lines = [
        "# Evaluation Report",
        "",
        "## Config",
        f"- top_k: {summary['top_k']}",
        f"- relevance_threshold: {summary['relevance_threshold']}",
        "",
        "## Popularity Baseline",
        f"- users_evaluated: {summary['popularity_baseline']['users_evaluated']}",
        f"- recall_at_{summary['top_k']}: {summary['popularity_baseline']['recall_at_k']:.4f}",
        f"- map_at_{summary['top_k']}: {summary['popularity_baseline']['map_at_k']:.4f}",
        f"- coverage: {summary['popularity_baseline']['coverage']:.4f}",
        "",
        "## Personalized SVD",
        f"- users_evaluated: {summary['personalized_svd']['users_evaluated']}",
        f"- recall_at_{summary['top_k']}: {summary['personalized_svd']['recall_at_k']:.4f}",
        f"- map_at_{summary['top_k']}: {summary['personalized_svd']['map_at_k']:.4f}",
        f"- coverage: {summary['personalized_svd']['coverage']:.4f}",
        "",
    ]

    if "personalized_hybrid" in summary:
        lines.extend(
            [
                "## Personalized Hybrid (SVD + Popularity)",
                f"- best_alpha: {summary['personalized_hybrid']['best_alpha']:.2f}",
                f"- users_evaluated: {summary['personalized_hybrid']['metrics']['users_evaluated']}",
                f"- recall_at_{summary['top_k']}: {summary['personalized_hybrid']['metrics']['recall_at_k']:.4f}",
                f"- map_at_{summary['top_k']}: {summary['personalized_hybrid']['metrics']['map_at_k']:.4f}",
                f"- coverage: {summary['personalized_hybrid']['metrics']['coverage']:.4f}",
                "",
            ]
        )

    lines.extend(["## Notes", str(summary["notes"])])

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved evaluation report to: {report_path}")
    return report_path


def run_evaluation(
    train_path: Path,
    test_path: Path,
    baseline_path: Path,
    personalized_path: Path,
    report_dir: Path,
    top_k: int,
    relevance_threshold: float,
    hybrid_alphas: Sequence[float] | None,
) -> Dict[str, object]:
    """Run evaluation for both requested models."""
    train_df = load_split(train_path)
    test_df = load_split(test_path)
    baseline_items = load_baseline_items(baseline_path)
    svd_bundle = load_personalized_bundle(personalized_path)

    popularity_metrics = evaluate_popularity_baseline(
        baseline_items=baseline_items,
        train_df=train_df,
        test_df=test_df,
        top_k=top_k,
        relevance_threshold=relevance_threshold,
    )

    personalized_metrics = evaluate_personalized_model(
        bundle=svd_bundle,
        train_df=train_df,
        test_df=test_df,
        top_k=top_k,
        relevance_threshold=relevance_threshold,
    )

    hybrid_result: Dict[str, object] | None = None
    if hybrid_alphas:
        hybrid_result = select_best_hybrid_alpha(
            bundle=svd_bundle,
            baseline_items=baseline_items,
            train_df=train_df,
            test_df=test_df,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
            alphas=hybrid_alphas,
        )

    summary: Dict[str, object] = {
        "top_k": top_k,
        "relevance_threshold": relevance_threshold,
        "popularity_baseline": popularity_metrics,
        "personalized_svd": personalized_metrics,
        "notes": (
            "Popularity baseline uses most_popular_items.parquet; personalized SVD uses svd_model.pkl."
        ),
    }

    if hybrid_result is not None:
        summary["personalized_hybrid"] = hybrid_result

    save_summary(summary, report_dir)
    save_report(summary, report_dir)
    return summary


def main() -> Dict[str, object]:
    """CLI entrypoint for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate popularity baseline and personalized SVD on the test split.")
    parser.add_argument(
        "--train-path",
        type=str,
        default=None,
        help="Path to train parquet (default: data/split/train.parquet).",
    )
    parser.add_argument(
        "--test-path",
        type=str,
        default=None,
        help="Path to test parquet (default: data/split/test.parquet).",
    )
    parser.add_argument(
        "--baseline-path",
        type=str,
        default=None,
        help="Path to the popularity baseline artifact (default: models/baseline/most_popular_items.parquet).",
    )
    parser.add_argument(
        "--personalized-path",
        type=str,
        default=None,
        help="Path to the personalized SVD bundle (default: models/personalized/svd_model.pkl).",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory for evaluation reports (default: reports/evaluation).",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Ranking cutoff for Recall@K and MAP@K.")
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=4.0,
        help="Minimum rating to count an item as relevant.",
    )
    parser.add_argument(
        "--hybrid-alphas",
        nargs="*",
        type=float,
        default=None,
        help="Optional list of alpha values for SVD+popularity hybrid evaluation (e.g. 0.5 0.7 0.8 0.9).",
    )

    args = parser.parse_args()
    train_path, test_path, baseline_path, personalized_path, report_dir = resolve_paths(
        args.train_path,
        args.test_path,
        args.baseline_path,
        args.personalized_path,
        args.report_dir,
    )

    print("Starting evaluation pipeline...")
    print(f"- train_path: {train_path}")
    print(f"- test_path: {test_path}")
    print(f"- baseline_path: {baseline_path}")
    print(f"- personalized_path: {personalized_path}")
    print(f"- top_k: {args.top_k}")
    print(f"- relevance_threshold: {args.relevance_threshold}")
    print(f"- hybrid_alphas: {args.hybrid_alphas}")

    return run_evaluation(
        train_path=train_path,
        test_path=test_path,
        baseline_path=baseline_path,
        personalized_path=personalized_path,
        report_dir=report_dir,
        top_k=args.top_k,
        relevance_threshold=args.relevance_threshold,
        hybrid_alphas=args.hybrid_alphas,
    )


if __name__ == "__main__":
    main()
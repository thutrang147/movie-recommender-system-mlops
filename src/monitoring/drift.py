"""Simple drift detection utilities for recommendation monitoring."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd  # type: ignore[import-not-found]


def _normalize_distribution(counts: Dict[int, float]) -> Dict[int, float]:
    if not counts:
        return {}
    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    return {int(key): float(value) / total for key, value in counts.items()}


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    eps = 1e-12
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / m))
    kl_qm = np.sum(q * np.log(q / m))
    return float(0.5 * (kl_pm + kl_qm))


def load_train_popularity_distribution(project_root: Path, registry_path: Path) -> Dict[int, float]:
    """Load training popularity distribution from model bundle or baseline artifact."""
    with open(registry_path, "r", encoding="utf-8") as file:
        registry = json.load(file)

    model_path = project_root / str(registry.get("active_model", {}).get("artifact_path", ""))
    if model_path.exists():
        with open(model_path, "rb") as file:
            bundle = pickle.load(file)
        if isinstance(bundle, dict):
            if "item_popularity_scores" in bundle and isinstance(bundle["item_popularity_scores"], dict):
                counts = {int(k): float(v) for k, v in bundle["item_popularity_scores"].items()}
                dist = _normalize_distribution(counts)
                if dist:
                    return dist
            if "best_item_scores" in bundle and isinstance(bundle["best_item_scores"], dict):
                counts = {int(k): float(v) for k, v in bundle["best_item_scores"].items()}
                dist = _normalize_distribution(counts)
                if dist:
                    return dist

    fallback_path = project_root / str(registry.get("fallback", {}).get("artifact_path", ""))
    if fallback_path.exists() and fallback_path.suffix == ".parquet":
        baseline_df = pd.read_parquet(fallback_path)
        if "movie_id" in baseline_df.columns:
            if "interaction_count" in baseline_df.columns:
                counts = dict(zip(baseline_df["movie_id"].astype(int), baseline_df["interaction_count"].astype(float)))
            elif "score" in baseline_df.columns:
                counts = dict(zip(baseline_df["movie_id"].astype(int), baseline_df["score"].astype(float)))
            else:
                counts = {int(movie_id): 1.0 for movie_id in baseline_df["movie_id"].astype(int).tolist()}
            return _normalize_distribution(counts)

    return {}


def load_production_recommendation_distribution(log_df: pd.DataFrame) -> Dict[int, float]:
    """Build item recommendation distribution from request logs."""
    if log_df.empty or "recommendations" not in log_df.columns:
        return {}

    item_counts: Dict[int, float] = {}
    for recs in log_df["recommendations"].tolist():
        if not isinstance(recs, list):
            continue
        for item_id in recs:
            try:
                key = int(item_id)
            except Exception:
                continue
            item_counts[key] = item_counts.get(key, 0.0) + 1.0

    return _normalize_distribution(item_counts)


def compute_drift_score(train_dist: Dict[int, float], prod_dist: Dict[int, float]) -> float:
    """Compute Jensen-Shannon divergence between train and production item distributions."""
    if not train_dist or not prod_dist:
        return 0.0

    keys = sorted(set(train_dist.keys()) | set(prod_dist.keys()))
    p = np.array([train_dist.get(key, 0.0) for key in keys], dtype=np.float64)
    q = np.array([prod_dist.get(key, 0.0) for key in keys], dtype=np.float64)
    return _js_divergence(p, q)


def evaluate_drift_warnings(
    drift_score: float,
    unknown_user_rate: float,
    thresholds: Dict[str, float],
) -> List[str]:
    """Apply simple rule-based drift alerts."""
    warnings: List[str] = []

    if drift_score > float(thresholds.get("drift_score_warn", 0.20)):
        warnings.append(
            f"Drift warning: recommendation distribution drift_score={drift_score:.4f} exceeds threshold."
        )

    if unknown_user_rate > float(thresholds.get("unknown_user_rate_warn", 0.40)):
        warnings.append(
            f"Drift warning: unknown_user_rate={unknown_user_rate:.4f} exceeds threshold."
        )

    return warnings


def build_top_item_shift(
    train_dist: Dict[int, float],
    prod_dist: Dict[int, float],
    top_n: int = 10,
) -> List[Tuple[int, float, float]]:
    """Return top-N production items with train/prod probability for diagnostics."""
    top_items = sorted(prod_dist.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [(int(item_id), float(train_dist.get(item_id, 0.0)), float(prod_prob)) for item_id, prod_prob in top_items]

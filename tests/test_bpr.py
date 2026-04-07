"""Unit tests for NumPy BPR personalized recommender."""

from __future__ import annotations

import pandas as pd

from src.models.bpr import BPRConfig, BPRModel, evaluate_bundle_ranking, recommend_with_bundle


def _build_small_train_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "movie_id": [10, 11, 12, 10, 13, 14, 11, 13, 15],
            "rating": [5.0, 4.0, 2.0, 5.0, 4.0, 2.0, 5.0, 4.0, 2.0],
        }
    )


def _build_small_val_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": [1, 2, 3],
            "movie_id": [13, 11, 10],
            "rating": [5.0, 5.0, 5.0],
        }
    )


def test_bpr_fit_and_recommend_excludes_seen_items() -> None:
    train_df = _build_small_train_df()
    val_df = _build_small_val_df()

    config = BPRConfig(
        factors=8,
        learning_rate=0.05,
        reg=0.001,
        epochs=3,
        n_samples_per_epoch=300,
        random_state=7,
        top_k=3,
        relevance_threshold=4.0,
        patience=2,
    )

    model = BPRModel(config=config).fit(train_df=train_df, val_df=val_df)
    recommendations = model.recommend_for_user(user_id=1, top_k=3)

    seen_items = set(train_df[train_df["user_id"] == 1]["movie_id"].tolist())
    assert recommendations
    assert not (set(recommendations) & seen_items)


def test_bpr_bundle_recommend_and_evaluate() -> None:
    train_df = _build_small_train_df()
    val_df = _build_small_val_df()

    config = BPRConfig(
        factors=8,
        learning_rate=0.05,
        reg=0.001,
        epochs=3,
        n_samples_per_epoch=300,
        random_state=11,
        top_k=3,
        relevance_threshold=4.0,
        patience=2,
    )

    model = BPRModel(config=config).fit(train_df=train_df, val_df=val_df)
    bundle = model.to_bundle()

    rec_rows = recommend_with_bundle(bundle=bundle, user_id=2, top_k=3)
    assert rec_rows
    assert all(len(row) == 2 for row in rec_rows)

    metrics = evaluate_bundle_ranking(bundle=bundle, test_df=val_df, top_k=3, relevance_threshold=4.0)
    assert set(metrics.keys()) == {"users_evaluated", "recall_at_k", "map_at_k", "hit_rate_at_k", "coverage"}
    assert metrics["users_evaluated"] >= 0
    assert metrics["recall_at_k"] >= 0.0
    assert metrics["map_at_k"] >= 0.0
    assert metrics["hit_rate_at_k"] >= 0.0
    assert metrics["coverage"] >= 0.0

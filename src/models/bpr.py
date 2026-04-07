"""Bayesian Personalized Ranking (BPR-MF) implemented with NumPy.

This module is dependency-light and designed to integrate with the existing
train/evaluate/recommend workflow in this repository.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd  # type: ignore[import-not-found]


@dataclass
class BPRConfig:
    """Configuration for BPR matrix factorization."""

    factors: int = 64
    learning_rate: float = 0.03
    reg: float = 0.001
    epochs: int = 30
    n_samples_per_epoch: int = 200000
    random_state: int = 42
    top_k: int = 10
    relevance_threshold: float = 4.0
    patience: int = 5


class BPRModel:
    """Simple BPR-MF model with SGD optimization on sampled triplets."""

    def __init__(self, config: BPRConfig):
        self.config = config
        self.rng = np.random.default_rng(config.random_state)

        self.user_map: Dict[int, int] = {}
        self.item_map: Dict[int, int] = {}
        self.reverse_user_map: Dict[int, int] = {}
        self.reverse_item_map: Dict[int, int] = {}

        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None
        self.user_bias: np.ndarray | None = None
        self.item_bias: np.ndarray | None = None

        self.train_user_seen_items: Dict[int, set[int]] = {}
        self.train_user_positive_items: Dict[int, List[int]] = {}
        self._positive_pairs: List[Tuple[int, int]] = []

        self.best_epoch: int = 0
        self.best_val_map: float = float("-inf")

    def _require_fitted(self) -> None:
        if self.user_factors is None or self.item_factors is None or self.user_bias is None or self.item_bias is None:
            raise RuntimeError("BPR model is not fitted yet.")

    def _validate_schema(self, df: pd.DataFrame) -> None:
        required = {"user_id", "movie_id", "rating"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Input split is missing required columns: {', '.join(sorted(missing))}")

    def _build_mappings(self, train_df: pd.DataFrame) -> None:
        users = sorted(int(user_id) for user_id in train_df["user_id"].unique())
        items = sorted(int(movie_id) for movie_id in train_df["movie_id"].unique())

        self.user_map = {user_id: idx for idx, user_id in enumerate(users)}
        self.item_map = {movie_id: idx for idx, movie_id in enumerate(items)}
        self.reverse_user_map = {idx: user_id for user_id, idx in self.user_map.items()}
        self.reverse_item_map = {idx: movie_id for movie_id, idx in self.item_map.items()}

    def _build_interactions(self, train_df: pd.DataFrame) -> None:
        seen_items: Dict[int, set[int]] = {}
        positive_items: Dict[int, set[int]] = {}

        for row in train_df.itertuples(index=False):
            user_id = int(row.user_id)
            movie_id = int(row.movie_id)
            rating = float(row.rating)

            user_idx = self.user_map[user_id]
            item_idx = self.item_map[movie_id]

            seen_items.setdefault(user_idx, set()).add(item_idx)
            if rating >= self.config.relevance_threshold:
                positive_items.setdefault(user_idx, set()).add(item_idx)

        positive_users = {user_idx: sorted(list(items)) for user_idx, items in positive_items.items() if items}
        if not positive_users:
            raise ValueError(
                "No positive interactions found in train split with "
                f"relevance_threshold >= {self.config.relevance_threshold}."
            )

        self.train_user_seen_items = {
            self.reverse_user_map[user_idx]: {self.reverse_item_map[item_idx] for item_idx in item_indices}
            for user_idx, item_indices in seen_items.items()
        }
        self.train_user_positive_items = {user_idx: item_indices for user_idx, item_indices in positive_users.items()}

        self._positive_pairs = [
            (user_idx, item_idx)
            for user_idx, item_indices in self.train_user_positive_items.items()
            for item_idx in item_indices
        ]

    def _initialize_parameters(self) -> None:
        n_users = len(self.user_map)
        n_items = len(self.item_map)
        factors = self.config.factors

        scale = 0.1 / np.sqrt(max(factors, 1))
        self.user_factors = self.rng.normal(loc=0.0, scale=scale, size=(n_users, factors)).astype(np.float32)
        self.item_factors = self.rng.normal(loc=0.0, scale=scale, size=(n_items, factors)).astype(np.float32)
        self.user_bias = np.zeros(n_users, dtype=np.float32)
        self.item_bias = np.zeros(n_items, dtype=np.float32)

    def _sample_negative(self, user_idx: int) -> int:
        seen = self.train_user_positive_items.get(user_idx, [])
        seen_set = set(seen)
        n_items = len(self.item_map)

        for _ in range(100):
            candidate = int(self.rng.integers(0, n_items))
            if candidate not in seen_set:
                return candidate

        # Fallback when user interacted with almost all items.
        for candidate in range(n_items):
            if candidate not in seen_set:
                return candidate

        return 0

    @staticmethod
    def _sigmoid(x: float) -> float:
        if x >= 0:
            z = np.exp(-x)
            return float(1.0 / (1.0 + z))
        z = np.exp(x)
        return float(z / (1.0 + z))

    def _score_index(self, user_idx: int, item_idx: int) -> float:
        self._require_fitted()
        assert self.user_factors is not None
        assert self.item_factors is not None
        assert self.user_bias is not None
        assert self.item_bias is not None

        return float(
            np.dot(self.user_factors[user_idx], self.item_factors[item_idx])
            + self.user_bias[user_idx]
            + self.item_bias[item_idx]
        )

    def recommend_for_user(self, user_id: int, top_k: int | None = None) -> List[int]:
        """Recommend top-K unseen items for one user in raw ID space."""
        self._require_fitted()

        k = int(top_k if top_k is not None else self.config.top_k)
        if k <= 0:
            return []

        if user_id not in self.user_map:
            # Cold-start fallback: global item bias ranking.
            assert self.item_bias is not None
            top_indices = np.argsort(-self.item_bias)[:k]
            return [self.reverse_item_map[int(idx)] for idx in top_indices]

        user_idx = self.user_map[user_id]
        assert self.user_factors is not None
        assert self.item_factors is not None
        assert self.user_bias is not None
        assert self.item_bias is not None

        scores = self.item_factors @ self.user_factors[user_idx]
        scores = scores + self.item_bias + self.user_bias[user_idx]

        seen_raw = self.train_user_seen_items.get(user_id, set())
        seen_indices = [self.item_map[movie_id] for movie_id in seen_raw if movie_id in self.item_map]
        if seen_indices:
            scores[seen_indices] = -np.inf

        k = min(k, len(scores))
        if k == 0:
            return []

        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [self.reverse_item_map[int(idx)] for idx in top_indices]

    def ranking_metrics(self, eval_df: pd.DataFrame, top_k: int | None = None) -> Dict[str, float | int]:
        """Evaluate Recall@K, MAP@K, HitRate@K and Coverage on holdout split."""
        self._validate_schema(eval_df)
        if eval_df.empty:
            return {
                "users_evaluated": 0,
                "recall_at_k": 0.0,
                "map_at_k": 0.0,
                "hit_rate_at_k": 0.0,
                "coverage": 0.0,
            }

        k = int(top_k if top_k is not None else self.config.top_k)
        if k <= 0:
            return {
                "users_evaluated": 0,
                "recall_at_k": 0.0,
                "map_at_k": 0.0,
                "hit_rate_at_k": 0.0,
                "coverage": 0.0,
            }

        relevant_df = eval_df[eval_df["rating"] >= self.config.relevance_threshold]
        relevant_user_items = (
            relevant_df.groupby("user_id")["movie_id"]
            .apply(lambda values: set(int(item_id) for item_id in values.tolist()))
            .to_dict()
        )

        users = [
            int(user_id)
            for user_id in relevant_user_items.keys()
            if int(user_id) in self.train_user_seen_items
        ]
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
            recommended_items = self.recommend_for_user(user_id=user_id, top_k=k)
            relevant_items = relevant_user_items[user_id]
            recommended_pool.update(recommended_items)

            hit_count = len(set(recommended_items) & relevant_items)
            recall = hit_count / len(relevant_items) if relevant_items else 0.0

            hits_so_far = 0
            precision_sum = 0.0
            for rank, movie_id in enumerate(recommended_items, start=1):
                if movie_id in relevant_items:
                    hits_so_far += 1
                    precision_sum += hits_so_far / rank
            ap = precision_sum / min(len(relevant_items), k) if relevant_items else 0.0

            recalls.append(float(recall))
            aps.append(float(ap))
            hits.append(int(hit_count > 0))

        coverage = len(recommended_pool) / max(len(self.item_map), 1)
        return {
            "users_evaluated": len(users),
            "recall_at_k": float(sum(recalls) / len(recalls)),
            "map_at_k": float(sum(aps) / len(aps)),
            "hit_rate_at_k": float(sum(hits) / len(hits)),
            "coverage": float(coverage),
        }

    def _train_one_triplet(self, user_idx: int, pos_idx: int, neg_idx: int) -> float:
        self._require_fitted()
        assert self.user_factors is not None
        assert self.item_factors is not None
        assert self.user_bias is not None
        assert self.item_bias is not None

        lr = float(self.config.learning_rate)
        reg = float(self.config.reg)

        u_vec = self.user_factors[user_idx].copy()
        i_vec = self.item_factors[pos_idx].copy()
        j_vec = self.item_factors[neg_idx].copy()

        x_uij = (
            float(np.dot(u_vec, i_vec - j_vec))
            + float(self.item_bias[pos_idx] - self.item_bias[neg_idx])
        )
        sigmoid = self._sigmoid(x_uij)
        gradient = 1.0 - sigmoid

        self.user_factors[user_idx] += lr * (gradient * (i_vec - j_vec) - reg * u_vec)
        self.item_factors[pos_idx] += lr * (gradient * u_vec - reg * i_vec)
        self.item_factors[neg_idx] += lr * (-gradient * u_vec - reg * j_vec)

        self.item_bias[pos_idx] += lr * (gradient - reg * float(self.item_bias[pos_idx]))
        self.item_bias[neg_idx] += lr * (-gradient - reg * float(self.item_bias[neg_idx]))

        return float(-np.log(max(sigmoid, 1e-12)))

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None) -> "BPRModel":
        """Train model using sampled BPR triplets and optional validation early stopping."""
        self._validate_schema(train_df)
        self._build_mappings(train_df)
        self._build_interactions(train_df)
        self._initialize_parameters()

        positive_pairs = self._positive_pairs
        if not positive_pairs:
            raise ValueError("No positive pairs were produced from train split.")

        n_samples = int(self.config.n_samples_per_epoch)
        n_samples = max(n_samples, len(positive_pairs))

        best_snapshot: Dict[str, np.ndarray] | None = None
        bad_epochs = 0
        self.best_epoch = 0
        self.best_val_map = float("-inf")

        for epoch in range(1, self.config.epochs + 1):
            losses: List[float] = []

            sampled_indices = self.rng.integers(0, len(positive_pairs), size=n_samples)
            for pair_idx in sampled_indices:
                user_idx, pos_idx = positive_pairs[int(pair_idx)]
                neg_idx = self._sample_negative(user_idx)
                losses.append(self._train_one_triplet(user_idx=user_idx, pos_idx=pos_idx, neg_idx=neg_idx))

            if val_df is None:
                continue

            val_metrics = self.ranking_metrics(val_df, top_k=self.config.top_k)
            current_map = float(val_metrics["map_at_k"])

            if current_map > self.best_val_map:
                self.best_val_map = current_map
                self.best_epoch = epoch
                bad_epochs = 0
                assert self.user_factors is not None
                assert self.item_factors is not None
                assert self.user_bias is not None
                assert self.item_bias is not None
                best_snapshot = {
                    "user_factors": self.user_factors.copy(),
                    "item_factors": self.item_factors.copy(),
                    "user_bias": self.user_bias.copy(),
                    "item_bias": self.item_bias.copy(),
                }
            else:
                bad_epochs += 1

            mean_loss = float(sum(losses) / max(len(losses), 1))
            print(
                f"Epoch {epoch:02d}/{self.config.epochs} | loss={mean_loss:.4f} "
                f"| val_map@{self.config.top_k}={current_map:.4f}"
            )

            if bad_epochs >= self.config.patience:
                print(
                    "Early stopping triggered at epoch "
                    f"{epoch}. Best epoch={self.best_epoch}, best MAP@{self.config.top_k}={self.best_val_map:.4f}"
                )
                break

        if val_df is not None and best_snapshot is not None:
            self.user_factors = best_snapshot["user_factors"]
            self.item_factors = best_snapshot["item_factors"]
            self.user_bias = best_snapshot["user_bias"]
            self.item_bias = best_snapshot["item_bias"]

        return self

    def to_bundle(self) -> Dict[str, object]:
        """Serialize model into a dict artifact compatible with repo workflow."""
        self._require_fitted()
        assert self.user_factors is not None
        assert self.item_factors is not None
        assert self.user_bias is not None
        assert self.item_bias is not None

        # Popularity fallback based on training exposure.
        movie_counts: Dict[int, int] = {}
        for _, seen_movies in self.train_user_seen_items.items():
            for movie_id in seen_movies:
                movie_counts[movie_id] = movie_counts.get(movie_id, 0) + 1
        popularity_order = sorted(movie_counts.keys(), key=lambda movie_id: (-movie_counts[movie_id], movie_id))

        bundle: Dict[str, object] = {
            "algorithm": "bpr_mf_numpy",
            "config": asdict(self.config),
            "user_map": self.user_map,
            "item_map": self.item_map,
            "reverse_user_map": self.reverse_user_map,
            "reverse_item_map": self.reverse_item_map,
            "user_factors": self.user_factors,
            "item_factors": self.item_factors,
            "user_bias": self.user_bias,
            "item_bias": self.item_bias,
            "train_user_seen_items": self.train_user_seen_items,
            "item_ids": [self.reverse_item_map[idx] for idx in range(len(self.item_map))],
            "item_popularity_order": popularity_order,
            "item_popularity_scores": movie_counts,
            "best_epoch": self.best_epoch,
            "best_val_map": self.best_val_map,
        }
        return bundle


def recommend_with_bundle(bundle: Dict[str, object], user_id: int, top_k: int) -> List[Tuple[int, float]]:
    """Generate BPR recommendations directly from a serialized bundle."""
    user_map: Dict[int, int] = bundle["user_map"]  # type: ignore[assignment]
    item_map: Dict[int, int] = bundle["item_map"]  # type: ignore[assignment]
    reverse_item_map: Dict[int, int] = bundle["reverse_item_map"]  # type: ignore[assignment]
    train_user_seen_items: Dict[int, set[int]] = bundle["train_user_seen_items"]  # type: ignore[assignment]
    item_popularity_order: List[int] = [int(item_id) for item_id in bundle.get("item_popularity_order", [])]

    user_factors = np.asarray(bundle["user_factors"], dtype=np.float32)
    item_factors = np.asarray(bundle["item_factors"], dtype=np.float32)
    user_bias = np.asarray(bundle["user_bias"], dtype=np.float32)
    item_bias = np.asarray(bundle["item_bias"], dtype=np.float32)

    if top_k <= 0:
        return []

    if user_id not in user_map:
        fallback = item_popularity_order[:top_k]
        return [(movie_id, float(top_k - rank)) for rank, movie_id in enumerate(fallback)]

    user_idx = int(user_map[user_id])
    scores = item_factors @ user_factors[user_idx]
    scores = scores + item_bias + user_bias[user_idx]

    seen_raw = train_user_seen_items.get(user_id, set())
    seen_idx = [item_map[movie_id] for movie_id in seen_raw if movie_id in item_map]
    if seen_idx:
        scores[seen_idx] = -np.inf

    k = min(top_k, len(scores))
    if k == 0:
        return []

    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

    return [(reverse_item_map[int(item_idx)], float(scores[int(item_idx)])) for item_idx in top_indices]


def evaluate_bundle_ranking(
    bundle: Dict[str, object],
    test_df: pd.DataFrame,
    top_k: int,
    relevance_threshold: float,
) -> Dict[str, float | int]:
    """Evaluate a serialized BPR bundle on a holdout split."""
    if test_df.empty:
        return {
            "users_evaluated": 0,
            "recall_at_k": 0.0,
            "map_at_k": 0.0,
            "hit_rate_at_k": 0.0,
            "coverage": 0.0,
        }

    train_user_seen_items: Dict[int, set[int]] = bundle["train_user_seen_items"]  # type: ignore[assignment]

    relevant_df = test_df[test_df["rating"] >= relevance_threshold]
    relevant_user_items = (
        relevant_df.groupby("user_id")["movie_id"]
        .apply(lambda values: set(int(item_id) for item_id in values.tolist()))
        .to_dict()
    )

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
        ranked = recommend_with_bundle(bundle=bundle, user_id=user_id, top_k=top_k)
        recommendations = [movie_id for movie_id, _ in ranked]
        recommended_pool.update(recommendations)

        relevant_items = relevant_user_items[user_id]
        hit_count = len(set(recommendations) & relevant_items)
        recalls.append(hit_count / len(relevant_items) if relevant_items else 0.0)

        hits_so_far = 0
        precision_sum = 0.0
        for rank, movie_id in enumerate(recommendations, start=1):
            if movie_id in relevant_items:
                hits_so_far += 1
                precision_sum += hits_so_far / rank
        aps.append(precision_sum / min(len(relevant_items), top_k) if relevant_items else 0.0)
        hits.append(int(hit_count > 0))

    catalog_size = len(bundle.get("item_ids", []))
    coverage = len(recommended_pool) / max(catalog_size, 1)
    return {
        "users_evaluated": len(users),
        "recall_at_k": float(sum(recalls) / len(recalls)),
        "map_at_k": float(sum(aps) / len(aps)),
        "hit_rate_at_k": float(sum(hits) / len(hits)),
        "coverage": float(coverage),
    }

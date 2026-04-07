"""Content-based recommender using movie metadata and user preference profiles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd  # type: ignore[import-not-found]
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


@dataclass
class ContentBasedConfig:
    """Configuration for the content-based recommender."""

    top_k: int = 10
    relevance_threshold: float = 4.0
    min_df: int = 2
    max_features: int | None = 12000
    ngram_range: tuple[int, int] = (1, 2)


class ContentBasedRecommender:
    """Content-based recommender built from movie text features."""

    def __init__(self, config: ContentBasedConfig):
        self.config = config
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            min_df=config.min_df,
            max_features=config.max_features,
            ngram_range=config.ngram_range,
        )

        self.movie_features = None
        self.movie_ids: List[int] = []
        self.movie_id_to_index: Dict[int, int] = {}
        self.user_profiles: Dict[int, np.ndarray] = {}
        self.train_user_seen_items: Dict[int, set[int]] = {}
        self.movie_popularity_order: List[int] = []
        self.best_item_scores: Dict[int, float] = {}

    @staticmethod
    def _build_text_columns(movies_df: pd.DataFrame) -> pd.Series:
        title = movies_df["title"].fillna("").astype(str)
        genres = movies_df["genres"].fillna("").astype(str).str.replace("|", " ", regex=False)
        return (title + " " + genres).str.strip()

    def _validate_schema(self, train_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        required_train = {"user_id", "movie_id", "rating"}
        required_movies = {"movie_id", "title", "genres"}
        missing_train = required_train - set(train_df.columns)
        missing_movies = required_movies - set(movies_df.columns)
        if missing_train:
            raise ValueError(f"Training split is missing required columns: {', '.join(sorted(missing_train))}")
        if missing_movies:
            raise ValueError(f"Movie metadata is missing required columns: {', '.join(sorted(missing_movies))}")

    def fit(self, train_df: pd.DataFrame, movies_df: pd.DataFrame) -> "ContentBasedRecommender":
        """Fit the content-based recommender from training interactions and movie metadata."""
        self._validate_schema(train_df, movies_df)

        movies_df = movies_df.copy()
        movies_df["movie_id"] = movies_df["movie_id"].astype(int)
        movies_df = movies_df.drop_duplicates(subset=["movie_id"]).reset_index(drop=True)

        self.movie_ids = movies_df["movie_id"].astype(int).tolist()
        self.movie_id_to_index = {movie_id: idx for idx, movie_id in enumerate(self.movie_ids)}

        movie_text = self._build_text_columns(movies_df)
        self.movie_features = self.vectorizer.fit_transform(movie_text.tolist())
        self.movie_features = normalize(self.movie_features, axis=1)

        self.train_user_seen_items = {
            int(user_id): set(map(int, group["movie_id"].tolist()))
            for user_id, group in train_df.groupby("user_id")
        }

        self.user_profiles = {}
        for user_id, group in train_df.groupby("user_id"):
            positive_rows = group[group["rating"] >= self.config.relevance_threshold]
            if positive_rows.empty:
                positive_rows = group

            weights = positive_rows["rating"].astype(float).to_numpy()
            movie_indices = [self.movie_id_to_index[movie_id] for movie_id in positive_rows["movie_id"].astype(int).tolist() if movie_id in self.movie_id_to_index]
            if not movie_indices:
                continue

            matrix = self.movie_features[movie_indices]
            weighted = matrix.multiply(weights[: matrix.shape[0]].reshape(-1, 1))
            profile = np.asarray(weighted.sum(axis=0)).ravel()
            norm = np.linalg.norm(profile)
            if norm > 0:
                profile = profile / norm
            self.user_profiles[int(user_id)] = profile.astype(np.float32)

        popularity = (
            train_df.groupby("movie_id", as_index=False)
            .agg(interaction_count=("rating", "size"), mean_rating=("rating", "mean"))
            .sort_values(["interaction_count", "mean_rating", "movie_id"], ascending=[False, False, True])
        )
        self.movie_popularity_order = popularity["movie_id"].astype(int).tolist()
        self.best_item_scores = dict(zip(popularity["movie_id"].astype(int).tolist(), popularity["interaction_count"].astype(float).tolist()))
        return self

    def recommend_for_user(self, user_id: int, top_k: int | None = None) -> List[int]:
        """Recommend unseen movies for a user."""
        if self.movie_features is None:
            raise RuntimeError("Model is not fitted yet.")

        k = int(top_k if top_k is not None else self.config.top_k)
        if k <= 0:
            return []

        if user_id not in self.user_profiles:
            fallback = [movie_id for movie_id in self.movie_popularity_order if movie_id not in self.train_user_seen_items.get(user_id, set())]
            return fallback[:k]

        profile = self.user_profiles[user_id]
        scores = np.asarray(self.movie_features @ profile.reshape(-1, 1)).ravel()

        seen_items = self.train_user_seen_items.get(user_id, set())
        seen_indices = [self.movie_id_to_index[movie_id] for movie_id in seen_items if movie_id in self.movie_id_to_index]
        if seen_indices:
            scores[seen_indices] = -np.inf

        k = min(k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
        return [self.movie_ids[int(idx)] for idx in top_indices]

    def ranking_metrics(self, eval_df: pd.DataFrame, top_k: int | None = None) -> Dict[str, float | int]:
        """Evaluate ranking metrics on a holdout split."""
        if self.movie_features is None:
            raise RuntimeError("Model is not fitted yet.")
        if eval_df.empty:
            return {"users_evaluated": 0, "recall_at_k": 0.0, "map_at_k": 0.0, "hit_rate_at_k": 0.0, "coverage": 0.0}

        k = int(top_k if top_k is not None else self.config.top_k)
        if k <= 0:
            return {"users_evaluated": 0, "recall_at_k": 0.0, "map_at_k": 0.0, "hit_rate_at_k": 0.0, "coverage": 0.0}

        relevant_df = eval_df[eval_df["rating"] >= self.config.relevance_threshold]
        relevant_user_items = (
            relevant_df.groupby("user_id")["movie_id"]
            .apply(lambda values: set(int(movie_id) for movie_id in values.tolist()))
            .to_dict()
        )

        users = [int(user_id) for user_id in relevant_user_items.keys() if int(user_id) in self.train_user_seen_items]
        if not users:
            return {"users_evaluated": 0, "recall_at_k": 0.0, "map_at_k": 0.0, "hit_rate_at_k": 0.0, "coverage": 0.0}

        recalls: List[float] = []
        aps: List[float] = []
        hits: List[int] = []
        recommended_pool: set[int] = set()

        for user_id in users:
            recommendations = self.recommend_for_user(user_id=user_id, top_k=k)
            relevant_items = relevant_user_items[user_id]
            recommended_pool.update(recommendations)

            hit_count = len(set(recommendations) & relevant_items)
            recalls.append(hit_count / len(relevant_items) if relevant_items else 0.0)

            hits_so_far = 0
            precision_sum = 0.0
            for rank, movie_id in enumerate(recommendations, start=1):
                if movie_id in relevant_items:
                    hits_so_far += 1
                    precision_sum += hits_so_far / rank
            aps.append(precision_sum / min(len(relevant_items), k) if relevant_items else 0.0)
            hits.append(int(hit_count > 0))

        coverage = len(recommended_pool) / max(len(self.movie_ids), 1)
        return {
            "users_evaluated": len(users),
            "recall_at_k": float(sum(recalls) / len(recalls)),
            "map_at_k": float(sum(aps) / len(aps)),
            "hit_rate_at_k": float(sum(hits) / len(hits)),
            "coverage": float(coverage),
        }

    def to_bundle(self) -> Dict[str, object]:
        """Serialize fitted model to a portable bundle."""
        if self.movie_features is None:
            raise RuntimeError("Model is not fitted yet.")
        return {
            "algorithm": "content_based_tfidf",
            "config": asdict(self.config),
            "movie_ids": self.movie_ids,
            "movie_id_to_index": self.movie_id_to_index,
            "movie_features": self.movie_features,
            "user_profiles": self.user_profiles,
            "train_user_seen_items": self.train_user_seen_items,
            "movie_popularity_order": self.movie_popularity_order,
            "best_item_scores": self.best_item_scores,
        }


def recommend_with_bundle(bundle: Dict[str, object], user_id: int, top_k: int) -> List[tuple[int, float]]:
    """Recommend from a serialized content-based bundle."""
    movie_ids: List[int] = [int(movie_id) for movie_id in bundle["movie_ids"]]
    movie_id_to_index: Dict[int, int] = bundle["movie_id_to_index"]  # type: ignore[assignment]
    movie_features = bundle["movie_features"]
    user_profiles: Dict[int, np.ndarray] = bundle["user_profiles"]  # type: ignore[assignment]
    train_user_seen_items: Dict[int, set[int]] = bundle["train_user_seen_items"]  # type: ignore[assignment]
    movie_popularity_order: List[int] = [int(movie_id) for movie_id in bundle.get("movie_popularity_order", [])]

    if top_k <= 0:
        return []

    if user_id not in user_profiles:
        fallback = [movie_id for movie_id in movie_popularity_order if movie_id not in train_user_seen_items.get(user_id, set())]
        return [(movie_id, float(top_k - rank)) for rank, movie_id in enumerate(fallback[:top_k])]

    profile = user_profiles[user_id]
    scores = np.asarray(movie_features @ profile.reshape(-1, 1)).ravel()

    seen_items = train_user_seen_items.get(user_id, set())
    seen_indices = [movie_id_to_index[movie_id] for movie_id in seen_items if movie_id in movie_id_to_index]
    if seen_indices:
        scores[seen_indices] = -np.inf

    k = min(top_k, len(scores))
    if k == 0:
        return []

    top_indices = np.argpartition(scores, -k)[-k:]
    top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]
    return [(movie_ids[int(idx)], float(scores[int(idx)])) for idx in top_indices]

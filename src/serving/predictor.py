"""Serving-time predictor with model-registry lookup and fallback strategies."""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd  # type: ignore[import-not-found]

from src.models.bpr import recommend_with_bundle as recommend_with_bpr
from src.models.content_based import recommend_with_bundle as recommend_with_content
from src.models.recommend import recommend_for_user as recommend_generic


@dataclass
class RegistryModel:
    name: str
    version: str
    artifact_path: Path


@dataclass
class RegistryFallback:
    kind: str
    artifact_path: Path | None


class RecommendationPredictor:
    """Load one active model from registry and serve recommendations."""

    def __init__(self, project_root: Path, registry_path: Path, default_top_k: int = 10):
        self.project_root = project_root
        self.registry_path = registry_path
        self.default_top_k = default_top_k

        self.model_info: RegistryModel | None = None
        self.fallback_info: RegistryFallback | None = None
        self.bundle: Dict[str, object] | None = None
        self.algorithm: str | None = None

        self.train_user_seen_items: Dict[int, set[int]] = {}
        self.popularity_fallback_items: List[int] = []

    def load(self) -> None:
        if not self.registry_path.exists():
            raise FileNotFoundError(f"Missing registry file: {self.registry_path}")

        with open(self.registry_path, "r", encoding="utf-8") as file:
            registry = json.load(file)

        model_cfg = registry.get("active_model", {})
        fallback_cfg = registry.get("fallback", {})

        model_path = self.project_root / str(model_cfg.get("artifact_path", ""))
        if not model_path.exists():
            raise FileNotFoundError(f"Missing active model artifact: {model_path}")

        self.model_info = RegistryModel(
            name=str(model_cfg.get("name", "unknown")),
            version=str(model_cfg.get("version", "unknown")),
            artifact_path=model_path,
        )

        fallback_path_raw = fallback_cfg.get("artifact_path")
        fallback_path = self.project_root / str(fallback_path_raw) if fallback_path_raw else None
        self.fallback_info = RegistryFallback(
            kind=str(fallback_cfg.get("type", "popularity")),
            artifact_path=fallback_path,
        )

        with open(model_path, "rb") as file:
            bundle = pickle.load(file)
        if not isinstance(bundle, dict):
            raise ValueError("Model artifact is not a dict bundle.")

        self.bundle = bundle
        self.algorithm = str(bundle.get("algorithm", "svd_surprise"))

        raw_seen = bundle.get("train_user_seen_items", {})
        if isinstance(raw_seen, dict):
            self.train_user_seen_items = {
                int(user_id): {int(movie_id) for movie_id in set(items)}
                for user_id, items in raw_seen.items()
            }
        else:
            self.train_user_seen_items = {}

        self.popularity_fallback_items = self._load_popularity_fallback_items(bundle)

    def _load_popularity_fallback_items(self, bundle: Dict[str, object]) -> List[int]:
        if self.fallback_info and self.fallback_info.artifact_path and self.fallback_info.artifact_path.exists():
            if self.fallback_info.artifact_path.suffix == ".parquet":
                baseline_df = pd.read_parquet(self.fallback_info.artifact_path)
                sort_cols = [col for col in ["score", "interaction_count", "movie_id"] if col in baseline_df.columns]
                if sort_cols:
                    ascending = [False, False, True][: len(sort_cols)]
                    baseline_df = baseline_df.sort_values(sort_cols, ascending=ascending)
                return baseline_df["movie_id"].astype(int).tolist() if "movie_id" in baseline_df.columns else []

        if "item_popularity_order" in bundle:
            return [int(item_id) for item_id in bundle.get("item_popularity_order", [])]
        if "movie_popularity_order" in bundle:
            return [int(item_id) for item_id in bundle.get("movie_popularity_order", [])]
        return [int(item_id) for item_id in bundle.get("item_ids", [])]

    def _recommend_from_active_model(self, user_id: int, top_k: int) -> List[int]:
        if self.bundle is None:
            raise RuntimeError("Predictor is not loaded.")

        if self.algorithm == "bpr_mf_numpy":
            rows = recommend_with_bpr(self.bundle, user_id=user_id, top_k=top_k)
            return [int(movie_id) for movie_id, _ in rows]
        if self.algorithm == "content_based_tfidf":
            rows = recommend_with_content(self.bundle, user_id=user_id, top_k=top_k)
            return [int(movie_id) for movie_id, _ in rows]

        rec_df = recommend_generic(bundle=self.bundle, user_id=user_id, top_k=top_k)
        return rec_df["movie_id"].astype(int).tolist() if "movie_id" in rec_df.columns else []

    def recommend(self, user_id: int, top_k: int | None = None) -> Dict[str, object]:
        if self.bundle is None:
            raise RuntimeError("Predictor is not loaded.")

        k = int(top_k if top_k is not None else self.default_top_k)
        k = max(1, k)

        if user_id in self.train_user_seen_items:
            recs = self._recommend_from_active_model(user_id=user_id, top_k=k)
            return {
                "user_id": user_id,
                "strategy": str(self.model_info.name if self.model_info else self.algorithm),
                "model_version": str(self.model_info.version if self.model_info else "unknown"),
                "recommendations": recs[:k],
            }

        # Unknown user fallback.
        return {
            "user_id": user_id,
            "strategy": "popularity_fallback",
            "model_version": str(self.model_info.version if self.model_info else "unknown"),
            "recommendations": self.popularity_fallback_items[:k],
        }

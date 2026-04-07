"""Generate top-N personalized recommendations for a user."""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd  # type: ignore[import-not-found]

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.bpr import recommend_with_bundle
from src.models.content_based import recommend_with_bundle as recommend_with_content_bundle


def resolve_paths(
    artifact_path: str | None,
    movies_path: str | None,
) -> tuple[Path, Path | None]:
    """Resolve project-relative defaults for inference inputs."""
    project_root = Path(__file__).resolve().parents[2]
    resolved_artifact_path = Path(artifact_path) if artifact_path else project_root / "models" / "personalized" / "svd_model.pkl"
    resolved_movies_path = Path(movies_path) if movies_path else project_root / "data" / "processed" / "movies_preprocessed.parquet"
    return resolved_artifact_path, resolved_movies_path


def load_bundle(artifact_path: Path) -> Dict[str, object]:
    """Load the pickled model bundle."""
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing model artifact: {artifact_path}")
    with open(artifact_path, "rb") as file:
        bundle = pickle.load(file)
    if not isinstance(bundle, dict):
        raise ValueError("Model artifact must contain a dictionary bundle.")
    return bundle


def load_movies(movies_path: Path | None) -> pd.DataFrame | None:
    """Load the optional movie metadata table."""
    if movies_path is None or not movies_path.exists():
        return None
    return pd.read_parquet(movies_path)


def recommend_for_user(
    bundle: Dict[str, object],
    user_id: int,
    top_k: int,
) -> pd.DataFrame:
    """Rank unseen items for one user using the trained bundle."""
    algorithm = str(bundle.get("algorithm", "svd_surprise"))
    train_user_seen_items = bundle["train_user_seen_items"]
    popularity_order_key = "item_popularity_order" if "item_popularity_order" in bundle else "movie_popularity_order"
    popularity_order: List[int] = [int(item_id) for item_id in bundle.get(popularity_order_key, [])]

    if algorithm == "bpr_mf_numpy":
        rows = recommend_with_bundle(bundle=bundle, user_id=user_id, top_k=top_k)
        if not rows:
            return pd.DataFrame(columns=["rank", "movie_id", "score"])
        recommendations = pd.DataFrame(rows, columns=["movie_id", "score"])
        recommendations["rank"] = range(1, len(recommendations) + 1)
        return recommendations.loc[:, ["rank", "movie_id", "score"]]

    if algorithm == "content_based_tfidf":
        recommendations_list = recommend_with_content_bundle(bundle=bundle, user_id=user_id, top_k=top_k)
        if not recommendations_list:
            return pd.DataFrame(columns=["rank", "movie_id", "score"])
        recommendations = pd.DataFrame(recommendations_list, columns=["movie_id", "score"])
        recommendations["rank"] = range(1, len(recommendations) + 1)
        return recommendations.loc[:, ["rank", "movie_id", "score"]]

    model = bundle["model"]
    item_ids: List[int] = [int(item_id) for item_id in bundle["item_ids"]]

    if user_id not in train_user_seen_items:
        fallback = pd.DataFrame({"movie_id": popularity_order[:top_k], "score": [float(top_k - index) for index in range(min(top_k, len(popularity_order)))]})
        fallback["rank"] = range(1, len(fallback) + 1)
        return fallback.loc[:, ["rank", "movie_id", "score"]]

    seen_items = set(int(item_id) for item_id in train_user_seen_items[user_id])
    rows = []

    for movie_id in item_ids:
        if movie_id in seen_items:
            continue
        estimate = float(model.predict(str(user_id), str(movie_id)).est)
        rows.append({"movie_id": movie_id, "score": estimate})

    recommendations = pd.DataFrame(rows)
    if recommendations.empty:
        return pd.DataFrame(columns=["rank", "movie_id", "score"])

    recommendations = recommendations.sort_values(["score", "movie_id"], ascending=[False, True]).head(top_k).copy()
    recommendations["rank"] = range(1, len(recommendations) + 1)
    return recommendations.loc[:, ["rank", "movie_id", "score"]]


def attach_titles(recommendations: pd.DataFrame, movies_df: pd.DataFrame | None) -> pd.DataFrame:
    """Attach movie titles when metadata is available."""
    if movies_df is None or recommendations.empty or "movie_id" not in movies_df.columns:
        return recommendations

    merged = recommendations.merge(movies_df.loc[:, ["movie_id", "title"]], on="movie_id", how="left")
    column_order = ["rank", "movie_id", "title", "score"] if "title" in merged.columns else ["rank", "movie_id", "score"]
    return merged.loc[:, column_order]


def run_recommendation(
    artifact_path: Path,
    movies_path: Path | None,
    user_id: int,
    top_k: int,
) -> pd.DataFrame:
    """Load the bundle and generate recommendations for one user."""
    bundle = load_bundle(artifact_path)
    recommendations = recommend_for_user(bundle=bundle, user_id=user_id, top_k=top_k)
    movies_df = load_movies(movies_path)
    recommendations = attach_titles(recommendations, movies_df)

    print(f"Recommendations for user_id={user_id}")
    if recommendations.empty:
        print("No recommendations available.")
    else:
        print(recommendations.to_string(index=False))

    return recommendations


def main() -> pd.DataFrame:
    """CLI entrypoint for generating recommendations."""
    parser = argparse.ArgumentParser(description="Generate top-N personalized recommendations for a user.")
    parser.add_argument("--user-id", type=int, required=True, help="Target user id for recommendations.")
    parser.add_argument(
        "--artifact-path",
        type=str,
        default=None,
        help="Path to the pickled model bundle (default: models/personalized/svd_model.pkl).",
    )
    parser.add_argument(
        "--movies-path",
        type=str,
        default=None,
        help="Optional path to movie metadata parquet for title lookup.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of recommendations to return.")

    args = parser.parse_args()
    artifact_path, default_movies_path = resolve_paths(args.artifact_path, args.movies_path)

    movies_path = Path(args.movies_path) if args.movies_path else default_movies_path

    return run_recommendation(
        artifact_path=artifact_path,
        movies_path=movies_path,
        user_id=args.user_id,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
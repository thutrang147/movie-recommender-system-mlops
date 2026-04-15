import numpy as np
import pandas as pd

def robust_scale_to_rating(scores: np.ndarray, min_rating: float = 1.0, max_rating: float = 5.0) -> np.ndarray:
    """
    Scale an array of scores to the target rating range [min_rating, max_rating] using robust min-max scaling.
    If all scores are equal, returns the midpoint of the rating range.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) == 0:
        return np.array([])
    min_score = np.min(scores)
    max_score = np.max(scores)
    if np.isclose(max_score, min_score):
        return np.full_like(scores, (min_rating + max_rating) / 2, dtype=np.float32)
    scaled = (scores - min_score) / (max_score - min_score)
    return scaled * (max_rating - min_rating) + min_rating

def bayesian_average_score(item_mean: float, item_count: int, global_mean: float, m: int = 50) -> float:
    """
    Bayesian average for popularity/baseline/fallback.
    Returns a score in the original rating scale, e.g. [1, 5].
    """
    item_count = max(int(item_count), 0)
    return ((item_count / (item_count + m)) * item_mean) + ((m / (item_count + m)) * global_mean)

def build_popularity_table(
    ratings_df: pd.DataFrame,
    user_col: str = "user_id",
    item_col: str = "movie_id",
    rating_col: str = "rating",
    m: int = 50,
) -> pd.DataFrame:
    """
    Build a popularity table for baseline and cold-start fallback.
    Output columns:
    - movie_id
    - mean_rating
    - rating_count
    - popularity_score
    """
    global_mean = ratings_df[rating_col].mean()

    stats = (
        ratings_df.groupby(item_col)[rating_col]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "mean_rating", "count": "rating_count"})
    )

    stats["popularity_score"] = stats.apply(
        lambda row: bayesian_average_score(
            item_mean=row["mean_rating"],
            item_count=row["rating_count"],
            global_mean=global_mean,
            m=m,
        ),
        axis=1,
    )

    stats = stats.sort_values(
        by=["popularity_score", "rating_count", item_col],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    return stats

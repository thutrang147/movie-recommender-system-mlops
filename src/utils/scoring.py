import numpy as np
import pandas as pd

EPS = 1e-12

def clip_rating(score: float, min_rating: float = 1.0, max_rating: float = 5.0) -> float:
    return float(np.clip(score, min_rating, max_rating))

def robust_scale_to_rating(
    raw_scores,
    min_rating: float = 1.0,
    max_rating: float = 5.0,
    lower_q: float = 5.0,
    upper_q: float = 95.0,
):
    """
    Scale raw scores -> [min_rating, max_rating] bằng quantile clipping.
    Dùng cho BPR / Content-Based để hiển thị ra UI.
    """
    raw_scores = np.asarray(raw_scores, dtype=float)

    if raw_scores.size == 0:
        return raw_scores

    lo = np.percentile(raw_scores, lower_q)
    hi = np.percentile(raw_scores, upper_q)

    if hi - lo < EPS:
        mid = (min_rating + max_rating) / 2.0
        return np.full_like(raw_scores, mid, dtype=float)

    clipped = np.clip(raw_scores, lo, hi)
    scaled_01 = (clipped - lo) / (hi - lo)
    scaled = min_rating + scaled_01 * (max_rating - min_rating)
    return scaled

def bayesian_average_score(
    item_mean: float,
    item_count: int,
    global_mean: float,
    m: int = 50,
) -> float:
    """
    Bayesian average cho popularity/baseline/fallback.
    Trả ra score nằm tự nhiên trong thang rating gốc, ví dụ [1, 5].
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
    Xây bảng popular items dùng chung cho baseline + fallback cold-start.
    Output gồm:
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

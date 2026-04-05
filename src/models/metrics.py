import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from collections import defaultdict


def compute_rmse(y_true, y_pred) -> float:
    """
    Tính RMSE cho bài toán rating prediction.
    """
    if len(y_true) == 0:
        return 0.0
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def build_relevant_items_dict(
    test_df: pd.DataFrame,
    relevance_threshold: float = 4.0,
) -> dict:
    """
    Tạo ground-truth relevant items theo user từ test_df.
    Chỉ lấy các item có Rating >= relevance_threshold.
    """
    relevant_df = test_df[test_df["Rating"] >= relevance_threshold]
    return relevant_df.groupby("UserID")["MovieID"].apply(set).to_dict()


def build_seen_items_dict(train_df: pd.DataFrame) -> dict:
    """
    Tạo lịch sử item đã xem trong train theo từng user.
    """
    return train_df.groupby("UserID")["MovieID"].apply(set).to_dict()


def compute_ranking_metrics_from_recommendations(
    recommendations: dict,
    ground_truth: dict,
    catalog_size: int,
    top_k: int = 10,
):
    """
    Tính Recall@K, MAP@K, Coverage từ:
    - recommendations: dict[user_id] = list[item_id]
    - ground_truth: dict[user_id] = set[item_id relevant trong test]
    - catalog_size: tổng số item trong catalog model

    Lưu ý:
    - Hàm này giả định recommendations đã loại seen items rồi.
    - Hàm này chỉ đo, không tự split dữ liệu, nên không tự tạo leakage.
    """
    recalls = []
    aps = []
    recommended_items_pool = set()

    eval_users = [u for u in ground_truth.keys() if u in recommendations]

    for user_id in eval_users:
        pred_items = recommendations.get(user_id, [])[:top_k]
        true_items = ground_truth.get(user_id, set())

        if not true_items:
            continue

        recommended_items_pool.update(pred_items)

        hits = 0
        sum_precs = 0.0

        for rank, item in enumerate(pred_items, start=1):
            if item in true_items:
                hits += 1
                sum_precs += hits / rank

        recall = hits / len(true_items)
        ap = sum_precs / min(len(true_items), top_k)

        recalls.append(recall)
        aps.append(ap)

    mean_recall = float(np.mean(recalls)) if recalls else 0.0
    map_k = float(np.mean(aps)) if aps else 0.0
    coverage = float(len(recommended_items_pool) / catalog_size) if catalog_size > 0 else 0.0

    return mean_recall, map_k, coverage


def evaluate_top_k_recommendations(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    recommendation_fn,
    catalog_size: int,
    top_k: int = 10,
    relevance_threshold: float = 4.0,
):
    """
    Hàm đánh giá ranking dùng chung cho mọi model.

    Parameters
    ----------
    train_df : pd.DataFrame
        Dùng để biết item nào user đã xem và cần loại khỏi candidate set.
    test_df : pd.DataFrame
        Chỉ dùng làm ground-truth để đo metric.
    recommendation_fn : callable
        Hàm có dạng:
            recommendation_fn(user_id, seen_items, top_k) -> list[item_id]
        Model cụ thể sẽ tự implement phần sinh recommendation.
    catalog_size : int
        Tổng số item model biết.
    top_k : int
        K trong top-K metrics.
    relevance_threshold : float
        Ngưỡng xác định item relevant trong test.

    Returns
    -------
    recall_k, map_k, coverage
    """
    ground_truth = build_relevant_items_dict(
        test_df=test_df,
        relevance_threshold=relevance_threshold,
    )
    seen_items_dict = build_seen_items_dict(train_df)

    recommendations = {}

    for user_id in ground_truth.keys():
        seen_items = seen_items_dict.get(user_id, set())
        recs = recommendation_fn(user_id, seen_items, top_k)
        recommendations[user_id] = recs

    return compute_ranking_metrics_from_recommendations(
        recommendations=recommendations,
        ground_truth=ground_truth,
        catalog_size=catalog_size,
        top_k=top_k,
    )


def evaluate_explicit_predictions(test_df: pd.DataFrame, predict_fn) -> float:
    """
    Hàm chung để tính RMSE cho model có predict(user_id, item_id).

    Parameters
    ----------
    test_df : pd.DataFrame
        Tập test đã tách sẵn.
    predict_fn : callable
        Hàm có dạng:
            predict_fn(user_id, movie_id) -> float

    Returns
    -------
    rmse : float
    """
    y_true = test_df["Rating"].values
    y_pred = [predict_fn(row.UserID, row.MovieID) for row in test_df.itertuples()]
    return compute_rmse(y_true, y_pred)
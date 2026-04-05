import os
import sys
import pickle
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.models.content_based_recommender import ContentBasedRecommender


def load_movies_data(movies_path: str) -> pd.DataFrame:
    if not os.path.exists(movies_path):
        raise FileNotFoundError(f"Không tìm thấy file movies tại: {movies_path}")

    print(f"Loading movies data from: {movies_path}")
    if movies_path.endswith(".parquet"):
        return pd.read_parquet(movies_path)
    elif movies_path.endswith(".csv"):
        return pd.read_csv(movies_path)
    else:
        raise ValueError("Chỉ hỗ trợ file .csv hoặc .parquet cho movies data")


def load_ratings_data(ratings_path: str) -> pd.DataFrame:
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Không tìm thấy file ratings tại: {ratings_path}")

    print(f"Loading ratings data from: {ratings_path}")
    if ratings_path.endswith(".parquet"):
        return pd.read_parquet(ratings_path)
    elif ratings_path.endswith(".csv"):
        return pd.read_csv(ratings_path)
    else:
        raise ValueError("Chỉ hỗ trợ file .csv hoặc .parquet cho ratings data")


def run_content_training_pipeline():
    print("🚀 [Pipeline] Khởi động huấn luyện Content-Based Recommender...")

    movies_path = "data/processed/movies.parquet"
    ratings_path = "data/processed/ratings.parquet"

    movies_df = load_movies_data(movies_path)
    ratings_df = load_ratings_data(ratings_path)

    print(
        f"   -> Kích thước movies: {movies_df.shape} | "
        f"ratings: {ratings_df.shape}"
    )

    model = ContentBasedRecommender(
        max_features=5000,
        ngram_range=(1, 2),
    )

    model.fit(movies_df=movies_df, ratings_df=ratings_df)

    save_path = "models/content_based_recommender.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\n✅ Đã lưu Content-Based model tại: {save_path}")

    # demo nhanh
    sample_movie_id = movies_df["MovieID"].iloc[0]
    print(f"\n🎬 Demo similar items cho MovieID={sample_movie_id}:")
    print(model.recommend_similar_items(sample_movie_id, top_k=5)[["MovieID", "Title", "SimilarityScore"]])

    sample_user_id = ratings_df["UserID"].iloc[0]
    print(f"\n👤 Demo recommend cho UserID={sample_user_id}:")
    print(model.recommend_for_user(sample_user_id, top_k=5)[["MovieID", "Title", "ProfileScore"]])


if __name__ == "__main__":
    run_content_training_pipeline()
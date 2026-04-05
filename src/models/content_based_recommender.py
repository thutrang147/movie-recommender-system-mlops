import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class ContentBasedRecommender:
    """
    Content-Based Filtering dùng TF-IDF + Cosine Similarity.
    
    Mục tiêu:
    - Gợi ý phim tương tự một phim đầu vào
    - Hỗ trợ cold-start item
    - Hỗ trợ fallback cho user mới / user ít lịch sử
    - Có thể recommend từ lịch sử user bằng cách cộng gộp profile nội dung
    """

    def __init__(self, max_features: int = 5000, ngram_range=(1, 2)):
        self.max_features = max_features
        self.ngram_range = ngram_range

        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            max_features=self.max_features,
            ngram_range=self.ngram_range,
        )

        self.movies_df = None
        self.tfidf_matrix = None

        self.item_map = {}
        self.reverse_item_map = {}

        # optional: lưu lịch sử user nếu fit thêm ratings
        self.user_history = defaultdict(set)

    def _validate_movies_input(self, movies_df: pd.DataFrame) -> None:
        required_cols = {"MovieID", "Title", "Genres"}
        missing = required_cols - set(movies_df.columns)
        if missing:
            raise ValueError(f"Thiếu các cột bắt buộc trong movies_df: {missing}")

    def _validate_ratings_input(self, ratings_df: pd.DataFrame) -> None:
        required_cols = {"UserID", "MovieID", "Rating"}
        missing = required_cols - set(ratings_df.columns)
        if missing:
            raise ValueError(f"Thiếu các cột bắt buộc trong ratings_df: {missing}")

    def _preprocess_text(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()

        df["Title"] = df["Title"].fillna("").astype(str)
        df["Genres"] = df["Genres"].fillna("").astype(str)

        # "Action|Adventure" -> "Action Adventure"
        df["Genres_Clean"] = df["Genres"].str.replace("|", " ", regex=False)

        # tăng trọng số thể loại bằng cách lặp lại genres
        documents = (
            df["Title"] + " " +
            df["Genres_Clean"] + " " +
            df["Genres_Clean"]
        )

        return documents.str.strip()

    def fit(self, movies_df: pd.DataFrame, ratings_df: pd.DataFrame | None = None):
        """
        Fit content model từ movies_df.
        Nếu truyền ratings_df, sẽ lưu user history để hỗ trợ recommend theo user.
        """
        self._validate_movies_input(movies_df)

        self.movies_df = movies_df[["MovieID", "Title", "Genres"]].copy().reset_index(drop=True)

        movie_ids = self.movies_df["MovieID"].tolist()
        self.item_map = {movie_id: idx for idx, movie_id in enumerate(movie_ids)}
        self.reverse_item_map = {idx: movie_id for idx, movie_id in enumerate(movie_ids)}

        print("   [Model] Đang xây dựng không gian TF-IDF...")
        documents = self._preprocess_text(self.movies_df)
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

        print(f"   [Model] TF-IDF matrix shape: {self.tfidf_matrix.shape}")

        if ratings_df is not None:
            self._validate_ratings_input(ratings_df)
            self._build_user_history(ratings_df)

        print("   [Model] Content-Based Recommender đã sẵn sàng.")
        return self

    def _build_user_history(self, ratings_df: pd.DataFrame, min_rating: float = 4.0):
        """
        Lưu các phim user đã thích để hỗ trợ recommend theo user profile.
        """
        self.user_history = defaultdict(set)

        liked_df = ratings_df[ratings_df["Rating"] >= min_rating].copy()
        liked_df = liked_df[liked_df["MovieID"].isin(self.item_map)]

        for row in liked_df.itertuples():
            self.user_history[row.UserID].add(row.MovieID)

    def recommend_similar_items(self, movie_id: int, top_k: int = 10) -> pd.DataFrame:
        """
        Recommend các phim tương tự 1 movie_id.
        """
        if self.tfidf_matrix is None:
            raise ValueError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")

        if movie_id not in self.item_map:
            return pd.DataFrame(columns=["MovieID", "Title", "Genres", "SimilarityScore"])

        idx = self.item_map[movie_id]

        sim_scores = linear_kernel(self.tfidf_matrix[idx:idx + 1], self.tfidf_matrix).flatten()

        top_indices = sim_scores.argsort()[-(top_k + 1):][::-1]
        top_indices = [i for i in top_indices if i != idx][:top_k]

        result_df = self.movies_df.iloc[top_indices].copy()
        result_df["SimilarityScore"] = sim_scores[top_indices]
        result_df = result_df.reset_index(drop=True)

        return result_df

    def recommend_for_user(self, user_id: int, top_k: int = 10) -> pd.DataFrame:
        """
        Recommend phim cho user dựa trên profile nội dung từ các phim user đã thích.
        Phù hợp để demo web app.
        """
        if self.tfidf_matrix is None:
            raise ValueError("Mô hình chưa được huấn luyện. Hãy gọi fit() trước.")

        if user_id not in self.user_history or len(self.user_history[user_id]) == 0:
            return pd.DataFrame(columns=["MovieID", "Title", "Genres", "ProfileScore"])

        liked_movie_ids = [m for m in self.user_history[user_id] if m in self.item_map]
        if not liked_movie_ids:
            return pd.DataFrame(columns=["MovieID", "Title", "Genres", "ProfileScore"])

        liked_indices = [self.item_map[m] for m in liked_movie_ids]

        # user profile = trung bình vector TF-IDF của các phim user thích
        user_profile = self.tfidf_matrix[liked_indices].mean(axis=0)
        user_profile = csr_matrix(user_profile)

        scores = linear_kernel(user_profile, self.tfidf_matrix).flatten()

        # loại các phim user đã thích/xem
        scores[liked_indices] = -np.inf

        top_indices = scores.argsort()[-top_k:][::-1]

        result_df = self.movies_df.iloc[top_indices].copy()
        result_df["ProfileScore"] = scores[top_indices]
        result_df = result_df.reset_index(drop=True)

        return result_df

    def save(self, save_path: str) -> None:
        with open(save_path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(model_path: str):
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model
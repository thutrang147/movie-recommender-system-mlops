import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_squared_error, mean_absolute_error
from src.models.metrics import evaluate_explicit_predictions, evaluate_top_k_recommendations

class RobustSparseSVD:
    """
    Mô hình SVD được tối ưu hóa cho MLOps:
    1. Chống tràn RAM (OOM) bằng scipy.sparse.csr_matrix
    2. Khử nhiễu User/Item Bias
    3. Xử lý Cold-Start bằng Baseline Predictor
    """
    def __init__(self, k=20):
        self.k = k
        self.global_mean = 0
        self.user_bias = {}
        self.item_bias = {}
        self.U = None
        self.sigma = None
        self.Vt = None
        self.user_map = {}
        self.item_map = {}

    def fit(self, df):
        print("   [Model] Đang khởi tạo bộ ánh xạ ID...")
        # 1. Map ID liên tục để nạp vào tọa độ ma trận
        users = df['UserID'].unique()
        items = df['MovieID'].unique()
        self.user_map = {u: i for i, u in enumerate(users)}
        self.item_map = {i: j for j, i in enumerate(items)}

        n_users = len(users)
        n_items = len(items)

        print("   [Model] Đang tính toán User/Item Bias...")
        # 2. Tính toán Bias để khử nhiễu
        self.global_mean = df['Rating'].mean()
        
        user_means = df.groupby('UserID')['Rating'].mean()
        self.user_bias = (user_means - self.global_mean).to_dict()
        
        item_means = df.groupby('MovieID')['Rating'].mean()
        self.item_bias = (item_means - self.global_mean).to_dict()

        print("   [Model] Đang xây dựng Ma trận thưa (Sparse Matrix)...")
        # 3. Tạo Ma trận thưa (Chỉ tốn vài MB RAM thay vì hàng GB)
        # Lấy tọa độ
        rows = df['UserID'].map(self.user_map).values
        cols = df['MovieID'].map(self.item_map).values
        
        # Lấy Bias tương ứng cho từng dòng dữ liệu
        user_b = np.array([self.user_bias.get(u, 0) for u in df['UserID']])
        item_b = np.array([self.item_bias.get(i, 0) for i in df['MovieID']])
        
        # Chỉ nạp phần Thặng dư (Residual) vào SVD để phân tích phần lõi sở thích
        residuals = df['Rating'].values - (self.global_mean + user_b + item_b)

        # Khởi tạo ma trận nén (Compressed Sparse Row matrix)
        sparse_R = csr_matrix((residuals, (rows, cols)), shape=(n_users, n_items))

        print(f"   [Model] Bắt đầu phân rã SVD với k={self.k}...")
        # 4. Phân rã ma trận
        actual_k = min(self.k, min(n_users, n_items) - 1)
        self.U, sigma_vals, self.Vt = svds(sparse_R, k=actual_k)
        self.sigma = np.diag(sigma_vals)
        
        return self

    def predict(self, user_id, movie_id):
        """Dự đoán cho 1 cặp điểm. Có cơ chế chặn Cold Start."""
        b_u = self.user_bias.get(user_id, 0)
        b_i = self.item_bias.get(movie_id, 0)
        baseline = self.global_mean + b_u + b_i

        # Nếu user và movie đều đã tồn tại trong lịch sử
        if user_id in self.user_map and movie_id in self.item_map:
            u_idx = self.user_map[user_id]
            i_idx = self.item_map[movie_id]
            # Công thức: Baseline + Tương tác ẩn (U * Sigma * Vt)
            interaction = np.dot(np.dot(self.U[u_idx, :], self.sigma), self.Vt[:, i_idx])
            pred = baseline + interaction
        else:
            # Xử lý Cold Start: Người lạ/Phim lạ thì chỉ dùng Baseline để đoán
            pred = baseline

        # Chặn giá trị ảo (Không để rating < 1 hoặc > 5)
        return np.clip(pred, 1, 5)

    def evaluate(self, test_df):
        rmse = evaluate_explicit_predictions(
            test_df=test_df,
            predict_fn=self.predict,
        )
        return rmse, None

    def evaluate_ranking(self, train_df, test_df, top_k=10, relevance_threshold=4.0):
        reverse_item_map = {idx: i_id for i_id, idx in self.item_map.items()}
        item_biases = np.zeros(len(self.item_map))
        for i_id, i_idx in self.item_map.items():
            item_biases[i_idx] = self.item_bias.get(i_id, 0)

        def recommendation_fn(user_id, seen_items, top_k):
            if user_id not in self.user_map:
                return []

            u_idx = self.user_map[user_id]
            interaction = np.dot(np.dot(self.U[u_idx, :], self.sigma), self.Vt)
            baseline = self.global_mean + self.user_bias.get(user_id, 0)
            preds = baseline + item_biases + interaction

            seen_indices = [self.item_map[i] for i in seen_items if i in self.item_map]
            if seen_indices:
                preds[seen_indices] = -np.inf

            top_indices = preds.argsort()[-top_k:][::-1]
            return [reverse_item_map[idx] for idx in top_indices]

        return evaluate_top_k_recommendations(
            train_df=train_df,  
            test_df=test_df,
            recommendation_fn=recommendation_fn,
            catalog_size=len(self.item_map),
            top_k=top_k,
            relevance_threshold=relevance_threshold,
        )
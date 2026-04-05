import random
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from src.models.metrics import evaluate_top_k_recommendations


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class BPRMFModel(nn.Module):
    def __init__(self, num_users: int, num_items: int, factors: int = 64):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, factors)
        self.item_emb = nn.Embedding(num_items, factors)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        nn.init.normal_(self.user_emb.weight, std=0.01)
        nn.init.normal_(self.item_emb.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor):
        u = self.user_emb(users)
        i = self.item_emb(pos_items)
        j = self.item_emb(neg_items)

        bu = self.user_bias(users).squeeze(-1)
        bi = self.item_bias(pos_items).squeeze(-1)
        bj = self.item_bias(neg_items).squeeze(-1)

        pos_scores = (u * i).sum(dim=1) + bu + bi
        neg_scores = (u * j).sum(dim=1) + bu + bj
        return pos_scores, neg_scores

    def get_user_scores(self, users: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(users)  # [B, F]
        scores = torch.matmul(u, self.item_emb.weight.T)  # [B, I]
        scores = scores + self.item_bias.weight.squeeze(-1).unsqueeze(0)
        scores = scores + self.user_bias(users)
        return scores


class PyTorchBPR:
    def __init__(
        self,
        factors: int = 64,
        learning_rate: float = 1e-3,
        epochs: int = 30,
        batch_size: int = 2048,
        reg: float = 1e-5,
        positive_threshold: float = 4.0,
        top_k: int = 10,
        patience: int = 5,
        neg_alpha: float = 0.75,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.factors = factors
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg = reg
        self.positive_threshold = positive_threshold
        self.top_k = top_k
        self.patience = patience
        self.neg_alpha = neg_alpha
        self.seed = seed
        self.verbose = verbose

        set_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.user_map = {}
        self.item_map = {}
        self.reverse_user_map = {}
        self.reverse_item_map = {}
        self.num_users = 0
        self.num_items = 0

        self.user_history = defaultdict(set)     # positive items
        self.all_train_seen = defaultdict(set)   # all seen train items
        self.item_sampling_probs = None

        self.best_epoch = 0
        self.best_val_map = -np.inf

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)

    def _build_mappings(self, train_df: pd.DataFrame) -> None:
        unique_users = train_df["UserID"].unique()
        unique_items = train_df["MovieID"].unique()

        self.user_map = {u: idx for idx, u in enumerate(unique_users)}
        self.item_map = {i: idx for idx, i in enumerate(unique_items)}
        self.reverse_user_map = {idx: u for u, idx in self.user_map.items()}
        self.reverse_item_map = {idx: i for i, idx in self.item_map.items()}

        self.num_users = len(unique_users)
        self.num_items = len(unique_items)

    def _prepare_training_data(self, train_df: pd.DataFrame):
        pos_df = train_df[train_df["Rating"] >= self.positive_threshold].copy()
        if pos_df.empty:
            raise ValueError(
                f"Không có positive interactions nào với threshold >= {self.positive_threshold}."
            )

        self.user_history = defaultdict(set)
        self.all_train_seen = defaultdict(set)

        for row in train_df.itertuples():
            if row.UserID in self.user_map and row.MovieID in self.item_map:
                u_idx = self.user_map[row.UserID]
                i_idx = self.item_map[row.MovieID]
                self.all_train_seen[u_idx].add(i_idx)

        train_pairs = []
        pair_weights = []

        for row in pos_df.itertuples():
            u_idx = self.user_map[row.UserID]
            i_idx = self.item_map[row.MovieID]
            self.user_history[u_idx].add(i_idx)

            weight = max(float(row.Rating) - (self.positive_threshold - 1.0), 1.0)
            train_pairs.append((u_idx, i_idx))
            pair_weights.append(weight)

        item_counts = train_df["MovieID"].value_counts()
        probs = np.zeros(self.num_items, dtype=np.float64)
        for item_id, count in item_counts.items():
            if item_id in self.item_map:
                probs[self.item_map[item_id]] = float(count)

        probs = np.power(probs + 1e-12, self.neg_alpha)
        probs = probs / probs.sum()
        self.item_sampling_probs = probs

        return train_pairs, np.asarray(pair_weights, dtype=np.float32)

    def _sample_negative(self, u_idx: int) -> int:
        while True:
            neg_i = np.random.choice(self.num_items, p=self.item_sampling_probs)
            if neg_i not in self.user_history[u_idx]:
                return int(neg_i)

    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame | None = None):
        set_seed(self.seed)
        self._log(f"   [System] PyTorch đang sử dụng thiết bị: {self.device}")
        self._log("   [Model] Đang khởi tạo bộ ánh xạ ID (Mapping)...")

        self._build_mappings(train_df)
        train_pairs, pair_weights = self._prepare_training_data(train_df)

        self._log(
            f"   [Data] Positive pairs: {len(train_pairs):,} | Users: {self.num_users:,} | Items: {self.num_items:,}"
        )

        self.model = BPRMFModel(
            num_users=self.num_users,
            num_items=self.num_items,
            factors=self.factors,
        ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.reg)

        best_state = None
        self.best_val_map = -np.inf
        self.best_epoch = 0
        bad_epochs = 0

        self._log(f"   [Model] Bắt đầu huấn luyện BPR-MF ({self.epochs} epochs)...")

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            indices = np.arange(len(train_pairs))
            np.random.shuffle(indices)

            total_loss = 0.0
            num_batches = 0

            for start in range(0, len(indices), self.batch_size):
                batch_idx = indices[start:start + self.batch_size]
                batch = [train_pairs[i] for i in batch_idx]
                batch_weights = pair_weights[batch_idx]

                users, pos_items, neg_items = [], [], []
                for u_idx, pos_i in batch:
                    neg_i = self._sample_negative(u_idx)
                    users.append(u_idx)
                    pos_items.append(pos_i)
                    neg_items.append(neg_i)

                users_t = torch.tensor(users, dtype=torch.long, device=self.device)
                pos_t = torch.tensor(pos_items, dtype=torch.long, device=self.device)
                neg_t = torch.tensor(neg_items, dtype=torch.long, device=self.device)
                w_t = torch.tensor(batch_weights, dtype=torch.float32, device=self.device)

                optimizer.zero_grad()
                pos_scores, neg_scores = self.model(users_t, pos_t, neg_t)

                base_loss = -torch.nn.functional.logsigmoid(pos_scores - neg_scores)
                loss = (base_loss * w_t).mean()

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)

            if val_df is not None:
                _, val_map, _ = self.evaluate_ranking(
                    train_df=train_df,
                    test_df=val_df,
                    top_k=self.top_k,
                    relevance_threshold=self.positive_threshold,
                    verbose=False,
                )

                self._log(
                    f"      -> Epoch {epoch:02d}/{self.epochs} | Loss: {avg_loss:.4f} | Val MAP@{self.top_k}: {val_map:.4f}"
                )

                if val_map > self.best_val_map:
                    self.best_val_map = val_map
                    self.best_epoch = epoch
                    best_state = {
                        "model_state_dict": {
                            k: v.detach().cpu().clone()
                            for k, v in self.model.state_dict().items()
                        }
                    }
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.patience:
                        self._log(
                            f"   [EarlyStopping] Dừng sớm tại epoch {epoch}. Best epoch={self.best_epoch}, Best Val MAP@{self.top_k}: {self.best_val_map:.4f}"
                        )
                        break
            else:
                self.best_epoch = epoch
                self._log(f"      -> Epoch {epoch:02d}/{self.epochs} | Loss: {avg_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state["model_state_dict"])
            self._log(
                f"   [Model] Đã khôi phục trọng số tốt nhất tại epoch={self.best_epoch} với Val MAP@{self.top_k}: {self.best_val_map:.4f}"
            )

        return self

    def evaluate(self, test_df: pd.DataFrame):
        return None, None


    def evaluate_ranking(self, train_df, test_df, top_k=10, relevance_threshold=4.0, verbose=True):
        if self.model is None:
            raise ValueError("Model chưa được fit.")

        if verbose:
            print(f"   [Metrics] Đang tính toán Top-{top_k} Recommendation Metrics (BPR)...")

        def recommendation_fn(user_id, seen_items, top_k):
            if user_id not in self.user_map:
                return []

            u_idx = self.user_map[user_id]

            self.model.eval()
            with torch.no_grad():
                users_t = torch.tensor([u_idx], dtype=torch.long, device=self.device)
                scores = self.model.get_user_scores(users_t).cpu().numpy().flatten()

            seen_indices = [self.item_map[i] for i in seen_items if i in self.item_map]
            if seen_indices:
                scores[seen_indices] = -np.inf

            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

            return [self.reverse_item_map[i] for i in top_indices.tolist()]

        recall_k, map_k, coverage = evaluate_top_k_recommendations(
            train_df=train_df,
            test_df=test_df,
            recommendation_fn=recommendation_fn,
            catalog_size=self.num_items,
            top_k=top_k,
            relevance_threshold=relevance_threshold,
        )

        if verbose:
            ground_truth = build_relevant_items_dict(test_df, relevance_threshold)
            eval_users = [u for u in ground_truth if u in self.user_map]
            print(f"      -> Số user được evaluate: {len(eval_users):,}")

        return recall_k, map_k, coverage

    def recommend_top_k(self, user_id, k: int = 10):
        """
        Dùng trực tiếp cho web app, không cần truyền train_df.
        self.all_train_seen đã được restore từ checkpoint.
        """
        if self.model is None:
            raise ValueError("Model chưa được load/fit.")
        if user_id not in self.user_map:
            return []

        self.model.eval()
        u_idx = self.user_map[user_id]
        seen_items = list(self.all_train_seen.get(u_idx, set()))

        with torch.no_grad():
            users_t = torch.tensor([u_idx], dtype=torch.long, device=self.device)
            scores = self.model.get_user_scores(users_t).cpu().numpy().flatten()

        if seen_items:
            scores[seen_items] = -np.inf

        top_items = np.argpartition(scores, -k)[-k:]
        top_items = top_items[np.argsort(scores[top_items])[::-1]]

        return [self.reverse_item_map[i] for i in top_items.tolist()]
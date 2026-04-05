import os
import sys
import pandas as pd
import torch

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.data_loader import load_processed_data, time_based_split
from src.models.bpr_recommender import PyTorchBPR


def save_bpr_checkpoint(model: PyTorchBPR, save_path: str) -> None:
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    checkpoint = {
        "model_state_dict": model.model.state_dict(),
        "user_map": model.user_map,
        "item_map": model.item_map,
        "reverse_user_map": model.reverse_user_map,
        "reverse_item_map": model.reverse_item_map,
        "num_users": model.num_users,
        "num_items": model.num_items,
        "all_train_seen": dict(model.all_train_seen),
        "config": {
            "factors": model.factors,
            "learning_rate": model.lr,
            "epochs": model.epochs,
            "batch_size": model.batch_size,
            "reg": model.reg,
            "positive_threshold": model.positive_threshold,
            "top_k": model.top_k,
            "patience": model.patience,
            "neg_alpha": model.neg_alpha,
            "seed": model.seed,
        },
        "best_epoch": model.best_epoch,
        "best_val_map": model.best_val_map,
    }
    torch.save(checkpoint, save_path)


def run_bpr_training_pipeline():
    print("🚀 [Pipeline] Khởi động huấn luyện PyTorch BPR...")

    data_path = "data/processed/ratings.parquet"
    df = load_processed_data(data_path)

    train_df, val_df, test_df = time_based_split(
        df,
        val_size=0.1,
        test_size=0.1,
        filter_cold_items=True,
    )

    print(
        f"   -> Kích thước: Train ({len(train_df):,}) | "
        f"Val ({len(val_df):,}) | Test ({len(test_df):,})"
    )

    print("\n🔍 [Training] Huấn luyện BPR với Validation...")
    tuning_model = PyTorchBPR(
        factors=64,
        learning_rate=1e-3,
        epochs=30,
        batch_size=2048,
        reg=1e-5,
        positive_threshold=4.0,
        top_k=10,
        patience=5,
        neg_alpha=0.75,
        seed=42,
        verbose=True,
    )
    tuning_model.fit(train_df, val_df)

    print(
        f"\n🏆 [Selection] Best epoch={tuning_model.best_epoch} | "
        f"Best Val MAP@10={tuning_model.best_val_map:.4f}"
    )

    print("\n🚀 [Retrain] Gộp Train + Val để huấn luyện final model...")
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    final_epochs = max(1, tuning_model.best_epoch)

    final_model = PyTorchBPR(
        factors=64,
        learning_rate=1e-3,
        epochs=final_epochs,
        batch_size=2048,
        reg=1e-5,
        positive_threshold=4.0,
        top_k=10,
        patience=5,
        neg_alpha=0.75,
        seed=42,
        verbose=True,
    )
    final_model.fit(full_train_df, val_df=None)

    print("\n🎯 [Testing] Đánh giá trên tập Test...")
    recall_10, map_10, coverage = final_model.evaluate_ranking(
        train_df=full_train_df,
        test_df=test_df,
        top_k=10,
        relevance_threshold=4.0,
        verbose=True,
    )

    print(f"   => Recall@10: {recall_10:.4f}")
    print(f"   => MAP@10: {map_10:.4f}")
    print(f"   => Coverage: {coverage:.4f}")

    save_path = "models/pytorch_bpr_ranker.pt"
    save_bpr_checkpoint(final_model, save_path)

    print(f"\n✅ Đã lưu checkpoint BPR tại: {save_path}")


if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    run_bpr_training_pipeline()
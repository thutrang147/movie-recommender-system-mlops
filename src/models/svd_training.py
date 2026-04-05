import os
import sys
import pickle
import pandas as pd

# Thiết lập đường dẫn root để chạy bằng nút Play trên VS Code không bị lỗi ModuleNotFoundError
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.data.data_loader import load_processed_data, time_based_split
from src.models.svd_recommender import RobustSparseSVD

def run_training_pipeline():
    print("🚀 [Pipeline] Khởi động quy trình huấn luyện Baseline SVD...")
    
    # 1. NẠP VÀ CHIA DỮ LIỆU
    # Đảm bảo đường dẫn này khớp với cấu hình của bạn
    data_path = "data/processed/ratings.parquet" 
    df = load_processed_data(data_path)
    
    train_df, val_df, test_df = time_based_split(df, val_size=0.1, test_size=0.1, filter_cold_items=True)

    # 2. TUNING TRÊN TẬP VALIDATION
    k_candidates = [10, 20, 50, 100]
    best_k = 0
    best_map_val = -1

    print(f"\n🔍 [Validation] Đang tìm tham số k tốt nhất trên tập Validation: {k_candidates}")
    for k in k_candidates:
        print(f"   --- Thử nghiệm k = {k} ---")
        model = RobustSparseSVD(k=k)
        model.fit(train_df)
        
        _, map_10_val, _ = model.evaluate_ranking(train_df, val_df, top_k=10, relevance_threshold=4.0)
        print(f"   => [Val] MAP@10: {map_10_val:.4f}")
        
        if map_10_val > best_map_val:
            best_map_val = map_10_val
            best_k = k

    print(f"\n🏆 Thông số k tối ưu là: k={best_k}")

    # 3. RETRAIN (GỘP TRAIN + VAL)
    print("\n🚀 [Retrain] Đang gộp Train + Val để huấn luyện lần cuối trước khi Test...")
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)
    
    final_model = RobustSparseSVD(k=best_k)
    final_model.fit(full_train_df)

    # 4. KIỂM THỬ KHÁCH QUAN (TESTING)
    print("\n🎯 ĐÁNH GIÁ KHÁCH QUAN TRÊN TẬP TEST (UNSEEN DATA):")
    rmse_test, _ = final_model.evaluate(test_df)
    recall_10_test, map_10_test, cov_test = final_model.evaluate_ranking(full_train_df, test_df, top_k=10, relevance_threshold=4.0)
    
    print(f"   => RMSE: {rmse_test:.4f}")
    print(f"   => Recall@10: {recall_10_test:.4f} | MAP@10: {map_10_test:.4f} | Coverage: {cov_test:.4f}")

    # 5. LƯU TRỮ MÔ HÌNH BASELINE
    save_path = "models/robust_svd_baseline.pkl"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(final_model, f)
    print(f"\n✅ Trạng thái mô hình SVD Baseline đã được lưu trữ thành công tại: {save_path}")

if __name__ == "__main__":
    run_training_pipeline()
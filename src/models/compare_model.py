import os
import pickle
import sys
import pandas as pd
from src.models.svd_recommender import RobustSparseSVD
from src.models.bpr_recommender import PyTorchBPR

# Thêm thư mục gốc vào path để Python tìm thấy các module trong src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sklearn.model_selection import train_test_split
from src.data.data_loader import load_config, load_processed_data, time_based_split
from src.models.svd_recommender import RobustSparseSVD

def run_training_pipeline():
    print("🚀 [Pipeline] Khởi động quy trình huấn luyện Baseline...")

    # 1. Nạp cấu hình từ file YAML
    config = load_config("configs/config.yaml")
    
# 2. Nạp dữ liệu & Chia tập (Train/Val/Test)
    df = load_processed_data(config['data']['processed_path'])
    
    # Cấu hình cứng tạm thời (hoặc bạn có thể thêm val_size vào config.yaml)
    train_df, val_df, test_df = time_based_split(df, val_size=0.1, test_size=0.1, filter_cold_items=True)

    # 3. TUNING (Tối ưu Hyperparameter) TRÊN TẬP VALIDATION
    k_candidates = config['model']['k_candidates']
    best_k = 0
    best_map_val = -1
    best_model = None

    print(f"\n🔍 [Validation] Đang tìm tham số k tốt nhất trên tập Validation: {k_candidates}")
    for k in k_candidates:
        print(f"   --- Thử nghiệm k = {k} ---")
        model = RobustSparseSVD(k=k)
        model.fit(train_df) # Chỉ học trên Train
        
        # CHỈ ĐÁNH GIÁ TRÊN VAL
        _, map_10_val, _ = model.evaluate_ranking(train_df, val_df, top_k=10, relevance_threshold=4.0)
        print(f"   => [Val] MAP@10: {map_10_val:.4f}")
        
        if map_10_val > best_map_val:
            best_map_val = map_10_val
            best_k = k
            best_model = model

    print(f"\nThông số k tối ưu  là: k={best_k}")
    print("\n🚀 [Retrain] Đang gộp Train + Val để huấn luyện lần cuối trước khi Test...")
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Tạo một model mới tinh với k tốt nhất và cho học trên toàn bộ dữ liệu quá khứ
    final_model = RobustSparseSVD(k=best_k)
    final_model.fit(full_train_df)

# =====================================================================
    # TIẾN TRÌNH ĐÁNH GIÁ ĐỐI CHỨNG (BENCHMARKING): SVD (BASELINE) vs PyTorch BPR
    # =====================================================================
    print("\n" + "="*60)
    print("KHỞI CHẠY QUÁ TRÌNH ĐÁNH GIÁ ĐỐI CHỨNG (RANKING BENCHMARK)")
    print("="*60)

    # Tích hợp tập huấn luyện (Train) và tập kiểm chứng (Validation) nhằm tối ưu hóa 
    # trọng số mô hình trước giai đoạn kiểm thử cuối cùng.
    full_train_df = pd.concat([train_df, val_df], ignore_index=True)

    # 1. HUẤN LUYỆN MÔ HÌNH CƠ SỞ (BASELINE MODEL - SVD)
    print("\n[1/2] Đang tiến hành huấn luyện mô hình cơ sở SVD...")
    svd_model = RobustSparseSVD(k=best_k) 
    svd_model.fit(full_train_df)
    
    # 2. HUẤN LUYỆN MÔ HÌNH ĐỀ XUẤT (PROPOSED MODEL - PyTorch BPR)
    print("\n[2/2] Đang tiến hành huấn luyện mô hình mạng nơ-ron PyTorch BPR...")
    bpr_model = PyTorchBPR(factors=64, epochs=15) 
    bpr_model.fit(full_train_df)

    # 3. ĐO LƯỜNG HIỆU NĂNG TRÊN TẬP DỮ LIỆU KIỂM THỬ (TEST SET / UNSEEN DATA)
    print("\nĐANG TRÍCH XUẤT CÁC CHỈ SỐ ĐÁNH GIÁ TRÊN TẬP KIỂM THỬ...")
    
    # Đánh giá hiệu năng xếp hạng của mô hình SVD
    _, svd_map, svd_cov = svd_model.evaluate_ranking(full_train_df, test_df, top_k=10, relevance_threshold=4.0)
    
    # Đánh giá hiệu năng xếp hạng của mô hình BPR
    _, bpr_map, bpr_cov = bpr_model.evaluate_ranking(full_train_df, test_df, top_k=10, relevance_threshold=4.0)

    # =====================================================================
    # BÁO CÁO TỔNG HỢP HIỆU NĂNG MÔ HÌNH (MODEL PERFORMANCE SUMMARY)
    # =====================================================================
    print("\n" + "="*60)
    print(f"{'BẢNG ĐỐI CHỨNG HIỆU NĂNG GỢI Ý TOP-10 (TOP-10 RECOMMENDATION)':^60}")
    print("="*60)
    print(f"{'Mô hình (Model)':<20} | {'MAP@10':<15} | {'Độ phủ (Coverage)':<15}")
    print("-" * 60)
    print(f"{'SVD (Baseline)':<20} | {svd_map:.4f} ({svd_map*100:.2f}%) | {svd_cov:.4f} ({svd_cov*100:.2f}%)")
    print(f"{'PyTorch BPR':<20} | {bpr_map:.4f} ({bpr_map*100:.2f}%) | {bpr_cov:.4f} ({bpr_cov*100:.2f}%)")
    print("-" * 60)
    
    # Đo lường mức độ cải thiện tương đối (Relative Uplift)
    if svd_map > 0:
        uplift = ((bpr_map - svd_map) / svd_map) * 100
        print(f"Ghi nhận mức độ cải thiện (Uplift) MAP@10 của PyTorch BPR: +{uplift:.1f}% so với SVD")
    print("="*60)

    # Lưu trữ trạng thái mô hình (Serialization)
    save_path = config['model']['save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'wb') as f:
        pickle.dump(bpr_model, f)
    print(f"\n✅ Trạng thái mô hình BPR đã được xuất và lưu trữ thành công tại: {save_path}")

if __name__ == "__main__":
    run_training_pipeline()
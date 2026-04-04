import os
import pickle
import sys

# Thêm thư mục gốc vào path để Python tìm thấy các module trong src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from sklearn.model_selection import train_test_split
from src.data.data_loader import load_config, load_processed_data
from src.models.svd_recommender import RobustSparseSVD

def run_training_pipeline():
    print("🚀 [Pipeline] Khởi động quy trình huấn luyện Baseline...")

    # 1. Nạp cấu hình từ file YAML
    config = load_config("configs/config.yaml")
    
    # 2. Nạp dữ liệu qua Data Loader
    df = load_processed_data(config['data']['processed_path'])
    
    # 3. Chia tập Train/Test để kiểm soát Overfitting
    train_df, test_df = train_test_split(
        df, 
        test_size=config['training']['test_size'], 
        random_state=config['training']['random_state']
    )
    print(f"   [Data] Tập Train: {len(train_df):,} | Tập Test: {len(test_df):,}")

    # 4. Grid Search tìm tham số k tối ưu dựa trên MAP@10
    k_candidates = config['model']['k_candidates']
    best_k = 0
    best_map = -1  # MAP càng cao càng tốt
    best_model = None

    print(f"\n🔍 [Optimization] Đang rà soát các tham số k: {k_candidates}")
    for k in k_candidates:
        print(f"   --- Thử nghiệm k = {k} ---")
        model = RobustSparseSVD(k=k)
        model.fit(train_df)
        
        # Đo lường RMSE phụ
        rmse, _ = model.evaluate(test_df)
        
        # Đo lường Ranking Metrics chính (Top 10)
        recall_10, map_10, coverage = model.evaluate_ranking(train_df, test_df, top_k=10, relevance_threshold=4.0)
        
        print(f"   => [Metrics] RMSE: {rmse:.4f} | Recall@10: {recall_10:.4f} | MAP@10: {map_10:.4f} | Coverage: {coverage:.4f}")
        
        # Tối ưu hóa theo MAP@10 thay vì RMSE
        if map_10 > best_map:
            best_map = map_10
            best_k = k
            best_model = model

    print(f"\n🏆 [Result] Mô hình xuất sắc nhất đạt MAP@10: {best_map:.4f} (k={best_k})")

    # 5. Lưu trữ Model chuẩn MLOps
    save_path = config['model']['save_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'wb') as f:
        pickle.dump(best_model, f)
        
    print(f"💾 [Output] Model đã được đóng gói tại: {save_path}")
    print("\n✅ TẤT CẢ TÁC VỤ HOÀN TẤT. BẠN ĐÃ CÓ MỘT BASELINE HOÀN HẢO!")

if __name__ == "__main__":
    run_training_pipeline()
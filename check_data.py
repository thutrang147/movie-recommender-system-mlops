import os
import pandas as pd

def inspect_parquet_file(file_path):
    print(f"\n" + "="*70)
    print(f"📊 ĐANG PHÂN TÍCH TỆP: {file_path}")
    print("="*70)
    
    if not os.path.exists(file_path):
        print(f"❌ LỖI: Không tìm thấy tệp tại đường dẫn này.")
        return
        
    try:
        # Nạp dữ liệu
        df = pd.read_parquet(file_path)
        
        # 1. Kích thước tổng quan
        print(f"✅ Kích thước dữ liệu (Shape): {df.shape[0]:,} dòng | {df.shape[1]} cột")
        
        # 2. Cấu trúc Schema (Tên cột và Kiểu dữ liệu)
        print("\n📋 CẤU TRÚC LƯỚI (SCHEMA):")
        print("-" * 40)
        for col in df.columns:
            # Hiển thị số lượng giá trị thiếu (Null) nếu có
            null_count = df[col].isnull().sum()
            null_warning = f"(Cảnh báo: {null_count:,} dòng bị Null)" if null_count > 0 else ""
            print(f"   🔸 {col:<15} | Type: {str(df[col].dtype):<10} {null_warning}")
            
        # 3. Trích xuất mẫu dữ liệu
        print("\n🔍 TRÍCH XUẤT 3 DÒNG DỮ LIỆU ĐẦU TIÊN (HEAD):")
        print("-" * 40)
        print(df.head(3).to_string())
        print("="*70)
        
    except Exception as e:
        print(f"❌ LỖI KHI ĐỌC TỆP: {e}")

if __name__ == "__main__":
    # Danh sách các tệp nghi ngờ chứa thông tin phim (bạn có thể thêm bớt đường dẫn)
    target_files = [
        "data/processed/movies.parquet",
        "data/raw/movies.parquet",
        "data/processed/movies_metadata.parquet",
        "data/processed/ratings.parquet" # Xem luôn ratings cho chắc
    ]
    
    for file in target_files:
        inspect_parquet_file(file)
import pandas as pd
import os

# Đường dẫn chuẩn theo cấu trúc repo
RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed"

def ingest_data():
    # Đảm bảo thư mục đầu ra tồn tại
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    print("Đang đọc dữ liệu thô (.dat)...")
    # Định nghĩa Cột theo chuẩn MovieLens 1M
    users_cols =['UserID', 'Gender', 'Age', 'Occupation', 'ZipCode']
    movies_cols = ['MovieID', 'Title', 'Genres']
    ratings_cols = ['UserID', 'MovieID', 'Rating', 'Timestamp']

    # Đọc data (MovieLens dùng '::' làm dấu phân cách)
    users = pd.read_csv(f"{RAW_DIR}/users.dat", sep='::', engine='python', names=users_cols, encoding='latin-1')
    movies = pd.read_csv(f"{RAW_DIR}/movies.dat", sep='::', engine='python', names=movies_cols, encoding='latin-1')
    ratings = pd.read_csv(f"{RAW_DIR}/ratings.dat", sep='::', engine='python', names=ratings_cols, encoding='latin-1')

    print("Đang chuyển đổi và lưu sang định dạng Parquet...")
    users.to_parquet(f"{PROCESSED_DIR}/users.parquet", index=False)
    movies.to_parquet(f"{PROCESSED_DIR}/movies.parquet", index=False)
    ratings.to_parquet(f"{PROCESSED_DIR}/ratings.parquet", index=False)
    
    print(f"✅ Xong! Dữ liệu sạch đã lưu tại: {PROCESSED_DIR}")

if __name__ == "__main__":
    ingest_data()
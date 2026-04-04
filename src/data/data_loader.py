import yaml
import pandas as pd
import os

def load_config(config_path="configs/config.yaml"):
    """Đọc các tham số cấu hình từ file YAML."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"❌ Không tìm thấy file cấu hình tại: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config

def load_processed_data(data_path):
    """Nạp file dữ liệu đã qua xử lý (Parquet/CSV)."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"❌ Dữ liệu không tồn tại ở đường dẫn: {data_path}")
    
    print(f"Loading data from: {data_path}...")
    # Tự động nhận diện định dạng file dựa trên đuôi mở rộng
    if data_path.endswith('.parquet'):
        df = pd.read_parquet(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError("❌ Hàm load_data chỉ hỗ trợ file .csv hoặc .parquet")
    
    # Kiểm tra tính toàn vẹn của dữ liệu (Data Integrity Check cơ bản)
    required_columns = {'UserID', 'MovieID', 'Rating'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"❌ Dữ liệu thiếu các cột bắt buộc: {required_columns}")
        
    return df
.PHONY: install data dvc-add dvc-push setup

install:
	pip install --upgrade pip
	pip install -r requirements.txt
	@echo "=> [1/4] Cai dat thu vien thanh cong!"

data:
	python src/data/ingest.py
	@echo "=> [2/4] Xu ly va tao file Parquet thanh cong!"

dvc-add:
	dvc add data/processed/users.parquet data/processed/movies.parquet data/processed/ratings.parquet
	@echo "=> [3/4] Track file bang DVC thanh cong!"

dvc-push:
	dvc push
	@echo "=> [4/4] Day du lieu len Cloud thanh cong!"

# Lệnh gộp: Chạy tuần tự cả 4 bước trên chỉ với 1 thao tác
setup: install data dvc-add dvc-push
	@echo "=> TOAN BO HE THONG DA SAN SANG!"
	
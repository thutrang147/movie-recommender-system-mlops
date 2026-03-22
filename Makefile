.PHONY: install data dvc-add dvc-push

install:
	pip install -r requirements.txt

data:
	python src/data/ingest.py

dvc-add:
	dvc add data/processed/users.parquet data/processed/movies.parquet data/processed/ratings.parquet

dvc-push:
	dvc push
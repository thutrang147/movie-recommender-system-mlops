.PHONY: install data-csv validate report ingest-parquet data dvc-add dvc-push dvc-pull bpr-train content-train final-benchmark mlflow-log api-run api-batch monitoring-report

PYTHON_UV := $(shell command -v uv 2>/dev/null)

install:
	@if [ -n "$(PYTHON_UV)" ]; then \
		uv sync; \
	else \
		python -m pip install -e .; \
	fi

data-csv:
	@if [ -n "$(PYTHON_UV)" ]; then \
		uv run python src/data/load_data.py; \
	else \
		python src/data/load_data.py; \
	fi

validate:
	@if [ -n "$(PYTHON_UV)" ]; then \
		uv run python src/data/validate_data.py; \
	else \
		python src/data/validate_data.py; \
	fi

report:
	@if [ -n "$(PYTHON_UV)" ]; then \
		uv run python src/data/validate_data.py --save-report; \
	else \
		python src/data/validate_data.py --save-report; \
	fi

ingest-parquet:
	@if [ -n "$(PYTHON_UV)" ]; then \
		uv run python src/data/ingest.py; \
	else \
		python src/data/ingest.py; \
	fi

data: data-csv ingest-parquet

dvc-add:
	@if [ -n "$(PYTHON_UV)" ]; then \
		uv run dvc add data/processed/users.parquet data/processed/movies.parquet data/processed/ratings.parquet; \
	else \
		python -m dvc add data/processed/users.parquet data/processed/movies.parquet data/processed/ratings.parquet; \
	fi

dvc-push:
	@if [ -n "$(PYTHON_UV)" ]; then \
		uv run dvc push; \
	else \
		python -m dvc push; \
	fi

dvc-pull:
	@if [ -n "$(PYTHON_UV)" ]; then \
		uv run dvc pull; \
	else \
		python -m dvc pull; \
	fi

bpr-train:
	@if [ -n "$(PYTHON_UV)" ]; then \
		uv run python src/models/train_bpr.py; \
	else \
		python src/models/train_bpr.py; \
	fi

content-train:
	@if [ -n "$(PYTHON_UV)" ]; then \
		uv run python src/models/train_content_based.py; \
	else \
		python src/models/train_content_based.py; \
	fi

final-benchmark:
	@if [ -n "$(PYTHON_UV)" ]; then \
		uv run python src/models/final_benchmark.py; \
	else \
		python src/models/final_benchmark.py; \
	fi

mlflow-log:
	@if [ -n "$(PYTHON_UV)" ]; then \
		uv run python src/models/log_mlflow_benchmark.py; \
	else \
		python src/models/log_mlflow_benchmark.py; \
	fi

api-run:
	@if [ -n "$(PYTHON_UV)" ]; then \
		uv run uvicorn src.serving.app:app --host 0.0.0.0 --port 8000; \
	else \
		uvicorn src.serving.app:app --host 0.0.0.0 --port 8000; \
	fi

api-batch:
	@if [ -n "$(PYTHON_UV)" ]; then \
		uv run python src/serving/batch_recommend.py --user-ids-path data/split/test.parquet --top-k 10; \
	else \
		python src/serving/batch_recommend.py --user-ids-path data/split/test.parquet --top-k 10; \
	fi

monitoring-report:
	@if [ -n "$(PYTHON_UV)" ]; then \
		uv run python src/monitoring/report.py; \
	else \
		python src/monitoring/report.py; \
	fi

.PHONY: install data-csv validate report ingest-parquet data dvc-add dvc-push dvc-pull

PYTHON_UV := $(shell command -v uv 2>/dev/null)

install:
	@if [ -n "$(PYTHON_UV)" ]; then \
		uv sync --dev; \
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
		uv run python -m dvc add data/processed/users.parquet data/processed/movies.parquet data/processed/ratings.parquet; \
	else \
		python -m dvc add data/processed/users.parquet data/processed/movies.parquet data/processed/ratings.parquet; \
	fi

dvc-push:
	@if [ -n "$(PYTHON_UV)" ]; then \
		uv run python -m dvc push; \
	else \
		python -m dvc push; \
	fi

dvc-pull:
	@if [ -n "$(PYTHON_UV)" ]; then \
		uv run python -m dvc pull; \
	else \
		python -m dvc pull; \
	fi

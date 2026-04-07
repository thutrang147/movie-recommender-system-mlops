# Movie Recommender System (MLOps)

An MLOps-oriented MovieLens recommender system with data versioning, model training, API serving, monitoring, and retraining workflows.

## What This Repo Does

- Builds a recommendation pipeline from MovieLens raw data.
- Trains and evaluates Baseline, SVD, BPR, and Content-based models.
- Serves recommendations with FastAPI.
- Supports monitoring, drift checks, retraining, promotion, and rollback.

## Prerequisites

- macOS/Linux
- Python 3.10
- `uv` installed
- Git
- Optional: Docker Desktop

## 1) Clone And Setup

```bash
git clone <your-repo-url>
cd movie-recommender-system-mlops
uv sync
```

Sanity check:

```bash
uv run pytest -q
```

Expected: tests pass.

## 2) Choose One Run Path

### Path A (Recommended): Run fast with DVC artifacts

Use this path if you want the project running quickly with shared tracked outputs.

```bash
uv run dvc pull
uv run pytest -q
uv run python src/models/final_benchmark.py
```

Expected outputs:

- `models/personalized/*.pkl`
- `reports/evaluation/final_comparison.md`

### Path B: Full local rebuild from raw data

Use this path if you want to rebuild everything end-to-end yourself.

```bash
uv run python src/data/load_data.py
uv run python src/data/validate_data.py --save-report
uv run python src/data/preprocess.py
uv run python src/data/split.py
uv run python src/models/baseline.py
uv run python src/models/train.py
uv run python src/models/train_bpr.py
uv run python src/models/train_content_based.py
uv run python src/models/evaluate.py
uv run python src/models/final_benchmark.py
uv run pytest -q
```

Expected outputs:

- `data/split/train.parquet`, `data/split/val.parquet`, `data/split/test.parquet`
- `models/baseline/most_popular_items.parquet`
- `models/personalized/svd_model.pkl`
- `models/personalized/bpr_model.pkl`
- `models/personalized/content_based_model.pkl`
- `reports/evaluation/final_comparison.md`

## 3) Run API Locally

```bash
make api-run
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/recommend/1?top_k=10"
curl "http://127.0.0.1:8000/recommend/999999?top_k=10"
```

Expected:

- `/health` returns status and active model info.
- `/recommend/{user_id}` returns recommendation list.

## 4) Monitoring And Retraining

Generate monitoring report:

```bash
make monitoring-report
```

Run retraining flows:

```bash
make retrain-weekly
make retrain-trigger
make retrain-rollback
```

Expected outputs:

- `reports/monitoring/monitoring_report.md`
- `reports/retraining/retrain_report.md`

## 5) Docker (Optional)

```bash
docker build -t movielens-api .
docker run --rm -p 8000:8000 movielens-api
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/recommend/1?top_k=10"
```

Note: first Docker build can be slow due to dependency download.

## DVC Workflow (Team Use)

Pull shared outputs:

```bash
uv run dvc pull
```

If you changed tracked pipeline outputs:

```bash
uv run dvc repro
uv run dvc push
git add dvc.yaml dvc.lock
git commit -m "chore: update dvc pipeline metadata"
```

Check remote/config:

```bash
uv run dvc remote list
uv run dvc config --list
```

## Troubleshooting

- `dvc: command not found`: use `uv run dvc ...` instead of `dvc ...`.
- Docker API fails with missing package: rebuild image after pulling latest `Dockerfile`.
- `dvc pull` says missing on remote: ensure latest `dvc.lock` is pushed to Git and artifacts were pushed with `uv run dvc push`.

## Success Criteria (Fresh Clone Test)

From a fresh clone, the repo is considered healthy if all below pass:

1. `uv sync`
2. `uv run dvc pull`
3. `uv run pytest -q`
4. `make api-run` + successful `/health` and `/recommend`
5. `make monitoring-report`
6. `make retrain-trigger`

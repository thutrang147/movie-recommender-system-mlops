# Movie Recommender System (MLOps)

An MLOps-oriented MovieLens recommender system with data versioning, model training, API serving, monitoring, and retraining workflows.

## What This Repo Does

- Builds a recommendation pipeline from MovieLens raw data.
- Trains and evaluates Baseline, SVD, BPR, and Content-based models.
- Serves recommendations with FastAPI.
- Supports monitoring, drift checks, retraining, promotion, and rollback.

## Prerequisites

- macOS, Linux, or Windows 10/11
- Python 3.10.x (required: `>=3.10,<3.11`)
- `uv` installed
- Git
- Optional: Docker Desktop

Optional tool:

- `make` (optional convenience for macOS/Linux)

## 1) Clone And Setup

```bash
git clone <your-repo-url>
cd <repo-folder>
uv sync
```

Sanity check:

```bash
uv run pytest -q
```

Expected: tests pass.

## 2) Configure DVC Access (For Path A)

If your team uses Google Drive as DVC remote, configure credentials once on your machine.

```bash
uv run dvc remote list
uv run dvc config --list
uv run dvc remote modify --local gdrive_remote gdrive_client_id "<YOUR_CLIENT_ID>"
uv run dvc remote modify --local gdrive_remote gdrive_client_secret "<YOUR_CLIENT_SECRET>"
```

Notes:

- `--local` stores credentials in `.dvc/config.local` (git-ignored).
- Do not commit or share client secrets in chat/commit history.

## 3) Choose One Run Path

### Path A (Recommended): Run fast with DVC artifacts

Use this path if you want the project running quickly with shared tracked outputs.

```bash
uv run dvc pull
uv run pytest -q
uv run python src/models/final_benchmark.py
```

If `uv run dvc pull` fails with missing remote objects (`missing-files`), switch to Path B and rebuild locally.

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

## 4) Run API Locally

Before starting API, make sure model artifacts exist.

1) Ensure artifacts are available (choose one):

- Path A: `uv run dvc pull`
- Path B: finish full local rebuild steps in Section 3

2) Start API:

```bash
uv run uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

Note: if you are on macOS/Linux and prefer Makefile shortcuts, you can also run `make api-run`.

Open interactive API docs in browser:

- Swagger UI: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`
- OpenAPI JSON: `http://127.0.0.1:8000/openapi.json`

In another terminal:

```bash
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/recommend/1?top_k=10"
curl "http://127.0.0.1:8000/recommend/999999?top_k=10"
```

Expected:

- `/health` returns status and active model info.
- `/recommend/{user_id}` returns recommendation list.

## 5) Monitoring And Retraining

Generate monitoring report:

```bash
uv run python src/monitoring/report.py
```

Run retraining flows:

```bash
uv run python src/pipeline/retrain_pipeline.py --strategy schedule
uv run python src/pipeline/retrain_pipeline.py --strategy trigger
uv run python src/pipeline/retrain_pipeline.py --rollback
```

Note: macOS/Linux Makefile shortcuts are available: `make monitoring-report`, `make retrain-weekly`, `make retrain-trigger`, `make retrain-rollback`.

Expected outputs:

- `reports/monitoring/monitoring_report.md`
- `reports/retraining/retrain_report.md`

## 6) Docker (Optional)

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

## 7) Kubernetes Deployment (Minimal)

This project includes minimal Kubernetes manifests for deploying the recommender API service.

Files:

- `k8s/deployment.yaml`
- `k8s/service.yaml`
- `k8s/configmap.yaml`
- `k8s/secret.example.yaml`

Prerequisites:

- Docker
- `kubectl`
- A running local cluster (`minikube`, `kind`, or `k3d`)

Build image:

```bash
docker build -t movielens-api:latest .
```

If you use minikube, load image into the cluster runtime:

```bash
minikube image load movielens-api:latest
```

Apply manifests:

```bash
kubectl apply -f k8s/
```

Check status:

```bash
kubectl get pods
kubectl get svc
```

Access API locally via port-forward:

```bash
kubectl port-forward svc/recommender-service 8000:80
```

In another terminal, test endpoints:

```bash
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/recommend/1?top_k=10"
```

Cleanup:

```bash
kubectl delete -f k8s/
```

## 8) Streamlit UI (Model Comparison)

You can run a lightweight UI to compare BPR and Content-based recommendations.

Prerequisites:

- Model bundles exist at `models/personalized/bpr_model.pkl` and `models/personalized/content_based_model.pkl`

Run UI:

```bash
uv run streamlit run src/serving/ui_app.py
```

Note: macOS/Linux shortcut: `make ui-run`.

Then open the local Streamlit URL shown in terminal (usually `http://localhost:8501`).

## 9) CI/CD Overview

- CI: `.github/workflows/ci.yml` runs dependency install, smoke compile checks, and unit tests.
- CD: `.github/workflows/cd.yml` builds and publishes API Docker image to GHCR on push to `main` (or manually via workflow dispatch).
- Retraining automation: `.github/workflows/retrain.yml` for retrain-related workflow steps.

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

## Success Criteria (Fresh Clone Test)

Quick smoke test (recommended for first run):

1. `uv sync`
2. DVC access configured (if using Path A)
3. `uv run dvc pull` (or use Path B if remote is incomplete)
4. `uv run pytest -q`
5. `uv run uvicorn src.serving.app:app --host 0.0.0.0 --port 8000` + successful `/health` and `/recommend`

Full workflow validation:

1. `uv sync`
2. `uv run dvc pull`
3. `uv run pytest -q`
4. `uv run uvicorn src.serving.app:app --host 0.0.0.0 --port 8000` + successful `/health` and `/recommend`
5. `uv run python src/monitoring/report.py`
6. `uv run python src/pipeline/retrain_pipeline.py --strategy trigger`

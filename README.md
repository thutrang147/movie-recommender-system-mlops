# Movie Recommender System (MLOps)

An MLOps-oriented MovieLens recommender system with data versioning, model training, API serving, monitoring, and retraining workflows.

## What This Repo Does

- Builds a recommendation pipeline from raw MovieLens data.
- Trains and evaluates multiple models: popularity baseline, SVD, NumPy BPR, and content-based TF-IDF.
- Serves recommendations through FastAPI with registry-based model loading.
- Includes request monitoring, drift checks, and a retraining pipeline with promotion and rollback logic.

## Main Artifacts

- Data: `data/interim/`, `data/processed/`, `data/split/`
- Models: `models/baseline/`, `models/personalized/`
- Reports: `reports/`

## Key Links

- Serving app: [src/serving/app.py](src/serving/app.py)
- Monitoring: [src/monitoring/report.py](src/monitoring/report.py)
- Continuous training: [src/pipeline/retrain_pipeline.py](src/pipeline/retrain_pipeline.py)

## Quick Start

Install dependencies, pull shared artifacts, and run tests:
```bash
uv sync
uv run dvc pull
uv run pytest -q
```

## Full Local Rebuild

If you need to rebuild from raw data end-to-end:

```bash
uv sync
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

## End-to-End Flow

```text
data/raw/*.dat
  -> load_data.py
  -> validate_data.py
  -> preprocess.py
  -> split.py
  -> baseline.py / train.py / train_bpr.py / train_content_based.py
  -> evaluate.py / final_benchmark.py / log_mlflow_benchmark.py
  -> dvc add / dvc push
```

## DVC Workflow

Use DVC to sync shared data/model artifacts.

Pull tracked artifacts from remote:

```bash
uv run dvc pull
```

After rebuilding artifacts locally, update DVC metadata and upload:

```bash
uv run dvc repro
git add dvc.yaml dvc.lock
git commit -m "chore: update dvc pipeline and artifacts"
uv run dvc push
```

Check configured remotes:

```bash
uv run dvc remote list
uv run dvc config --list
```

## Common Commands

Train and evaluate:

```bash
uv run python src/models/train.py
uv run python src/models/train_bpr.py
uv run python src/models/train_content_based.py
uv run python src/models/evaluate.py
uv run python src/models/final_benchmark.py
```

Run serving locally:

```bash
make api-run
curl http://127.0.0.1:8000/health
curl "http://127.0.0.1:8000/recommend/1?top_k=6"
```

Run batch inference:

```bash
make api-batch
```

Run retraining:

```bash
make retrain-weekly
make retrain-trigger
make retrain-rollback
```

## Evaluation Summary

The frozen benchmark compares Baseline, SVD, BPR, and Content-based models using the same test split and metrics protocol.

- Final benchmark report: `reports/evaluation/final_comparison.md`
- Final benchmark JSON: `reports/evaluation/final_comparison.json`

## Serving

The API uses a lightweight registry at `models/registry.json`.

- `GET /health`
- `GET /recommend/{user_id}`
- Popularity fallback is used for unknown users.

## Monitoring

```bash
make monitoring-report
```

Artifacts:

- `reports/monitoring/request_logs.jsonl`
- `reports/monitoring/monitoring_report.md`
- `reports/monitoring/monitoring_report.json`

Thresholds:

- `configs/monitoring.yaml`

## Retraining Pipeline

Continuous training is implemented as a retraining pipeline with promotion and rollback.

### Retraining modes

- Schedule-based retraining: runs on a weekly cadence.
- Trigger-based retraining: runs when drift exceeds threshold.

### Pipeline flow

1. Load new feedback or monitoring context.
2. Validate schema.
3. Update the training set.
4. Train a candidate model.
5. Evaluate candidate vs active model on a frozen evaluation split.
6. Apply the promotion rule.
7. Promote the candidate or keep the current model.

### Promotion rule

- Promote if `Recall@10` improves.
- If `Recall@10` ties, promote only if `Coverage` improves.

### Rollback strategy

- The current active model is snapshotted before promotion.
- If a promoted model fails or degrades, rollback restores the previous active checkpoint.

### Automation

```bash
make retrain-weekly
make retrain-trigger
make retrain-rollback
./scripts/run_weekly_retrain.sh
```

- Workflow: [.github/workflows/retrain.yml](.github/workflows/retrain.yml)
- Pipeline: [src/pipeline/retrain_pipeline.py](src/pipeline/retrain_pipeline.py)
- Config: [configs/retraining.yaml](configs/retraining.yaml)
- Reports: `reports/retraining/retrain_report.md` and `reports/retraining/retrain_report.json`

## Reproducibility

- Config-first workflow in `configs/`
- DVC-managed data artifacts
- MLflow experiment logging
- CI smoke compile + pytest checks
- Generated reports in `reports/`

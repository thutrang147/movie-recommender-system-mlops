"""Log final benchmark rows to MLflow for reproducible experiment tracking."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import mlflow
import yaml


project_root = Path(__file__).resolve().parents[2]


def load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dictionary: {path}")
    return data


def load_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing benchmark json: {path}")
    with open(path, "r", encoding="utf-8") as file:
        rows = json.load(file)
    if not isinstance(rows, list):
        raise ValueError("Benchmark json must be a list of row objects.")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Log final benchmark rows to MLflow.")
    parser.add_argument("--eval-config", type=str, default="configs/evaluation.yaml", help="Path to evaluation config YAML.")
    parser.add_argument("--mlflow-config", type=str, default="configs/mlflow.yaml", help="Path to MLflow config YAML.")
    args = parser.parse_args()

    eval_cfg = load_yaml(project_root / args.eval_config)
    mlflow_cfg = load_yaml(project_root / args.mlflow_config)

    benchmark_json_path = project_root / str(eval_cfg["outputs"]["final_comparison_json"])  # type: ignore[index]
    rows = load_rows(benchmark_json_path)

    tracking_uri = str(mlflow_cfg.get("tracking_uri", "file:./mlruns"))
    experiment_name = str(mlflow_cfg.get("experiment_name", "movie-recommender-final-benchmark"))
    run_name_prefix = str(mlflow_cfg.get("run_name_prefix", "week11"))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    for row in rows:
        model_name = str(row["Model"])
        run_name = f"{run_name_prefix}-{model_name.lower().replace(' ', '-')}-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model", model_name)
            mlflow.log_metric("users_evaluated", float(row["users_evaluated"]))
            mlflow.log_metric("recall_at_10", float(row["Recall@10"]))
            mlflow.log_metric("map_at_10", float(row["MAP@10"]))
            mlflow.log_metric("hit_rate_at_10", float(row["HitRate@10"]))
            mlflow.log_metric("coverage", float(row["Coverage"]))

    print(f"Logged {len(rows)} benchmark runs to MLflow experiment '{experiment_name}'.")


if __name__ == "__main__":
    main()

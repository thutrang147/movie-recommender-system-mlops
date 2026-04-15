"""Train a BPR personalized recommender on train/val splits."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Dict

import pandas as pd  # type: ignore[import-not-found]

# MLflow integration
import mlflow
import mlflow.sklearn
import yaml

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))
def load_yaml(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dictionary: {path}")
    return data

from src.models.bpr import BPRConfig, BPRModel


def resolve_paths(
    train_path: str | None,
    val_path: str | None,
    model_path: str | None,
    report_dir: str | None,
) -> tuple[Path, Path, Path, Path]:
    """Resolve project-relative defaults for split input and model outputs."""
    project_root = Path(__file__).resolve().parents[2]
    resolved_train_path = Path(train_path) if train_path else project_root / "data" / "split" / "train.parquet"
    resolved_val_path = Path(val_path) if val_path else project_root / "data" / "split" / "val.parquet"
    resolved_model_path = Path(model_path) if model_path else project_root / "models" / "personalized" / "bpr_model.pkl"
    resolved_report_dir = Path(report_dir) if report_dir else project_root / "reports" / "personalized"
    return resolved_train_path, resolved_val_path, resolved_model_path, resolved_report_dir


def load_split(path: Path) -> pd.DataFrame:
    """Load one split parquet file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_parquet(path)


def save_bundle(bundle: Dict[str, object], model_path: Path) -> None:
    """Persist the BPR bundle as a pickle artifact."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as file:
        pickle.dump(bundle, file)
    print(f"Saved BPR model bundle to: {model_path}")


def save_summary(summary: Dict[str, object], model_path: Path) -> Path:
    """Persist JSON summary alongside the model artifact."""
    summary_path = model_path.with_suffix(".json")
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
    print(f"Saved BPR training summary to: {summary_path}")
    return summary_path


def save_report(summary: Dict[str, object], report_dir: Path) -> Path:
    """Write a markdown report for the BPR run."""
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "bpr_train_report.md"

    lines = [
        "# BPR Personalized Model Training Report",
        "",
        "## Config",
        f"- factors: {summary['config']['factors']}",
        f"- learning_rate: {summary['config']['learning_rate']}",
        f"- reg: {summary['config']['reg']}",
        f"- epochs: {summary['config']['epochs']}",
        f"- n_samples_per_epoch: {summary['config']['n_samples_per_epoch']}",
        f"- top_k: {summary['config']['top_k']}",
        f"- relevance_threshold: {summary['config']['relevance_threshold']}",
        f"- patience: {summary['config']['patience']}",
        "",
        "## Validation Metrics",
        f"- users_evaluated: {summary['validation_metrics']['users_evaluated']}",
        f"- recall_at_k: {summary['validation_metrics']['recall_at_k']:.4f}",
        f"- map_at_k: {summary['validation_metrics']['map_at_k']:.4f}",
        f"- hit_rate_at_k: {summary['validation_metrics']['hit_rate_at_k']:.4f}",
        f"- coverage: {summary['validation_metrics']['coverage']:.4f}",
        "",
        "## Early Stopping",
        f"- best_epoch: {summary['best_epoch']}",
        f"- best_val_map: {summary['best_val_map']:.4f}",
        "",
        "## Notes",
        str(summary["notes"]),
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved BPR training report to: {report_path}")
    return report_path


def run_training(
    train_path: Path,
    val_path: Path,
    model_path: Path,
    report_dir: Path,
    config: BPRConfig,
) -> Dict[str, object]:
    """Fit the BPR model and persist artifacts."""
    train_df = load_split(train_path)
    val_df = load_split(val_path)

    # Load mlflow config
    mlflow_cfg_path = project_root / "configs" / "mlflow.yaml"
    mlflow_cfg = load_yaml(mlflow_cfg_path)
    tracking_uri = str(mlflow_cfg.get("tracking_uri", "file:./mlruns"))
    experiment_name = str(mlflow_cfg.get("experiment_name", "movie-recommender-bpr"))
    run_name_prefix = str(mlflow_cfg.get("run_name_prefix", "bpr"))

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    run_name = f"{run_name_prefix}-train-{config.factors}f-{config.learning_rate}lr"
    with mlflow.start_run(run_name=run_name):
        # Log hyperparameters
        mlflow.log_params({
            "factors": config.factors,
            "learning_rate": config.learning_rate,
            "reg": config.reg,
            "epochs": config.epochs,
            "n_samples_per_epoch": config.n_samples_per_epoch,
            "top_k": config.top_k,
            "relevance_threshold": config.relevance_threshold,
            "patience": config.patience,
            "random_state": config.random_state,
        })

        model = BPRModel(config=config).fit(train_df=train_df, val_df=val_df)
        validation_metrics = model.ranking_metrics(val_df, top_k=config.top_k)

        # Log metrics
        mlflow.log_metrics({
            "users_evaluated": validation_metrics['users_evaluated'],
            "recall_at_k": validation_metrics['recall_at_k'],
            "map_at_k": validation_metrics['map_at_k'],
            "hit_rate_at_k": validation_metrics['hit_rate_at_k'],
            "coverage": validation_metrics['coverage']
        })

        bundle = model.to_bundle()
        save_bundle(bundle=bundle, model_path=model_path)

        # Log model artifact & register
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="bpr_model",
            registered_model_name="BPR_Recommender"
        )

        summary: Dict[str, object] = {
            "algorithm": "bpr_mf_numpy",
            "config": bundle["config"],
            "validation_metrics": validation_metrics,
            "best_epoch": bundle["best_epoch"],
            "best_val_map": bundle["best_val_map"],
            "artifact_path": str(model_path),
            "notes": "BPR is trained with sampled pairwise triplets and early stopping on validation MAP@K.",
        }

        save_summary(summary=summary, model_path=model_path)
        save_report(summary=summary, report_dir=report_dir)
        return summary


def main() -> Dict[str, object]:
    """CLI entrypoint for training the BPR personalized model."""
    parser = argparse.ArgumentParser(description="Train a BPR personalized recommender on split data.")
    parser.add_argument("--train-path", type=str, default=None, help="Path to train parquet (default: data/split/train.parquet).")
    parser.add_argument("--val-path", type=str, default=None, help="Path to validation parquet (default: data/split/val.parquet).")
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to save the pickled model bundle (default: models/personalized/bpr_model.pkl).",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=None,
        help="Directory for training report (default: reports/personalized).",
    )
    parser.add_argument("--factors", type=int, default=64, help="Number of latent factors.")
    parser.add_argument("--learning-rate", type=float, default=0.03, help="Learning rate for SGD updates.")
    parser.add_argument("--reg", type=float, default=0.001, help="L2 regularization strength.")
    parser.add_argument("--epochs", type=int, default=30, help="Maximum number of training epochs.")
    parser.add_argument(
        "--n-samples-per-epoch",
        type=int,
        default=200000,
        help="How many positive pairs to sample per epoch for SGD.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K cutoff for ranking metrics.")
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=4.0,
        help="Minimum rating to treat an item as relevant.",
    )
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience on validation MAP@K.")

    args = parser.parse_args()
    train_path, val_path, model_path, report_dir = resolve_paths(args.train_path, args.val_path, args.model_path, args.report_dir)

    config = BPRConfig(
        factors=args.factors,
        learning_rate=args.learning_rate,
        reg=args.reg,
        epochs=args.epochs,
        n_samples_per_epoch=args.n_samples_per_epoch,
        random_state=args.random_state,
        top_k=args.top_k,
        relevance_threshold=args.relevance_threshold,
        patience=args.patience,
    )

    print("Starting BPR training pipeline...")
    print(f"- train_path: {train_path}")
    print(f"- val_path: {val_path}")
    print(f"- model_path: {model_path}")
    print(f"- config: {config}")

    return run_training(
        train_path=train_path,
        val_path=val_path,
        model_path=model_path,
        report_dir=report_dir,
        config=config,
    )


if __name__ == "__main__":
    main()

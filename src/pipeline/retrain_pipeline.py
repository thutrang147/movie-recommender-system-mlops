"""Week 14 continuous-training pipeline with retrain, promotion, and rollback."""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd  # type: ignore[import-not-found]
import yaml
from src.models.evaluate import evaluate_personalized_model
from src.models.train_bpr import BPRConfig, run_training

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))


@dataclass
class PromotionDecision:
    promote: bool
    reason: str


def summarize_registry_model(model_cfg: Dict[str, object] | None) -> Dict[str, str]:
    model_cfg = model_cfg if isinstance(model_cfg, dict) else {}
    return {
        "name": str(model_cfg.get("name", "unknown")),
        "version": str(model_cfg.get("version", "unknown")),
        "artifact_path": str(model_cfg.get("artifact_path", "N/A")),
    }


def load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dict: {path}")
    return data


def load_registry(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing model registry: {path}")
    with open(path, "r", encoding="utf-8") as file:
        registry = json.load(file)
    if not isinstance(registry, dict):
        raise ValueError("Registry must be a JSON object.")
    return registry


def save_registry(path: Path, registry: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(registry, file, indent=2, ensure_ascii=False)
        file.write("\n")


def load_bundle(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    with open(path, "rb") as file:
        bundle = pickle.load(file)
    if not isinstance(bundle, dict):
        raise ValueError(f"Model artifact is not a dict bundle: {path}")
    return bundle


def validate_split(df: pd.DataFrame, name: str) -> None:
    required = {"user_id", "movie_id", "rating"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{name} split missing columns: {sorted(missing)}")
    if df.empty:
        raise ValueError(f"{name} split is empty.")


def load_parquet(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing {name} parquet: {path}")
    df = pd.read_parquet(path)
    validate_split(df, name)
    return df


def normalize_new_feedback(new_df: pd.DataFrame) -> pd.DataFrame:
    required = {"user_id", "movie_id", "rating"}
    missing = required - set(new_df.columns)
    if missing:
        raise ValueError(f"New feedback file missing required columns: {sorted(missing)}")

    normalized = new_df.loc[:, ["user_id", "movie_id", "rating"]].copy()
    normalized["user_id"] = pd.to_numeric(normalized["user_id"], errors="coerce")
    normalized["movie_id"] = pd.to_numeric(normalized["movie_id"], errors="coerce")
    normalized["rating"] = pd.to_numeric(normalized["rating"], errors="coerce")

    if "timestamp" in new_df.columns:
        normalized["timestamp"] = pd.to_numeric(new_df["timestamp"], errors="coerce")
    else:
        normalized["timestamp"] = int(datetime.now(tz=timezone.utc).timestamp())

    normalized = normalized.dropna(subset=["user_id", "movie_id", "rating", "timestamp"])
    normalized["user_id"] = normalized["user_id"].astype(int)
    normalized["movie_id"] = normalized["movie_id"].astype(int)
    normalized["rating"] = normalized["rating"].astype(float)
    normalized["timestamp"] = normalized["timestamp"].astype(int)

    if normalized.empty:
        raise ValueError("New feedback contains no valid records after validation.")
    return normalized


def load_new_feedback(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing new feedback file: {path}")

    if path.suffix.lower() == ".parquet":
        new_df = pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        new_df = pd.read_csv(path)
    else:
        raise ValueError("New feedback file must be .csv or .parquet")

    return normalize_new_feedback(new_df)


def update_training_data(
    base_train_df: pd.DataFrame,
    retrain_train_path: Path,
    new_feedback_path: Path | None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    base = base_train_df.copy()
    if "timestamp" not in base.columns:
        base["timestamp"] = int(datetime.now(tz=timezone.utc).timestamp())
    base["timestamp"] = pd.to_numeric(base["timestamp"], errors="coerce").fillna(0).astype(int)

    if new_feedback_path is None:
        retrain_train_path.parent.mkdir(parents=True, exist_ok=True)
        base.to_parquet(retrain_train_path, index=False)
        return base, {"new_feedback_rows": 0, "updated_train_rows": int(len(base))}

    new_feedback_df = load_new_feedback(new_feedback_path)
    combined = pd.concat([base, new_feedback_df], ignore_index=True)
    combined = combined.sort_values(["timestamp", "user_id", "movie_id"], ascending=[True, True, True])
    combined = combined.drop_duplicates(subset=["user_id", "movie_id"], keep="last")

    retrain_train_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(retrain_train_path, index=False)
    return combined, {"new_feedback_rows": int(len(new_feedback_df)), "updated_train_rows": int(len(combined))}


def decide_promotion(current_metrics: Dict[str, float | int], candidate_metrics: Dict[str, float | int], epsilon: float = 1e-9) -> PromotionDecision:
    current_recall = float(current_metrics.get("recall_at_k", 0.0))
    candidate_recall = float(candidate_metrics.get("recall_at_k", 0.0))
    current_coverage = float(current_metrics.get("coverage", 0.0))
    candidate_coverage = float(candidate_metrics.get("coverage", 0.0))

    if candidate_recall > current_recall + epsilon:
        return PromotionDecision(
            promote=True,
            reason=(
                f"Promote: recall_at_k improved from {current_recall:.6f} "
                f"to {candidate_recall:.6f}."
            ),
        )

    recall_gap = abs(candidate_recall - current_recall)
    if recall_gap <= epsilon and candidate_coverage > current_coverage + epsilon:
        return PromotionDecision(
            promote=True,
            reason=(
                "Promote: recall_at_k is equivalent and coverage improved "
                f"from {current_coverage:.6f} to {candidate_coverage:.6f}."
            ),
        )

    return PromotionDecision(
        promote=False,
        reason=(
            "Keep current model: candidate did not satisfy promotion rule "
            f"(candidate recall={candidate_recall:.6f}, current recall={current_recall:.6f}, "
            f"candidate coverage={candidate_coverage:.6f}, current coverage={current_coverage:.6f})."
        ),
    )


def should_retrain_triggered(strategy: str, drift_score: float | None, drift_threshold: float) -> Tuple[bool, str]:
    if strategy == "schedule":
        return True, "Schedule-based retraining selected: run every cycle."

    if drift_score is None:
        return False, "Trigger-based retraining skipped: missing drift score."

    if drift_score >= drift_threshold:
        return True, f"Trigger-based retraining activated: drift_score={drift_score:.4f} >= {drift_threshold:.4f}."

    return False, f"Trigger-based retraining skipped: drift_score={drift_score:.4f} < {drift_threshold:.4f}."


def to_relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path)


def infer_drift_score(trigger_report_path: Path) -> float | None:
    if not trigger_report_path.exists():
        return None
    with open(trigger_report_path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    try:
        return float(payload.get("drift", {}).get("drift_score"))
    except Exception:
        return None


def build_metric_deltas(current_metrics: Dict[str, float | int], candidate_metrics: Dict[str, float | int]) -> Dict[str, float]:
    current_recall = float(current_metrics.get("recall_at_k", 0.0))
    candidate_recall = float(candidate_metrics.get("recall_at_k", 0.0))
    current_coverage = float(current_metrics.get("coverage", 0.0))
    candidate_coverage = float(candidate_metrics.get("coverage", 0.0))
    return {
        "recall_at_k_delta": candidate_recall - current_recall,
        "coverage_delta": candidate_coverage - current_coverage,
    }


def write_release_manifest(summary: Dict[str, object], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "released_at": datetime.now(tz=timezone.utc).isoformat(),
        "strategy": summary.get("strategy"),
        "status": summary.get("status"),
        "decision": summary.get("decision"),
        "dry_run": bool(summary.get("dry_run", False)),
        "current_model": summary.get("current_model", {}),
        "candidate_model": summary.get("candidate_model", {}),
        "current_metrics": summary.get("current_metrics", {}),
        "candidate_metrics": summary.get("candidate_metrics", {}),
        "metric_deltas": summary.get("metric_deltas", {}),
        "registry_path": summary.get("registry_path"),
        "candidate_artifact": summary.get("candidate_artifact"),
        "rollback_alias": summary.get("rollback_alias"),
    }
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, ensure_ascii=False)
        file.write("\n")


def write_report(summary: Dict[str, object], output_md: Path, output_json: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)
        file.write("\n")

    current_model = summary.get("current_model", {}) if isinstance(summary.get("current_model"), dict) else {}
    candidate_model = summary.get("candidate_model", {}) if isinstance(summary.get("candidate_model"), dict) else {}
    metric_deltas = summary.get("metric_deltas", {}) if isinstance(summary.get("metric_deltas"), dict) else {}
    update_stats = summary.get("update_stats", {}) if isinstance(summary.get("update_stats"), dict) else {}

    lines = [
        "# Week 14 Retraining Report",
        "",
        f"- strategy: {summary['strategy']}",
        f"- status: {summary['status']}",
        f"- dry_run: {bool(summary.get('dry_run', False))}",
        f"- trigger_reason: {summary['trigger_reason']}",
        f"- decision: {summary['decision']}",
        f"- decision_reason: {summary['decision_reason']}",
        f"- drift_score: {float(summary.get('drift_score', 0.0) or 0.0):.4f}",
        "",
        "## Model Selection",
        f"- current_model: {current_model.get('name', 'unknown')} ({current_model.get('version', 'unknown')})",
        f"- current_artifact: {current_model.get('artifact_path', 'N/A')}",
        f"- candidate_model: {candidate_model.get('name', 'candidate')} ({candidate_model.get('version', 'candidate')})",
        f"- candidate_artifact: {candidate_model.get('artifact_path', summary['candidate_artifact'])}",
        "",
        "## Metrics",
        f"- current_recall_at_k: {float(summary['current_metrics']['recall_at_k']):.6f}",
        f"- candidate_recall_at_k: {float(summary['candidate_metrics']['recall_at_k']):.6f}",
        f"- current_coverage: {float(summary['current_metrics']['coverage']):.6f}",
        f"- candidate_coverage: {float(summary['candidate_metrics']['coverage']):.6f}",
        f"- recall_at_k_delta: {float(metric_deltas.get('recall_at_k_delta', 0.0)):.6f}",
        f"- coverage_delta: {float(metric_deltas.get('coverage_delta', 0.0)):.6f}",
        "",
        "## Training Data Update",
        f"- new_feedback_rows: {int(update_stats.get('new_feedback_rows', 0))}",
        f"- updated_train_rows: {int(update_stats.get('updated_train_rows', 0))}",
        "",
        "## Artifacts",
        f"- registry_path: {summary['registry_path']}",
        f"- candidate_artifact: {summary['candidate_artifact']}",
        f"- rollback_alias: {summary['rollback_alias']}",
        "",
        "## Promotion Rule",
        "- Promote when candidate recall_at_k is higher than current recall_at_k.",
        "- If recall_at_k is tied, use coverage as the tie-breaker.",
        "",
        "## Rollback Strategy",
        "- If a promoted model is unstable in production or degrades KPI, run `make retrain-rollback`.",
        "- The rollback command restores active model to a real artifact path (never to alias path).",
    ]

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def rollback_to_previous(registry_path: Path, rollback_alias_path: Path, reason: str) -> Dict[str, object]:
    registry = load_registry(registry_path)
    previous_active = registry.get("active_model", {})
    if not isinstance(previous_active, dict):
        previous_active = {}

    metadata = registry.get("metadata", {}) if isinstance(registry.get("metadata"), dict) else {}
    previous_model = metadata.get("previous_model", {}) if isinstance(metadata.get("previous_model"), dict) else {}

    rollback_target = previous_model if previous_model else {}
    rollback_target_path = project_root / str(rollback_target.get("artifact_path", ""))

    if not rollback_target or not rollback_target_path.exists():
        if not rollback_alias_path.exists():
            raise FileNotFoundError(
                "Rollback failed: no valid previous model artifact and rollback alias is missing "
                f"({rollback_alias_path})."
            )

        timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        restored_path = rollback_alias_path.parent / f"rollback_restored_{timestamp}.pkl"
        shutil.copy2(rollback_alias_path, restored_path)
        rollback_target = {
            "name": str(previous_model.get("name", "bpr")),
            "version": f"rollback-{timestamp}",
            "artifact_path": to_relative(restored_path),
        }
        rollback_target_path = restored_path

    # Keep active_model pointed at a concrete artifact path, not at rollback alias.
    registry["active_model"] = {
        "name": str(rollback_target.get("name", "bpr")),
        "version": str(rollback_target.get("version", "unknown")),
        "artifact_path": to_relative(rollback_target_path),
    }

    metadata["rollback_at"] = datetime.now(tz=timezone.utc).date().isoformat()
    metadata["rollback_reason"] = reason
    metadata["rolled_back_from"] = previous_active
    metadata["rollback_alias"] = to_relative(rollback_alias_path)
    registry["metadata"] = metadata

    save_registry(registry_path, registry)
    return registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 14 retraining pipeline.")
    parser.add_argument("--strategy", choices=["schedule", "trigger"], default=None)
    parser.add_argument("--new-feedback-path", type=str, default=None)
    parser.add_argument("--drift-score", type=float, default=None)
    parser.add_argument("--rollback", action="store_true", help="Rollback active model to previous stable artifact.")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate and report without changing registry state.")
    parser.add_argument("--current-artifact-override", type=str, default=None, help="Optional artifact path to evaluate as the current model.")
    parser.add_argument("--candidate-artifact-override", type=str, default=None, help="Optional artifact path to evaluate as the candidate model.")
    parser.add_argument("--retrain-config", type=str, default="configs/retraining.yaml")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml")
    parser.add_argument("--eval-config", type=str, default="configs/evaluation.yaml")
    parser.add_argument("--force", action="store_true", help="Force run even if trigger condition is not met.")
    parser.add_argument("--epochs-override", type=int, default=None, help="Optional override for BPR epochs.")
    parser.add_argument(
        "--samples-per-epoch-override",
        type=int,
        default=None,
        help="Optional override for BPR n_samples_per_epoch.",
    )
    args = parser.parse_args()

    retrain_cfg = load_yaml(project_root / args.retrain_config)
    data_cfg = load_yaml(project_root / args.data_config)
    model_cfg = load_yaml(project_root / args.model_config)
    eval_cfg = load_yaml(project_root / args.eval_config)

    strategy_cfg = retrain_cfg.get("strategy", {}) if isinstance(retrain_cfg.get("strategy"), dict) else {}
    trigger_cfg = retrain_cfg.get("trigger", {}) if isinstance(retrain_cfg.get("trigger"), dict) else {}
    paths_cfg = retrain_cfg.get("paths", {}) if isinstance(retrain_cfg.get("paths"), dict) else {}

    strategy = args.strategy or str(strategy_cfg.get("default", "schedule"))
    drift_threshold = float(trigger_cfg.get("drift_threshold", 0.2))

    registry_path = project_root / str(paths_cfg.get("registry_path", "models/registry.json"))
    rollback_alias = project_root / str(paths_cfg.get("rollback_alias_path", "models/personalized/active_model_prev.pkl"))
    registry_snapshot = load_registry(registry_path)
    current_model_summary = summarize_registry_model(registry_snapshot.get("active_model", {}))

    output_md = project_root / str(paths_cfg.get("report_markdown", "reports/retraining/retrain_report.md"))
    output_json = project_root / str(paths_cfg.get("report_json", "reports/retraining/retrain_report.json"))
    release_manifest_json = project_root / str(paths_cfg.get("release_manifest_json", "reports/retraining/release_manifest.json"))

    if args.rollback:
        rollback_to_previous(
            registry_path=registry_path,
            rollback_alias_path=rollback_alias,
            reason="manual rollback after degraded or failed promoted model",
        )
        summary = {
            "strategy": strategy,
            "status": "rollback_completed",
            "dry_run": bool(args.dry_run),
            "trigger_reason": "manual rollback flag provided",
            "decision": "rollback",
            "decision_reason": "Active model restored to previous stable artifact path.",
            "drift_score": args.drift_score,
            "current_model": current_model_summary,
            "candidate_model": summarize_registry_model(None),
            "current_metrics": {"recall_at_k": 0.0, "coverage": 0.0},
            "candidate_metrics": {"recall_at_k": 0.0, "coverage": 0.0},
            "metric_deltas": {"recall_at_k_delta": 0.0, "coverage_delta": 0.0},
            "update_stats": {"new_feedback_rows": 0, "updated_train_rows": 0},
            "registry_path": to_relative(registry_path),
            "candidate_artifact": "N/A",
            "rollback_alias": to_relative(rollback_alias),
        }
        write_report(summary=summary, output_md=output_md, output_json=output_json)
        write_release_manifest(summary=summary, output_json=release_manifest_json)
        print(f"Rollback completed. Active model points to: {rollback_alias}")
        return

    trigger_report_path = project_root / str(trigger_cfg.get("monitoring_report_json", "reports/monitoring/monitoring_report.json"))
    drift_score = args.drift_score if args.drift_score is not None else infer_drift_score(trigger_report_path)
    should_run, trigger_reason = should_retrain_triggered(strategy=strategy, drift_score=drift_score, drift_threshold=drift_threshold)

    if not should_run and not args.force:
        summary = {
            "strategy": strategy,
            "status": "skipped",
            "dry_run": bool(args.dry_run),
            "trigger_reason": trigger_reason,
            "decision": "no_retrain",
            "decision_reason": "Retraining was not required by trigger strategy.",
            "drift_score": drift_score,
            "current_model": current_model_summary,
            "candidate_model": summarize_registry_model(None),
            "current_metrics": {"recall_at_k": 0.0, "coverage": 0.0},
            "candidate_metrics": {"recall_at_k": 0.0, "coverage": 0.0},
            "metric_deltas": {"recall_at_k_delta": 0.0, "coverage_delta": 0.0},
            "update_stats": {"new_feedback_rows": 0, "updated_train_rows": 0},
            "registry_path": to_relative(registry_path),
            "candidate_artifact": "N/A",
            "rollback_alias": to_relative(rollback_alias),
        }
        write_report(summary=summary, output_md=output_md, output_json=output_json)
        write_release_manifest(summary=summary, output_json=release_manifest_json)
        print(trigger_reason)
        print("Retraining skipped.")
        return

    split_cfg = data_cfg.get("paths", {}) if isinstance(data_cfg.get("paths"), dict) else {}
    train_path = project_root / str(split_cfg.get("train_split", "data/split/train.parquet"))
    val_path = project_root / str(split_cfg.get("val_split", "data/split/val.parquet"))
    test_path = project_root / str(split_cfg.get("test_split", "data/split/test.parquet"))

    retrain_train_path = project_root / str(paths_cfg.get("retrain_train_split", "data/split/train_retrain_latest.parquet"))
    candidate_dir = project_root / str(paths_cfg.get("candidate_dir", "models/personalized/candidates"))
    archive_dir = project_root / str(paths_cfg.get("archive_dir", "models/personalized/archive"))

    timestamp = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate_model_path = candidate_dir / f"bpr_model_{timestamp}.pkl"
    candidate_report_dir = project_root / "reports" / "retraining" / "candidate"

    base_train_df = load_parquet(train_path, "train")
    _ = load_parquet(val_path, "val")
    test_df = load_parquet(test_path, "test")

    new_feedback_path = Path(args.new_feedback_path) if args.new_feedback_path else None
    if new_feedback_path and not new_feedback_path.is_absolute():
        new_feedback_path = project_root / new_feedback_path

    _, update_stats = update_training_data(
        base_train_df=base_train_df,
        retrain_train_path=retrain_train_path,
        new_feedback_path=new_feedback_path,
    )

    bpr_cfg = model_cfg.get("bpr", {}) if isinstance(model_cfg.get("bpr"), dict) else {}
    eval_settings = eval_cfg.get("evaluation", {}) if isinstance(eval_cfg.get("evaluation"), dict) else {}

    config = BPRConfig(
        factors=int(bpr_cfg.get("factors", 64)),
        learning_rate=float(bpr_cfg.get("learning_rate", 0.03)),
        reg=float(bpr_cfg.get("reg", 0.001)),
        epochs=int(args.epochs_override if args.epochs_override is not None else bpr_cfg.get("epochs", 30)),
        n_samples_per_epoch=int(
            args.samples_per_epoch_override
            if args.samples_per_epoch_override is not None
            else bpr_cfg.get("n_samples_per_epoch", 200000)
        ),
        random_state=int(bpr_cfg.get("random_state", 42)),
        top_k=int(eval_settings.get("top_k", 10)),
        relevance_threshold=float(eval_settings.get("relevance_threshold", 4.0)),
        patience=int(bpr_cfg.get("patience", 5)),
    )

    registry = load_registry(registry_path)
    current_model_cfg = registry.get("active_model", {}) if isinstance(registry.get("active_model"), dict) else {}
    current_artifact = project_root / str(current_model_cfg.get("artifact_path", "models/personalized/bpr_model.pkl"))
    if args.current_artifact_override:
        current_artifact = Path(args.current_artifact_override)
        if not current_artifact.is_absolute():
            current_artifact = project_root / current_artifact
        current_model_cfg = {
            "name": f"{Path(current_artifact).stem}",
            "version": "override-current",
            "artifact_path": to_relative(current_artifact),
        }

    if args.candidate_artifact_override:
        candidate_model_path = Path(args.candidate_artifact_override)
        if not candidate_model_path.is_absolute():
            candidate_model_path = project_root / candidate_model_path
    else:
        run_training(
            train_path=retrain_train_path,
            val_path=val_path,
            model_path=candidate_model_path,
            report_dir=candidate_report_dir,
            config=config,
        )

    # Always refresh rollback alias from the current active model for safer recovery.
    rollback_alias.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(current_artifact, rollback_alias)

    current_bundle = load_bundle(current_artifact)
    candidate_bundle = load_bundle(candidate_model_path)

    current_metrics = evaluate_personalized_model(
        bundle=current_bundle,
        train_df=base_train_df,
        test_df=test_df,
        top_k=config.top_k,
        relevance_threshold=config.relevance_threshold,
    )
    candidate_metrics = evaluate_personalized_model(
        bundle=candidate_bundle,
        train_df=base_train_df,
        test_df=test_df,
        top_k=config.top_k,
        relevance_threshold=config.relevance_threshold,
    )

    decision = decide_promotion(current_metrics=current_metrics, candidate_metrics=candidate_metrics)
    metric_deltas = build_metric_deltas(current_metrics=current_metrics, candidate_metrics=candidate_metrics)
    status = "evaluated_keep_current"
    final_decision = "keep_current"

    if decision.promote and not args.dry_run:
        archive_dir.mkdir(parents=True, exist_ok=True)

        backup_name = f"{current_model_cfg.get('name', 'model')}_{current_model_cfg.get('version', 'unknown')}_{timestamp}.pkl"
        archive_path = archive_dir / backup_name
        shutil.copy2(current_artifact, archive_path)

        promoted_artifact_path = current_artifact
        shutil.copy2(candidate_model_path, promoted_artifact_path)

        candidate_summary_json = candidate_model_path.with_suffix(".json")
        promoted_summary_json = promoted_artifact_path.with_suffix(".json")
        if candidate_summary_json.exists():
            shutil.copy2(candidate_summary_json, promoted_summary_json)

        registry["active_model"] = {
            "name": "bpr",
            "version": f"weekly-{timestamp}",
            "artifact_path": to_relative(promoted_artifact_path),
        }
        metadata = registry.get("metadata", {}) if isinstance(registry.get("metadata"), dict) else {}
        metadata["updated_at"] = datetime.now(tz=timezone.utc).date().isoformat()
        metadata["previous_model"] = current_model_cfg
        metadata["rollback_alias"] = to_relative(rollback_alias)
        metadata["promotion_reason"] = decision.reason
        metadata["archive_backup"] = to_relative(archive_path)
        metadata["promoted_candidate_source"] = to_relative(candidate_model_path)
        registry["metadata"] = metadata

        save_registry(registry_path, registry)
        status = "promoted"
        final_decision = "promote"
    elif decision.promote and args.dry_run:
        status = "evaluated_promote_dry_run"
        final_decision = "promote"

    candidate_model_summary = {
        "name": "bpr_candidate" if not args.candidate_artifact_override else Path(candidate_model_path).stem,
        "version": f"candidate-{timestamp}" if not args.candidate_artifact_override else "override-candidate",
        "artifact_path": to_relative(candidate_model_path),
    }

    summary = {
        "strategy": strategy,
        "status": status,
        "dry_run": bool(args.dry_run),
        "trigger_reason": trigger_reason,
        "decision": final_decision,
        "decision_reason": decision.reason,
        "drift_score": drift_score,
        "current_model": summarize_registry_model(current_model_cfg),
        "candidate_model": candidate_model_summary,
        "current_metrics": current_metrics,
        "candidate_metrics": candidate_metrics,
        "metric_deltas": metric_deltas,
        "update_stats": update_stats,
        "registry_path": to_relative(registry_path),
        "candidate_artifact": to_relative(candidate_model_path),
        "rollback_alias": to_relative(rollback_alias),
    }

    write_report(summary=summary, output_md=output_md, output_json=output_json)
    write_release_manifest(summary=summary, output_json=release_manifest_json)

    print(trigger_reason)
    print(decision.reason)
    print(f"Retraining report markdown: {output_md}")
    print(f"Retraining report json: {output_json}")


if __name__ == "__main__":
    main()

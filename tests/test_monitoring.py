"""Unit tests for Week 13 monitoring components."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from src.monitoring.logger import MonitoringLogger
from src.monitoring.report import build_monitoring_summary, load_request_logs


def _write_registry_and_bundle(tmp_path: Path) -> tuple[Path, Path]:
    model_path = tmp_path / "bpr_bundle.pkl"
    registry_path = tmp_path / "registry.json"

    bundle = {
        "algorithm": "bpr_mf_numpy",
        "item_popularity_scores": {10: 5, 11: 4, 12: 3, 13: 2, 14: 1},
        "train_user_seen_items": {1: {10, 11}, 2: {12, 13}},
    }
    with open(model_path, "wb") as file:
        pickle.dump(bundle, file)

    registry = {
        "active_model": {"name": "bpr", "version": "vtest", "artifact_path": str(model_path)},
        "fallback": {"type": "popularity", "artifact_path": ""},
    }
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    return registry_path, model_path


def test_request_logger_and_report_summary(tmp_path: Path) -> None:
    log_path = tmp_path / "request_logs.jsonl"
    logger = MonitoringLogger(log_path=log_path)

    logger.log_request(user_id=1, strategy="bpr", latency_ms=12.3, top_k=5, response_status=200, recommendations=[10, 11, 12])
    logger.log_request(
        user_id=999,
        strategy="popularity_fallback",
        latency_ms=18.9,
        top_k=5,
        response_status=200,
        recommendations=[10, 13, 14],
    )

    logs_df = load_request_logs(log_path)
    assert len(logs_df) == 2
    assert set(logs_df.columns) >= {"timestamp", "user_id", "strategy", "latency_ms", "response_status", "recommendations"}

    registry_path, _ = _write_registry_and_bundle(tmp_path)

    monitoring_cfg = {
        "thresholds": {
            "fallback_rate_warn": 0.40,
            "avg_latency_ms_warn": 1000.0,
            "error_rate_warn": 0.02,
            "drift_score_warn": 0.20,
            "unknown_user_rate_warn": 0.40,
        }
    }
    api_cfg = {"serving": {"registry_path": str(registry_path)}}

    summary = build_monitoring_summary(project_root=Path("/"), logs_df=logs_df, monitoring_cfg=monitoring_cfg, api_cfg=api_cfg)

    assert summary["service_metrics"]["request_count"] == 2
    assert summary["service_metrics"]["error_count"] == 0
    assert summary["service_metrics"]["p95_latency_ms"] >= summary["service_metrics"]["avg_latency_ms"]
    assert summary["data_behavior"]["unique_users"] == 2
    assert summary["data_behavior"]["personalized_count"] == 1
    assert summary["data_behavior"]["fallback_count"] == 1
    assert summary["data_behavior"]["fallback_rate"] == 0.5
    assert summary["data_behavior"]["avg_top_k"] == 5.0
    assert isinstance(summary["drift"]["drift_score"], float)
    assert isinstance(summary["warnings"], list)

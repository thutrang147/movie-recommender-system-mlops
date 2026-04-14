"""Monitoring report generation and rule-based alerting."""
from __future__ import annotations


# --- Ensure project root is in sys.path for direct script execution ---
import sys
from pathlib import Path
import argparse
import json
from typing import Dict, List
import pandas as pd  # type: ignore[import-not-found]
import yaml
from src.monitoring.drift import (
    build_top_item_shift,
    compute_drift_score,
    evaluate_drift_warnings,
    load_production_recommendation_distribution,
    load_train_popularity_distribution,
)

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dictionary: {path}")
    return data


def load_request_logs(log_path: Path) -> pd.DataFrame:
    if not log_path.exists():
        return pd.DataFrame(columns=["timestamp", "user_id", "strategy", "latency_ms", "top_k", "response_status", "recommendations"])

    rows: List[Dict[str, object]] = []
    with open(log_path, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if not rows:
        return pd.DataFrame(columns=["timestamp", "user_id", "strategy", "latency_ms", "top_k", "response_status", "recommendations"])

    df = pd.DataFrame(rows)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    if "latency_ms" in df.columns:
        df["latency_ms"] = pd.to_numeric(df["latency_ms"], errors="coerce")
    if "response_status" in df.columns:
        df["response_status"] = pd.to_numeric(df["response_status"], errors="coerce")
    if "user_id" in df.columns:
        df["user_id"] = pd.to_numeric(df["user_id"], errors="coerce")

    return df


def _safe_float(value: float | int | None, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _safe_int(value: float | int | None, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def build_monitoring_summary(
    project_root: Path,
    logs_df: pd.DataFrame,
    monitoring_cfg: Dict[str, object],
    api_cfg: Dict[str, object],
) -> Dict[str, object]:
    thresholds = monitoring_cfg.get("thresholds", {}) if isinstance(monitoring_cfg.get("thresholds"), dict) else {}

    total_requests = int(len(logs_df))
    success_count = int((logs_df["response_status"] < 400).sum()) if total_requests else 0
    error_count = int((logs_df["response_status"] >= 400).sum()) if total_requests else 0
    avg_latency_ms = _safe_float(logs_df["latency_ms"].mean(), 0.0) if total_requests else 0.0
    p95_latency_ms = _safe_float(logs_df["latency_ms"].quantile(0.95), 0.0) if total_requests else 0.0
    success_rate = (success_count / total_requests) if total_requests else 0.0
    error_rate = (error_count / total_requests) if total_requests else 0.0

    fallback_count = (
        int((logs_df["strategy"] == "popularity_fallback").sum()) if total_requests and "strategy" in logs_df.columns else 0
    )
    personalized_count = max(total_requests - fallback_count, 0)
    fallback_rate = (
        float((logs_df["strategy"] == "popularity_fallback").mean()) if total_requests and "strategy" in logs_df.columns else 0.0
    )
    personalized_rate = (personalized_count / total_requests) if total_requests else 0.0
    unknown_user_rate = fallback_rate

    unique_users = _safe_int(logs_df["user_id"].nunique(), 0) if total_requests and "user_id" in logs_df.columns else 0
    requests_per_user = (
        float(logs_df.groupby("user_id").size().mean()) if total_requests and "user_id" in logs_df.columns else 0.0
    )
    avg_top_k = _safe_float(logs_df["top_k"].mean(), 0.0) if total_requests and "top_k" in logs_df.columns else 0.0

    volume_per_day = {}
    if total_requests and "timestamp" in logs_df.columns:
        day_series = logs_df["timestamp"].dt.strftime("%Y-%m-%d")
        volume_per_day = day_series.value_counts().sort_index().to_dict()

    registry_path = project_root / str(api_cfg.get("serving", {}).get("registry_path", "models/registry.json"))
    train_dist = load_train_popularity_distribution(project_root=project_root, registry_path=registry_path)
    prod_dist = load_production_recommendation_distribution(logs_df)
    drift_score = compute_drift_score(train_dist=train_dist, prod_dist=prod_dist)
    top_item_shift = build_top_item_shift(train_dist=train_dist, prod_dist=prod_dist, top_n=10)

    warnings: List[str] = []

    if fallback_rate > float(thresholds.get("fallback_rate_warn", 0.40)):
        warnings.append(
            f"Alert: fallback_rate={fallback_rate:.4f} exceeded threshold {float(thresholds.get('fallback_rate_warn', 0.40)):.2f}."
        )

    if avg_latency_ms > float(thresholds.get("avg_latency_ms_warn", 1000.0)):
        warnings.append(
            f"Alert: avg_latency_ms={avg_latency_ms:.2f} exceeded threshold {float(thresholds.get('avg_latency_ms_warn', 1000.0)):.2f}."
        )

    if error_rate > float(thresholds.get("error_rate_warn", 0.02)):
        warnings.append(
            f"Alert: error_rate={error_rate:.4f} exceeded threshold {float(thresholds.get('error_rate_warn', 0.02)):.2f}."
        )

    warnings.extend(evaluate_drift_warnings(drift_score=drift_score, unknown_user_rate=unknown_user_rate, thresholds=thresholds))

    return {
        "service_metrics": {
            "request_count": total_requests,
            "success_count": success_count,
            "error_count": error_count,
            "success_rate": success_rate,
            "error_rate": error_rate,
            "avg_latency_ms": avg_latency_ms,
            "p95_latency_ms": p95_latency_ms,
        },
        "data_behavior": {
            "unique_users": unique_users,
            "personalized_count": personalized_count,
            "personalized_rate": personalized_rate,
            "fallback_count": fallback_count,
            "fallback_rate": fallback_rate,
            "unknown_user_rate": unknown_user_rate,
            "requests_per_user": requests_per_user,
            "avg_top_k": avg_top_k,
            "request_volume_per_day": volume_per_day,
        },
        "drift": {
            "drift_score": drift_score,
            "top_item_shift": [
                {
                    "item_id": item_id,
                    "train_probability": train_prob,
                    "production_probability": prod_prob,
                }
                for item_id, train_prob, prod_prob in top_item_shift
            ],
        },
        "warnings": warnings,
    }


def save_report(summary: Dict[str, object], output_md: Path, output_json: Path) -> None:
    output_md.parent.mkdir(parents=True, exist_ok=True)

    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    service = summary["service_metrics"]
    data = summary["data_behavior"]
    drift = summary["drift"]
    warnings = summary["warnings"]

    lines = [
        "# Monitoring Report",
        "",
        "## Service Metrics",
        f"- request_count: {service['request_count']}",
        f"- success_count: {service['success_count']}",
        f"- error_count: {service['error_count']}",
        f"- success_rate: {float(service['success_rate']):.4f}",
        f"- error_rate: {float(service['error_rate']):.4f}",
        f"- avg_latency_ms: {float(service['avg_latency_ms']):.2f}",
        f"- p95_latency_ms: {float(service['p95_latency_ms']):.2f}",
        "",
        "## Data Behavior",
        f"- unique_users: {int(data['unique_users'])}",
        f"- personalized_count: {int(data['personalized_count'])}",
        f"- personalized_rate: {float(data['personalized_rate']):.4f}",
        f"- fallback_count: {int(data['fallback_count'])}",
        f"- fallback_rate: {float(data['fallback_rate']):.4f}",
        f"- unknown_user_rate: {float(data['unknown_user_rate']):.4f}",
        f"- requests_per_user: {float(data['requests_per_user']):.4f}",
        f"- avg_top_k: {float(data['avg_top_k']):.2f}",
        "",
        "## Drift",
        f"- drift_score: {float(drift['drift_score']):.4f}",
        "",
        "### Top Item Shift (Production Top 10)",
    ]

    top_shift = drift.get("top_item_shift", [])
    if not top_shift:
        lines.append("- No sufficient recommendation distribution data.")
    else:
        for row in top_shift:
            lines.append(
                f"- item_id={row['item_id']}: train_prob={float(row['train_probability']):.4f}, prod_prob={float(row['production_probability']):.4f}"
            )

    lines.append("")
    lines.append("## Alerts")
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- No alert triggered.")

    with open(output_md, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate monitoring report from request logs and drift checks.")
    parser.add_argument("--monitoring-config", type=str, default="configs/monitoring.yaml")
    parser.add_argument("--api-config", type=str, default="configs/api.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    monitoring_cfg = load_yaml(project_root / args.monitoring_config)
    api_cfg = load_yaml(project_root / args.api_config)

    log_path = project_root / str(monitoring_cfg.get("logging", {}).get("request_log_path", "reports/monitoring/request_logs.jsonl"))
    output_md = project_root / str(monitoring_cfg.get("report", {}).get("output_markdown", "reports/monitoring/monitoring_report.md"))
    output_json = project_root / str(monitoring_cfg.get("report", {}).get("output_json", "reports/monitoring/monitoring_report.json"))

    logs_df = load_request_logs(log_path)
    summary = build_monitoring_summary(project_root=project_root, logs_df=logs_df, monitoring_cfg=monitoring_cfg, api_cfg=api_cfg)
    save_report(summary=summary, output_md=output_md, output_json=output_json)

    print(f"Saved monitoring report markdown: {output_md}")
    print(f"Saved monitoring report json: {output_json}")
    if summary["warnings"]:
        print("Alerts triggered:")
        for warning in summary["warnings"]:
            print(f"- {warning}")


if __name__ == "__main__":
    main()

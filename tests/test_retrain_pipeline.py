"""Unit tests for Week 14 retraining pipeline decision logic."""

from __future__ import annotations

import pytest

from src.pipeline.retrain_pipeline import build_metric_deltas, decide_promotion, should_retrain_triggered, summarize_registry_model


def test_promote_when_recall_improves() -> None:
    decision = decide_promotion(
        current_metrics={"recall_at_k": 0.1000, "coverage": 0.2200},
        candidate_metrics={"recall_at_k": 0.1200, "coverage": 0.2100},
    )
    assert decision.promote is True
    assert "recall_at_k improved" in decision.reason


def test_promote_when_recall_tied_and_coverage_improves() -> None:
    decision = decide_promotion(
        current_metrics={"recall_at_k": 0.1200, "coverage": 0.2200},
        candidate_metrics={"recall_at_k": 0.1200, "coverage": 0.2600},
    )
    assert decision.promote is True
    assert "coverage improved" in decision.reason


def test_keep_when_recall_and_coverage_not_better() -> None:
    decision = decide_promotion(
        current_metrics={"recall_at_k": 0.1200, "coverage": 0.3000},
        candidate_metrics={"recall_at_k": 0.1190, "coverage": 0.2900},
    )
    assert decision.promote is False
    assert "did not satisfy promotion rule" in decision.reason


def test_trigger_based_retrain_threshold() -> None:
    should_run, reason = should_retrain_triggered(strategy="trigger", drift_score=0.25, drift_threshold=0.20)
    assert should_run is True
    assert "activated" in reason


def test_trigger_based_skip_when_below_threshold() -> None:
    should_run, reason = should_retrain_triggered(strategy="trigger", drift_score=0.10, drift_threshold=0.20)
    assert should_run is False
    assert "skipped" in reason


def test_schedule_based_always_runs() -> None:
    should_run, reason = should_retrain_triggered(strategy="schedule", drift_score=None, drift_threshold=0.20)
    assert should_run is True
    assert "Schedule-based" in reason


def test_build_metric_deltas() -> None:
    deltas = build_metric_deltas(
        current_metrics={"recall_at_k": 0.10, "coverage": 0.20},
        candidate_metrics={"recall_at_k": 0.12, "coverage": 0.18},
    )
    assert deltas["recall_at_k_delta"] == pytest.approx(0.02)
    assert deltas["coverage_delta"] == pytest.approx(-0.02)


def test_summarize_registry_model_defaults() -> None:
    summary = summarize_registry_model({})
    assert summary["name"] == "unknown"
    assert summary["version"] == "unknown"
    assert summary["artifact_path"] == "N/A"

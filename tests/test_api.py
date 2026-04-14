"""API smoke tests for serving endpoints."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient


def _create_tiny_bpr_bundle(path: Path) -> None:
    bundle = {
        "algorithm": "bpr_mf_numpy",
        "user_map": {1: 0, 2: 1},
        "item_map": {10: 0, 11: 1, 12: 2, 13: 3},
        "reverse_item_map": {0: 10, 1: 11, 2: 12, 3: 13},
        "train_user_seen_items": {1: {10, 11}, 2: {11, 12}},
        "user_factors": np.array([[0.3, 0.2], [0.1, 0.4]], dtype=np.float32),
        "item_factors": np.array([[0.5, 0.2], [0.2, 0.1], [0.7, 0.6], [0.8, 0.7]], dtype=np.float32),
        "user_bias": np.array([0.0, 0.0], dtype=np.float32),
        "item_bias": np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "item_ids": [10, 11, 12, 13],
        "item_popularity_order": [13, 12, 11, 10],
        "item_popularity_scores": {13: 4, 12: 3, 11: 2, 10: 1},
    }
    with open(path, "wb") as file:
        pickle.dump(bundle, file)


def _create_registry(tmp_path: Path) -> Path:
    model_path = tmp_path / "bpr_tiny.pkl"
    baseline_path = tmp_path / "baseline.parquet"
    registry_path = tmp_path / "registry.json"

    _create_tiny_bpr_bundle(model_path)
    pd.DataFrame(
        {
            "movie_id": [13, 12, 11, 10],
            "score": [10.0, 9.0, 8.0, 7.0],
            "interaction_count": [100, 80, 60, 40],
        }
    ).to_parquet(baseline_path, index=False)

    registry = {
        "active_model": {
            "name": "bpr",
            "version": "vtest",
            "artifact_path": str(model_path),
        },
        "fallback": {
            "type": "popularity",
            "artifact_path": str(baseline_path),
        },
    }
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    return registry_path


def test_health_and_recommend_endpoints(tmp_path: Path, monkeypatch) -> None:
    registry_path = _create_registry(tmp_path)
    monkeypatch.setenv("APP_REGISTRY_PATH", str(registry_path))
    monkeypatch.setenv("APP_DEFAULT_TOP_K", "5")

    from src.serving.app import create_app

    with TestClient(create_app()) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["active_model"] == "bpr"

        known = client.get("/recommend/1?top_k=3")
        assert known.status_code == 200
        known_json = known.json()
        assert known_json["strategy"] == "bpr"
        # Số lượng recommendation có thể nhỏ hơn top_k nếu user đã xem gần hết
        assert 0 < len(known_json["recommendations"]) <= 3

        unknown = client.get("/recommend/999?top_k=2")
        assert unknown.status_code == 200
        unknown_json = unknown.json()
        assert unknown_json["strategy"] == "popularity_fallback"
        assert unknown_json["recommendations"] == [13, 12]


def test_startup_with_missing_active_model_uses_rolled_back_from(tmp_path: Path, monkeypatch) -> None:
    missing_active_model = tmp_path / "missing_active.pkl"
    rolled_back_model = tmp_path / "bpr_rolled_back.pkl"
    baseline_path = tmp_path / "baseline.parquet"
    registry_path = tmp_path / "registry_missing_active.json"

    _create_tiny_bpr_bundle(rolled_back_model)
    pd.DataFrame(
        {
            "movie_id": [13, 12, 11, 10],
            "score": [10.0, 9.0, 8.0, 7.0],
            "interaction_count": [100, 80, 60, 40],
        }
    ).to_parquet(baseline_path, index=False)

    registry = {
        "active_model": {
            "name": "active_model_prev",
            "version": "rollback-bad",
            "artifact_path": str(missing_active_model),
        },
        "fallback": {
            "type": "popularity",
            "artifact_path": str(baseline_path),
        },
        "metadata": {
            "rolled_back_from": {
                "name": "bpr",
                "version": "vtest-rollback",
                "artifact_path": str(rolled_back_model),
            }
        },
    }
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    monkeypatch.setenv("APP_REGISTRY_PATH", str(registry_path))
    monkeypatch.setenv("APP_DEFAULT_TOP_K", "5")

    from src.serving.app import create_app

    with TestClient(create_app()) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["active_model"] == "bpr"
        assert health.json()["model_version"] == "vtest-rollback"

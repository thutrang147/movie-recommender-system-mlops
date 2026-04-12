"""Create lightweight serving artifacts for CI smoke tests.

This script is intended for CI environments where DVC model artifacts are not
available, such as forked pull requests without access to repository secrets.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]

    model_dir = project_root / "models" / "personalized"
    baseline_dir = project_root / "models" / "baseline"
    registry_path = project_root / "models" / "registry.json"

    model_dir.mkdir(parents=True, exist_ok=True)
    baseline_dir.mkdir(parents=True, exist_ok=True)

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
        "config": {
            "factors": 2,
            "learning_rate": 0.05,
            "reg": 0.001,
            "epochs": 1,
            "n_samples_per_epoch": 10,
            "random_state": 42,
            "top_k": 10,
            "relevance_threshold": 4.0,
            "patience": 1,
        },
        "best_epoch": 1,
        "best_val_map": 0.1,
    }

    model_path = model_dir / "bpr_model.pkl"
    with open(model_path, "wb") as file:
        pickle.dump(bundle, file)

    (model_dir / "bpr_model.json").write_text(
        json.dumps(
            {
                "algorithm": "bpr_mf_numpy",
                "artifact_path": "models/personalized/bpr_model.pkl",
                "note": "Temporary CI smoke artifact",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    pd.DataFrame(
        {
            "movie_id": [13, 12, 11, 10],
            "score": [10.0, 9.0, 8.0, 7.0],
            "interaction_count": [100, 80, 60, 40],
        }
    ).to_parquet(baseline_dir / "most_popular_items.parquet", index=False)

    registry = {
        "active_model": {
            "name": "bpr",
            "version": "ci-smoke",
            "artifact_path": "models/personalized/bpr_model.pkl",
        },
        "fallback": {
            "type": "popularity",
            "artifact_path": "models/baseline/most_popular_items.parquet",
        },
        "metadata": {
            "updated_at": "ci-smoke",
            "frozen_test_split": "data/split/test.parquet",
        },
    }
    registry_path.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")

    print(f"Created smoke artifacts at {model_path} and {registry_path}")


if __name__ == "__main__":
    main()

"""Batch recommendation job for offline inference output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd  # type: ignore[import-not-found]

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.serving.predictor import RecommendationPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch recommendations and save JSONL output.")
    parser.add_argument("--user-ids-path", type=str, required=True, help="CSV/Parquet file with user_id column.")
    parser.add_argument("--output-path", type=str, default="reports/evaluation/batch_recommendations.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--registry-path", type=str, default="models/registry.json")

    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[2]

    input_path = Path(args.user_ids_path)
    if not input_path.is_absolute():
        input_path = project_root / input_path

    if input_path.suffix == ".parquet":
        users_df = pd.read_parquet(input_path)
    else:
        users_df = pd.read_csv(input_path)

    if "user_id" not in users_df.columns:
        raise ValueError("Input file must contain user_id column.")

    registry_path = Path(args.registry_path)
    if not registry_path.is_absolute():
        registry_path = project_root / registry_path

    predictor = RecommendationPredictor(project_root=project_root, registry_path=registry_path, default_top_k=args.top_k)
    predictor.load()

    output_path = Path(args.output_path)
    if not output_path.is_absolute():
        output_path = project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as file:
        for user_id in users_df["user_id"].astype(int).tolist():
            result = predictor.recommend(user_id=user_id, top_k=args.top_k)
            file.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Saved batch recommendations to: {output_path}")


if __name__ == "__main__":
    main()

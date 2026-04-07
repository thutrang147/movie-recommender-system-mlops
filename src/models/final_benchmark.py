"""Build a final benchmark table across Baseline, SVD, BPR, and Content-based models."""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd  # type: ignore[import-not-found]
import yaml

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.evaluate import evaluate_personalized_model


def load_yaml(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dictionary: {path}")
    return data


def parse_baseline_report(path: Path) -> Dict[str, float | int]:
    if not path.exists():
        raise FileNotFoundError(f"Missing baseline report: {path}")

    text = path.read_text(encoding="utf-8")
    test_block = re.search(r"## Test Metrics\n(.*?)\n## Notes", text, re.S)
    if not test_block:
        raise ValueError("Could not find test metrics section in baseline report.")

    most_popular = re.search(
        r"### most_popular\n"
        r"- users_evaluated: (\d+)\n"
        r"- recall_at_k: ([0-9.]+)\n"
        r"- map_at_k: ([0-9.]+)\n"
        r"- hit_rate_at_k: ([0-9.]+)\n"
        r"- coverage: ([0-9.]+)",
        test_block.group(1),
    )
    if not most_popular:
        raise ValueError("Could not parse most_popular baseline metrics from report.")

    return {
        "users_evaluated": int(most_popular.group(1)),
        "recall_at_k": float(most_popular.group(2)),
        "map_at_k": float(most_popular.group(3)),
        "hit_rate_at_k": float(most_popular.group(4)),
        "coverage": float(most_popular.group(5)),
    }


def load_bundle(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    with open(path, "rb") as file:
        bundle = pickle.load(file)
    if not isinstance(bundle, dict):
        raise ValueError(f"Model artifact is not a dict bundle: {path}")
    return bundle


def render_markdown(rows: List[Dict[str, float | int | str]], top_k: int, threshold: float) -> str:
    lines = [
        "# Final Model Comparison",
        "",
        "| Model | users_evaluated | Recall@10 | MAP@10 | HitRate@10 | Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {Model} | {users_evaluated} | {Recall@10:.4f} | {MAP@10:.4f} | {HitRate@10:.4f} | {Coverage:.4f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Notes",
            f"- All models evaluated on the same test split with top_k={top_k} and relevance_threshold={threshold}.",
            "- Baseline row uses the most_popular strategy from baseline_report.md.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final benchmark table for Baseline/SVD/BPR/Content-based.")
    parser.add_argument("--data-config", type=str, default="configs/data.yaml", help="Path to data config YAML.")
    parser.add_argument("--model-config", type=str, default="configs/model.yaml", help="Path to model config YAML.")
    parser.add_argument("--eval-config", type=str, default="configs/evaluation.yaml", help="Path to evaluation config YAML.")

    args = parser.parse_args()

    data_cfg = load_yaml(project_root / args.data_config)
    model_cfg = load_yaml(project_root / args.model_config)
    eval_cfg = load_yaml(project_root / args.eval_config)

    train_path = project_root / str(data_cfg["paths"]["train_split"])  # type: ignore[index]
    test_path = project_root / str(data_cfg["paths"]["test_split"])  # type: ignore[index]
    baseline_report_path = project_root / str(data_cfg["paths"]["baseline_report"])  # type: ignore[index]

    top_k = int(eval_cfg["evaluation"]["top_k"])  # type: ignore[index]
    threshold = float(eval_cfg["evaluation"]["relevance_threshold"])  # type: ignore[index]

    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    baseline_metrics = parse_baseline_report(baseline_report_path)

    svd_bundle = load_bundle(project_root / str(model_cfg["artifacts"]["svd_bundle"]))  # type: ignore[index]
    bpr_bundle = load_bundle(project_root / str(model_cfg["artifacts"]["bpr_bundle"]))  # type: ignore[index]
    content_bundle = load_bundle(project_root / str(model_cfg["artifacts"]["content_bundle"]))  # type: ignore[index]

    rows: List[Dict[str, float | int | str]] = [
        {
            "Model": "Baseline",
            "users_evaluated": int(baseline_metrics["users_evaluated"]),
            "Recall@10": float(baseline_metrics["recall_at_k"]),
            "MAP@10": float(baseline_metrics["map_at_k"]),
            "HitRate@10": float(baseline_metrics["hit_rate_at_k"]),
            "Coverage": float(baseline_metrics["coverage"]),
        }
    ]

    for model_name, bundle in [
        ("SVD", svd_bundle),
        ("BPR", bpr_bundle),
        ("Content-based", content_bundle),
    ]:
        metrics = evaluate_personalized_model(
            bundle=bundle,
            train_df=train_df,
            test_df=test_df,
            top_k=top_k,
            relevance_threshold=threshold,
        )
        rows.append(
            {
                "Model": model_name,
                "users_evaluated": int(metrics["users_evaluated"]),
                "Recall@10": float(metrics["recall_at_k"]),
                "MAP@10": float(metrics["map_at_k"]),
                "HitRate@10": float(metrics["hit_rate_at_k"]),
                "Coverage": float(metrics["coverage"]),
            }
        )

    markdown = render_markdown(rows=rows, top_k=top_k, threshold=threshold)
    out_md = project_root / str(eval_cfg["outputs"]["final_comparison"])  # type: ignore[index]
    out_json = project_root / str(eval_cfg["outputs"]["final_comparison_json"])  # type: ignore[index]

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(markdown, encoding="utf-8")
    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Saved markdown benchmark: {out_md}")
    print(f"Saved json benchmark: {out_json}")


if __name__ == "__main__":
    main()

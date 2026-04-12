"""Generate realistic request logs for monitoring and demo recordings."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.monitoring.logger import MonitoringLogger
from src.serving.predictor import RecommendationPredictor


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate request logs for monitoring demos.")
    parser.add_argument("--registry-path", type=str, default="models/registry.json")
    parser.add_argument("--log-path", type=str, default="reports/monitoring/request_logs.jsonl")
    parser.add_argument("--known-users", type=int, nargs="+", default=None)
    parser.add_argument("--unknown-users", type=int, nargs="+", default=[900001, 900002, 900003, 900004])
    parser.add_argument("--known-requests", type=int, default=24)
    parser.add_argument("--unknown-requests", type=int, default=8)
    parser.add_argument("--default-top-k", type=int, default=10)
    parser.add_argument("--known-user-pool-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset-log", action="store_true", help="Delete the existing request log before writing new events.")
    return parser


def resolve_project_path(raw_path: str) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else project_root / path


def generate_requests(
    predictor: RecommendationPredictor,
    logger: MonitoringLogger,
    known_users: list[int],
    unknown_users: list[int],
    known_requests: int,
    unknown_requests: int,
    seed: int,
) -> dict[str, int]:
    rng = random.Random(seed)

    total = 0
    fallback = 0

    for _ in range(known_requests):
        user_id = rng.choice(known_users)
        top_k = rng.choice([5, 10])
        result = predictor.recommend(user_id=user_id, top_k=top_k)
        logger.log_request(
            user_id=user_id,
            strategy=str(result["strategy"]),
            latency_ms=rng.uniform(18.0, 75.0),
            top_k=top_k,
            response_status=200,
            recommendations=[int(item_id) for item_id in result["recommendations"]],
        )
        total += 1

    for _ in range(unknown_requests):
        user_id = rng.choice(unknown_users)
        top_k = rng.choice([5, 10])
        result = predictor.recommend(user_id=user_id, top_k=top_k)
        logger.log_request(
            user_id=user_id,
            strategy=str(result["strategy"]),
            latency_ms=rng.uniform(12.0, 45.0),
            top_k=top_k,
            response_status=200,
            recommendations=[int(item_id) for item_id in result["recommendations"]],
        )
        total += 1
        fallback += 1

    return {"total_requests": total, "fallback_requests": fallback}


def resolve_known_users(predictor: RecommendationPredictor, cli_known_users: list[int] | None, pool_size: int) -> list[int]:
    if cli_known_users:
        return [int(user_id) for user_id in cli_known_users]

    inferred = sorted(int(user_id) for user_id in predictor.train_user_seen_items.keys())
    if not inferred:
        raise ValueError("Could not infer known users from the active model bundle.")
    return inferred[: max(1, pool_size)]


def main() -> None:
    args = build_parser().parse_args()

    registry_path = resolve_project_path(args.registry_path)
    log_path = resolve_project_path(args.log_path)

    if args.reset_log and log_path.exists():
        log_path.unlink()

    predictor = RecommendationPredictor(
        project_root=project_root,
        registry_path=registry_path,
        default_top_k=args.default_top_k,
    )
    predictor.load()
    known_users = resolve_known_users(
        predictor=predictor,
        cli_known_users=args.known_users,
        pool_size=int(args.known_user_pool_size),
    )

    logger = MonitoringLogger(log_path=log_path)
    summary = generate_requests(
        predictor=predictor,
        logger=logger,
        known_users=known_users,
        unknown_users=[int(user_id) for user_id in args.unknown_users],
        known_requests=int(args.known_requests),
        unknown_requests=int(args.unknown_requests),
        seed=int(args.seed),
    )

    print(f"Generated {summary['total_requests']} request log events at {log_path}.")
    print(f"Fallback requests: {summary['fallback_requests']}")


if __name__ == "__main__":
    main()

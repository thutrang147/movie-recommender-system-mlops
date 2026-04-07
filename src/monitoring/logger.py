"""Request logging utilities for online monitoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class RequestLogEvent:
    timestamp: str
    user_id: int
    strategy: str
    latency_ms: float
    top_k: int
    response_status: int
    recommendations: List[int]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "user_id": self.user_id,
            "strategy": self.strategy,
            "latency_ms": self.latency_ms,
            "top_k": self.top_k,
            "response_status": self.response_status,
            "recommendations": self.recommendations,
        }


class MonitoringLogger:
    """Append request-level monitoring events to JSONL log files."""

    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_request(
        self,
        user_id: int,
        strategy: str,
        latency_ms: float,
        top_k: int,
        response_status: int,
        recommendations: List[int] | None = None,
    ) -> None:
        event = RequestLogEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            user_id=int(user_id),
            strategy=str(strategy),
            latency_ms=float(latency_ms),
            top_k=int(top_k),
            response_status=int(response_status),
            recommendations=[int(item_id) for item_id in (recommendations or [])],
        )

        with open(self.log_path, "a", encoding="utf-8") as file:
            file.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

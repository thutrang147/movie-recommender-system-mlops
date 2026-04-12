"""FastAPI serving app for movie recommendations."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from time import perf_counter
from pathlib import Path

import yaml
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from src.serving.predictor import RecommendationPredictor
from src.monitoring.logger import MonitoringLogger


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    status: str
    active_model: str
    model_version: str


class RecommendResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    user_id: int
    strategy: str
    model_version: str
    recommendations: list[int]


def load_serving_config(project_root: Path) -> tuple[Path, int, Path]:
    config_path = project_root / "configs" / "api.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as file:
            cfg = yaml.safe_load(file) or {}
        serving = cfg.get("serving", {})
        registry_path = project_root / str(serving.get("registry_path", "models/registry.json"))
        default_top_k = int(serving.get("default_top_k", 10))
        request_log_path = project_root / str(serving.get("request_log_path", "reports/monitoring/request_logs.jsonl"))
    else:
        registry_path = project_root / "models" / "registry.json"
        default_top_k = 10
        request_log_path = project_root / "reports" / "monitoring" / "request_logs.jsonl"

    env_registry = os.getenv("APP_REGISTRY_PATH")
    if env_registry:
        registry_path = Path(env_registry)
        if not registry_path.is_absolute():
            registry_path = project_root / registry_path

    env_top_k = os.getenv("APP_DEFAULT_TOP_K")
    if env_top_k:
        default_top_k = int(env_top_k)

    env_log_path = os.getenv("APP_REQUEST_LOG_PATH")
    if env_log_path:
        request_log_path = Path(env_log_path)
        if not request_log_path.is_absolute():
            request_log_path = project_root / request_log_path

    return registry_path, default_top_k, request_log_path


def create_app() -> FastAPI:
    project_root = Path(__file__).resolve().parents[2]

    registry_path, default_top_k, request_log_path = load_serving_config(project_root)
    predictor = RecommendationPredictor(project_root=project_root, registry_path=registry_path, default_top_k=default_top_k)
    monitor_logger = MonitoringLogger(log_path=request_log_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        predictor.load()
        app.state.predictor = predictor
        yield

    app = FastAPI(title="Movie Recommender API", version="1.0.0", lifespan=lifespan)

    def get_predictor() -> RecommendationPredictor:
        pred = getattr(app.state, "predictor", None)
        if pred is None:
            predictor.load()
            app.state.predictor = predictor
            pred = predictor
        return pred

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        pred = get_predictor()
        if pred.model_info is None:
            raise HTTPException(status_code=503, detail="Model is not loaded")
        return HealthResponse(
            status="ok",
            active_model=pred.model_info.name,
            model_version=pred.model_info.version,
        )

    @app.get("/recommend/{user_id}", response_model=RecommendResponse)
    def recommend(user_id: int, top_k: int = Query(10, ge=1, le=100)) -> RecommendResponse:
        pred = get_predictor()
        started = perf_counter()
        response_status = 200
        strategy = "unknown"
        recommendations: list[int] = []
        try:
            result = pred.recommend(user_id=user_id, top_k=top_k)
            strategy = str(result.get("strategy", "unknown"))
            recommendations = [int(item_id) for item_id in result.get("recommendations", [])]
        except Exception as exc:
            response_status = 500
            strategy = "error"
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        finally:
            latency_ms = (perf_counter() - started) * 1000.0
            monitor_logger.log_request(
                user_id=user_id,
                strategy=strategy,
                latency_ms=latency_ms,
                top_k=top_k,
                response_status=response_status,
                recommendations=recommendations,
            )
        return RecommendResponse(**result)

    return app


app = create_app()

from __future__ import annotations

from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from agroeye_decision_maker.runtime import DecisionRuntime
from agroeye_decision_maker.utils.config import load_yaml


class DecideRequest(BaseModel):
    timestamp_utc: str | None = None
    sensors: dict[str, float | None] = Field(default_factory=dict)
    override_config: dict[str, Any] | None = None


class HealthResponse(BaseModel):
    status: str
    model_mode: str


def create_app(config_path: str = "configs/base.yaml", safety_path: str = "configs/safety.yaml") -> FastAPI:
    cfg = load_yaml(config_path)
    safety_cfg = load_yaml(safety_path)
    runtime = DecisionRuntime(cfg, safety_cfg)

    app = FastAPI(title="AgroEye Decision Maker", version="0.1.0")

    @app.post("/control/decide")
    def decide(payload: DecideRequest) -> dict[str, Any]:
        return runtime.decide(payload.timestamp_utc, payload.sensors, payload.override_config)

    @app.post("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(status="ok", model_mode=runtime.mode)

    @app.get("/health", response_model=HealthResponse)
    def health_get() -> HealthResponse:
        return HealthResponse(status="ok", model_mode=runtime.mode)

    return app


app = create_app()

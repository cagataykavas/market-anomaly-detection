from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, FiniteFloat

from src.engine import MarketAnomalyEngine
from src.synthetic import synthetic_ohlcv

app = FastAPI(title="Market Anomaly Detection", version="1.0.0")
ENGINE = MarketAnomalyEngine()


class MarketRow(BaseModel):
    timestamp: str = Field(min_length=1, max_length=100)
    open: FiniteFloat = Field(gt=0)
    high: FiniteFloat = Field(gt=0)
    low: FiniteFloat = Field(gt=0)
    close: FiniteFloat = Field(gt=0)
    volume: FiniteFloat = Field(ge=0)


class AnalysisRequest(BaseModel):
    rows: list[MarketRow] = Field(min_length=80, max_length=100_000)
    contamination: float = Field(default=0.02, gt=0, lt=0.5)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/demo")
def demo(seed: int = 42) -> dict[str, object]:
    frame, _ = synthetic_ohlcv(seed=seed)
    return ENGINE.analyze(frame, seed=seed)


@app.post("/analyze")
def analyze(payload: AnalysisRequest) -> dict[str, object]:
    try:
        frame = pd.DataFrame([row.model_dump() for row in payload.rows])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame.set_index("timestamp").sort_index()
        return ENGINE.analyze(frame, contamination=payload.contamination)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from app.api import app
from src.engine import MarketAnomalyEngine
from src.evaluation import event_detection_metrics
from src.synthetic import synthetic_ohlcv


def test_engine_flags_some_injected_market_events():
    frame, labels = synthetic_ohlcv(rows=600, seed=42)
    result = MarketAnomalyEngine().analyze(frame, seed=42)
    flags = pd.Series({pd.Timestamp(key): value for key, value in result["flags"].items()}).sort_index()
    metrics = event_detection_metrics(flags, labels, tolerance_days=2)
    assert result["anomaly_count"] > 0
    assert metrics["matched_events"] >= 1


def test_events_have_explanations_and_regimes():
    frame, _ = synthetic_ohlcv(rows=500, seed=9)
    result = MarketAnomalyEngine().analyze(frame, seed=9)
    assert all("regime" in event for event in result["events"])
    assert all(event["reasons"] for event in result["events"])


def test_demo_api():
    response = TestClient(app).get("/demo?seed=42")
    assert response.status_code == 200
    payload = response.json()
    assert payload["rows"] == 900
    assert payload["anomaly_count"] > 0

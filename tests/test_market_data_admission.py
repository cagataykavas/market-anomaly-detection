from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.api import app
from src.admission import validate_market_frame
from src.engine import MarketAnomalyEngine
from src.synthetic import synthetic_ohlcv


def _frame(rows: int = 100) -> pd.DataFrame:
    frame, _ = synthetic_ohlcv(rows=rows, seed=17)
    return frame


def test_valid_market_frame_is_admitted() -> None:
    frame = _frame()

    validate_market_frame(frame)
    result = MarketAnomalyEngine().analyze(frame, seed=17)

    assert result["rows"] == len(frame)


def test_duplicate_timestamps_are_rejected_before_scoring() -> None:
    frame = pd.concat([_frame(), _frame().iloc[[-1]]]).sort_index()

    with pytest.raises(ValueError, match="timestamps must be unique"):
        MarketAnomalyEngine().analyze(frame)


def test_out_of_order_timestamps_are_rejected() -> None:
    frame = _frame().iloc[::-1]

    with pytest.raises(ValueError, match="strictly increasing"):
        validate_market_frame(frame)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.__setitem__("close", np.nan), "values must be finite"),
        (
            lambda frame: frame.__setitem__("volume", -1.0),
            "volume must be non-negative",
        ),
        (lambda frame: frame.__setitem__("low", 0.0), "prices must be positive"),
    ],
)
def test_invalid_numeric_values_are_rejected(mutation, message: str) -> None:
    frame = _frame()
    mutation(frame)

    with pytest.raises(ValueError, match=message):
        validate_market_frame(frame)


def test_inconsistent_candle_bounds_are_rejected() -> None:
    frame = _frame()
    frame.loc[frame.index[0], "high"] = frame.loc[frame.index[0], "low"] * 0.5

    with pytest.raises(ValueError, match="candle bounds are inconsistent in 1 rows"):
        validate_market_frame(frame)


def test_missing_columns_are_rejected_with_a_bounded_error() -> None:
    with pytest.raises(ValueError, match="missing required columns: volume"):
        validate_market_frame(_frame().drop(columns="volume"))


def test_api_rejects_duplicate_timestamps() -> None:
    frame = _frame(rows=80).reset_index(names="timestamp")
    rows = frame.to_dict(orient="records")
    rows[-1]["timestamp"] = rows[-2]["timestamp"]
    rows = [
        {**row, "timestamp": pd.Timestamp(row["timestamp"]).isoformat()} for row in rows
    ]

    response = TestClient(app).post("/analyze", json={"rows": rows})

    assert response.status_code == 422
    assert response.json()["detail"] == "market timestamps must be unique"

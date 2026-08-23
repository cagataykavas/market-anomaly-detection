from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_ohlcv(rows: int = 900, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    index = pd.date_range("2022-01-03", periods=rows, freq="B")
    volatility = np.full(rows, 0.010)
    volatility[rows // 3 : rows // 3 + 90] = 0.018
    volatility[2 * rows // 3 : 2 * rows // 3 + 70] = 0.006
    returns = rng.normal(0.00025, volatility)
    labels = np.zeros(rows, dtype=bool)

    # Known injected events for evaluation. These are construction labels, not claims
    # about any real market event.
    event_positions = [int(rows * 0.22), int(rows * 0.47), int(rows * 0.71), int(rows * 0.86)]
    shocks = [-0.075, 0.065, -0.095, 0.080]
    for pos, shock in zip(event_positions, shocks):
        if 2 <= pos < rows - 2:
            returns[pos] += shock
            labels[pos] = True

    close = 100.0 * np.cumprod(1.0 + returns)
    overnight = rng.normal(0.0, 0.0025, rows)
    open_price = np.r_[100.0, close[:-1]] * (1.0 + overnight)
    intraday_range = np.abs(rng.normal(0.009, 0.004, rows)) + np.abs(returns) * 0.55
    high = np.maximum(open_price, close) * (1.0 + intraday_range / 2)
    low = np.minimum(open_price, close) * (1.0 - intraday_range / 2)
    volume = rng.lognormal(15.0, 0.35, rows)
    for pos in event_positions:
        if 0 <= pos < rows:
            volume[pos] *= 5.5

    frame = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
    return frame, pd.Series(labels, index=index, name="injected_event")

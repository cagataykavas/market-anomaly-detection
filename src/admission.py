from __future__ import annotations

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_numeric_dtype

MARKET_COLUMNS = ("open", "high", "low", "close", "volume")
MIN_MARKET_ROWS = 80
MAX_MARKET_ROWS = 100_000


def validate_market_frame(
    frame: pd.DataFrame,
    *,
    min_rows: int = MIN_MARKET_ROWS,
    max_rows: int = MAX_MARKET_ROWS,
) -> None:
    """Fail closed when an OHLCV frame cannot support causal feature computation."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("market input must be a pandas DataFrame")
    if not 1 <= min_rows <= max_rows:
        raise ValueError("row limits must satisfy 1 <= min_rows <= max_rows")
    if not min_rows <= len(frame) <= max_rows:
        raise ValueError(f"market input must contain {min_rows} to {max_rows} rows")

    missing = sorted(set(MARKET_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(
            f"market input is missing required columns: {', '.join(missing)}"
        )

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("market input index must be a DatetimeIndex")
    if frame.index.hasnans:
        raise ValueError("market timestamps must not contain missing values")
    if not frame.index.is_unique:
        raise ValueError("market timestamps must be unique")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("market timestamps must be strictly increasing")

    invalid_types = [
        column
        for column in MARKET_COLUMNS
        if not is_numeric_dtype(frame[column]) or is_bool_dtype(frame[column])
    ]
    if invalid_types:
        raise ValueError(f"market columns must be numeric: {', '.join(invalid_types)}")

    values = frame.loc[:, MARKET_COLUMNS].to_numpy(dtype=float, copy=False)
    if not np.isfinite(values).all():
        raise ValueError("market values must be finite")
    if (frame.loc[:, ("open", "high", "low", "close")] <= 0).any(axis=None):
        raise ValueError("OHLC prices must be positive")
    if (frame["volume"] < 0).any():
        raise ValueError("volume must be non-negative")

    high_floor = frame.loc[:, ("open", "low", "close")].max(axis=1)
    low_ceiling = frame.loc[:, ("open", "high", "close")].min(axis=1)
    invalid_candles = (frame["high"] < high_floor) | (frame["low"] > low_ceiling)
    if invalid_candles.any():
        raise ValueError(
            f"OHLC candle bounds are inconsistent in {int(invalid_candles.sum())} rows"
        )

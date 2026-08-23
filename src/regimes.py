from __future__ import annotations

import numpy as np
import pandas as pd


def classify_volatility_regime(
    returns: pd.Series,
    *,
    lookback: int = 60,
    low_quantile: float = 0.33,
    high_quantile: float = 0.67,
) -> pd.Series:
    realized = returns.rolling(20, min_periods=10).std()
    low = realized.rolling(lookback, min_periods=max(20, lookback // 2)).quantile(low_quantile)
    high = realized.rolling(lookback, min_periods=max(20, lookback // 2)).quantile(high_quantile)
    labels = pd.Series("normal", index=returns.index, dtype="object")
    labels[realized <= low] = "low"
    labels[realized >= high] = "high"
    labels[realized.isna()] = "unknown"
    return labels


def regime_adjusted_threshold(
    base_threshold: float,
    regime: str,
    *,
    high_multiplier: float = 1.35,
    low_multiplier: float = 0.85,
) -> float:
    if regime == "high":
        return base_threshold * high_multiplier
    if regime == "low":
        return base_threshold * low_multiplier
    return base_threshold


def regime_aware_flags(scores: pd.Series, regimes: pd.Series, base_threshold: float) -> pd.Series:
    aligned = regimes.reindex(scores.index).fillna("normal")
    thresholds = aligned.map(lambda value: regime_adjusted_threshold(base_threshold, str(value))).astype(float)
    return (scores >= thresholds).astype(bool)

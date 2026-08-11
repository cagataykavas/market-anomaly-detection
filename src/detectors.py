from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


@dataclass(frozen=True)
class DetectionResult:
    scores: pd.Series
    flags: pd.Series


def robust_zscore(series: pd.Series, window: int = 50, threshold: float = 4.0) -> DetectionResult:
    median = series.rolling(window, min_periods=max(10, window // 3)).median()
    mad = (series - median).abs().rolling(window, min_periods=max(10, window // 3)).median()
    scale = 1.4826 * mad.replace(0, np.nan)
    z = ((series - median) / scale).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return DetectionResult(z.abs(), z.abs() >= threshold)


def build_market_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["return_1"] = out["close"].pct_change()
    out["abs_return"] = out["return_1"].abs()
    out["realized_vol_20"] = out["return_1"].rolling(20).std()
    out["volume_change"] = out["volume"].pct_change()
    out["range_pct"] = (out["high"] - out["low"]) / out["close"].replace(0, np.nan)
    return out.replace([np.inf, -np.inf], np.nan)


def isolation_forest(frame: pd.DataFrame, contamination: float = 0.02, seed: int = 42) -> DetectionResult:
    features = build_market_features(frame)[["return_1", "abs_return", "realized_vol_20", "volume_change", "range_pct"]].dropna()
    model = IsolationForest(n_estimators=300, contamination=contamination, random_state=seed)
    model.fit(features)
    score = pd.Series(-model.score_samples(features), index=features.index, name="anomaly_score")
    flags = pd.Series(model.predict(features) == -1, index=features.index, name="anomaly")
    return DetectionResult(score, flags)


def event_windows(frame: pd.DataFrame, flags: pd.Series, radius: int = 5) -> list[pd.DataFrame]:
    positions = [frame.index.get_loc(idx) for idx in flags.index[flags]]
    windows = []
    for pos in positions:
        start = max(0, pos - radius)
        stop = min(len(frame), pos + radius + 1)
        windows.append(frame.iloc[start:stop].copy())
    return windows

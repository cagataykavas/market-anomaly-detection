from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .detectors import build_market_features, isolation_forest, robust_zscore
from .regimes import classify_volatility_regime, regime_adjusted_threshold


@dataclass(frozen=True)
class AnomalyEvent:
    timestamp: str
    score: float
    severity: str
    regime: str
    return_1: float
    volume_change: float
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "score": self.score,
            "severity": self.severity,
            "regime": self.regime,
            "return_1": self.return_1,
            "volume_change": self.volume_change,
            "reasons": list(self.reasons),
        }


def _severity(score: float) -> str:
    if score >= 0.90:
        return "critical"
    if score >= 0.72:
        return "high"
    if score >= 0.52:
        return "medium"
    return "low"


class MarketAnomalyEngine:
    def analyze(
        self,
        frame: pd.DataFrame,
        *,
        robust_window: int = 50,
        robust_threshold: float = 4.0,
        contamination: float = 0.02,
        seed: int = 42,
    ) -> dict[str, Any]:
        features = build_market_features(frame)
        return_signal = robust_zscore(features["return_1"], robust_window, robust_threshold)
        volume_signal = robust_zscore(features["volume_change"], robust_window, robust_threshold)
        isolation = isolation_forest(frame, contamination=contamination, seed=seed)
        regimes = classify_volatility_regime(features["return_1"])

        common = isolation.scores.index.intersection(features.index)
        iso = isolation.scores.reindex(common)
        # Percentile-rank Isolation Forest scores so the ensemble is scale-free.
        iso_rank = iso.rank(pct=True).fillna(0.0)
        return_norm = (return_signal.scores.reindex(common) / max(robust_threshold, 1e-9)).clip(0, 2) / 2
        volume_norm = (volume_signal.scores.reindex(common) / max(robust_threshold, 1e-9)).clip(0, 2) / 2
        ensemble = (0.45 * iso_rank + 0.35 * return_norm + 0.20 * volume_norm).fillna(0.0)

        events: list[AnomalyEvent] = []
        score_series = pd.Series(ensemble, index=common, name="ensemble_score")
        flags = pd.Series(False, index=common)
        for timestamp in common:
            regime = str(regimes.get(timestamp, "normal"))
            threshold = regime_adjusted_threshold(0.62, regime, high_multiplier=1.15, low_multiplier=0.90)
            score = float(score_series.loc[timestamp])
            if score < threshold:
                continue
            flags.loc[timestamp] = True
            row = features.loc[timestamp]
            reasons: list[str] = []
            if float(return_signal.scores.get(timestamp, 0.0)) >= robust_threshold:
                reasons.append("extreme robust return z-score")
            if float(volume_signal.scores.get(timestamp, 0.0)) >= robust_threshold:
                reasons.append("extreme volume change")
            if bool(isolation.flags.get(timestamp, False)):
                reasons.append("multivariate Isolation Forest outlier")
            if regime == "high":
                reasons.append("high-volatility regime threshold applied")
            events.append(
                AnomalyEvent(
                    timestamp=str(timestamp),
                    score=score,
                    severity=_severity(score),
                    regime=regime,
                    return_1=float(row.get("return_1", np.nan)),
                    volume_change=float(row.get("volume_change", np.nan)),
                    reasons=tuple(reasons),
                )
            )

        return {
            "rows": len(frame),
            "feature_rows": len(common),
            "anomaly_count": len(events),
            "anomaly_rate": len(events) / len(common) if len(common) else 0.0,
            "events": [event.as_dict() for event in sorted(events, key=lambda item: item.score, reverse=True)],
            "scores": {str(index): float(value) for index, value in score_series.items()},
            "flags": {str(index): bool(value) for index, value in flags.items()},
        }

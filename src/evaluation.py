from __future__ import annotations

from typing import Any

import pandas as pd


def event_detection_metrics(
    predicted_flags: pd.Series,
    event_labels: pd.Series,
    *,
    tolerance_days: int = 2,
) -> dict[str, Any]:
    predicted = predicted_flags[predicted_flags].index
    events = event_labels[event_labels].index
    matched_predictions: set[object] = set()
    matched_events = 0

    for event in events:
        candidates = [
            timestamp
            for timestamp in predicted
            if abs((pd.Timestamp(timestamp) - pd.Timestamp(event)).days) <= tolerance_days
        ]
        if candidates:
            matched_events += 1
            closest = min(candidates, key=lambda timestamp: abs((pd.Timestamp(timestamp) - pd.Timestamp(event)).days))
            matched_predictions.add(closest)

    recall = matched_events / len(events) if len(events) else 0.0
    precision = len(matched_predictions) / len(predicted) if len(predicted) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "events": len(events),
        "predicted_flags": len(predicted),
        "matched_events": matched_events,
        "event_recall": recall,
        "event_precision": precision,
        "event_f1": f1,
        "tolerance_days": tolerance_days,
    }

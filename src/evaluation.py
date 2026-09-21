from __future__ import annotations

from typing import Any

import pandas as pd


def _timestamps(values: pd.Index, name: str) -> list[pd.Timestamp]:
    timestamps = [pd.Timestamp(value) for value in values]
    if any(pd.isna(value) for value in timestamps):
        raise ValueError(f"{name} contains an invalid timestamp")
    if len(set(timestamps)) != len(timestamps):
        raise ValueError(f"{name} contains duplicate timestamps")
    return sorted(timestamps)


def _optimal_matches(
    predictions: list[pd.Timestamp], events: list[pd.Timestamp], tolerance: pd.Timedelta
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Maximize one-to-one matches, then minimize total absolute delay."""
    rows, columns = len(events) + 1, len(predictions) + 1
    scores = [[(0, 0) for _ in range(columns)] for _ in range(rows)]
    choices = [["" for _ in range(columns)] for _ in range(rows)]

    for event_index in range(1, rows):
        choices[event_index][0] = "skip_event"
    for prediction_index in range(1, columns):
        choices[0][prediction_index] = "skip_prediction"

    for event_index in range(1, rows):
        for prediction_index in range(1, columns):
            candidates = [
                (scores[event_index - 1][prediction_index], "skip_event"),
                (scores[event_index][prediction_index - 1], "skip_prediction"),
            ]
            delay = predictions[prediction_index - 1] - events[event_index - 1]
            if abs(delay) <= tolerance:
                previous = scores[event_index - 1][prediction_index - 1]
                delay_ns = abs(delay.value)
                candidates.append(((previous[0] + 1, previous[1] - delay_ns), "match"))
            score, choice = max(candidates, key=lambda item: item[0])
            scores[event_index][prediction_index] = score
            choices[event_index][prediction_index] = choice

    matches: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    event_index, prediction_index = len(events), len(predictions)
    while event_index and prediction_index:
        choice = choices[event_index][prediction_index]
        if choice == "match":
            matches.append((events[event_index - 1], predictions[prediction_index - 1]))
            event_index -= 1
            prediction_index -= 1
        elif choice == "skip_event":
            event_index -= 1
        else:
            prediction_index -= 1
    return list(reversed(matches))


def event_detection_metrics(
    predicted_flags: pd.Series,
    event_labels: pd.Series,
    *,
    tolerance_days: int = 2,
) -> dict[str, Any]:
    """Evaluate timestamped alerts using optimal one-to-one temporal matching."""
    if isinstance(tolerance_days, bool) or not isinstance(tolerance_days, int):
        raise TypeError("tolerance_days must be an integer")
    if tolerance_days < 0:
        raise ValueError("tolerance_days must be non-negative")
    if predicted_flags.dtype != bool or event_labels.dtype != bool:
        raise TypeError("predicted_flags and event_labels must contain booleans")

    predictions = _timestamps(predicted_flags[predicted_flags].index, "predicted_flags")
    events = _timestamps(event_labels[event_labels].index, "event_labels")
    matches = _optimal_matches(predictions, events, pd.Timedelta(days=tolerance_days))
    matched_predictions = {prediction for _, prediction in matches}
    matched_events = {event for event, _ in matches}

    recall = len(matches) / len(events) if events else 0.0
    precision = len(matches) / len(predictions) if predictions else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    delays = [(prediction - event).total_seconds() / 86_400 for event, prediction in matches]
    evidence = [
        {
            "event_timestamp": event.isoformat(),
            "prediction_timestamp": prediction.isoformat(),
            "delay_days": delay,
        }
        for (event, prediction), delay in zip(matches, delays, strict=True)
    ]
    return {
        "events": len(events),
        "predicted_flags": len(predictions),
        "matched_events": len(matches),
        "false_positive_flags": len(predictions) - len(matches),
        "missed_events": len(events) - len(matches),
        "event_recall": recall,
        "event_precision": precision,
        "event_f1": f1,
        "tolerance_days": tolerance_days,
        "mean_absolute_delay_days": sum(abs(value) for value in delays) / len(delays)
        if delays
        else None,
        "max_absolute_delay_days": max((abs(value) for value in delays), default=None),
        "early_detections": sum(value < 0 for value in delays),
        "same_time_detections": sum(value == 0 for value in delays),
        "late_detections": sum(value > 0 for value in delays),
        "matches": evidence,
        "unmatched_event_timestamps": [
            value.isoformat() for value in events if value not in matched_events
        ],
        "unmatched_prediction_timestamps": [
            value.isoformat() for value in predictions if value not in matched_predictions
        ],
    }

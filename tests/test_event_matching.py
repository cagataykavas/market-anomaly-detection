import json

import pandas as pd
import pytest
from src.evaluation import event_detection_metrics


def series(true_dates, all_dates):
    true_dates = {pd.Timestamp(value) for value in true_dates}
    index = pd.DatetimeIndex(all_dates)
    return pd.Series([value in true_dates for value in index], index=index, dtype=bool)


def test_one_prediction_cannot_match_two_events():
    predictions = series(["2026-01-02"], ["2026-01-02"])
    events = series(["2026-01-01", "2026-01-03"], ["2026-01-01", "2026-01-03"])

    report = event_detection_metrics(predictions, events, tolerance_days=2)

    assert report["matched_events"] == 1
    assert report["event_precision"] == 1.0
    assert report["event_recall"] == 0.5
    assert report["missed_events"] == 1


def test_matching_maximizes_cardinality_before_minimizing_delay():
    predictions = series(["2026-01-02", "2026-01-04"], ["2026-01-02", "2026-01-04"])
    events = series(["2026-01-01", "2026-01-03"], ["2026-01-01", "2026-01-03"])

    report = event_detection_metrics(predictions, events, tolerance_days=2)

    assert report["matched_events"] == 2
    assert [item["delay_days"] for item in report["matches"]] == [1.0, 1.0]


def test_reports_early_late_and_unmatched_evidence_as_json():
    predictions = series(
        ["2026-01-09", "2026-01-21", "2026-02-01"],
        ["2026-01-09", "2026-01-21", "2026-02-01"],
    )
    events = series(["2026-01-10", "2026-01-20"], ["2026-01-10", "2026-01-20"])

    report = event_detection_metrics(predictions, events, tolerance_days=2)

    assert report["early_detections"] == 1
    assert report["late_detections"] == 1
    assert report["false_positive_flags"] == 1
    assert report["mean_absolute_delay_days"] == 1.0
    assert report["unmatched_prediction_timestamps"] == ["2026-02-01T00:00:00"]
    json.dumps(report)


def test_empty_evidence_is_explicit():
    empty = pd.Series([], index=pd.DatetimeIndex([]), dtype=bool)
    report = event_detection_metrics(empty, empty)

    assert report["event_precision"] == 0.0
    assert report["event_recall"] == 0.0
    assert report["mean_absolute_delay_days"] is None


@pytest.mark.parametrize("tolerance", [-1, 1.5, True])
def test_rejects_invalid_tolerance(tolerance):
    values = series(["2026-01-01"], ["2026-01-01"])
    with pytest.raises((TypeError, ValueError), match="tolerance_days"):
        event_detection_metrics(values, values, tolerance_days=tolerance)


def test_rejects_non_boolean_and_duplicate_timestamp_series():
    index = pd.DatetimeIndex(["2026-01-01"])
    with pytest.raises(TypeError, match="booleans"):
        event_detection_metrics(pd.Series([1], index=index), pd.Series([True], index=index))

    duplicate = pd.Series(
        [True, True], index=pd.DatetimeIndex(["2026-01-01", "2026-01-01"]), dtype=bool
    )
    with pytest.raises(ValueError, match="duplicate timestamps"):
        event_detection_metrics(duplicate, pd.Series([True], index=index, dtype=bool))

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.synthetic import synthetic_ohlcv
from src.walk_forward import (
    WalkForwardConfig,
    WalkForwardInputError,
    walk_forward_evaluate,
)


def market_frame(rows: int = 260, seed: int = 42) -> pd.DataFrame:
    frame, _ = synthetic_ohlcv(rows=rows, seed=seed)
    frame.index = frame.index.tz_localize("UTC")
    return frame


def config(**changes: object) -> WalkForwardConfig:
    values: dict[str, object] = {
        "min_history": 60,
        "training_window": 100,
        "refit_every": 15,
        "robust_window": 40,
        "regime_lookback": 60,
        "n_estimators": 25,
        "seed": 7,
        "max_rows": 2_000,
    }
    values.update(changes)
    return WalkForwardConfig(**values)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def report():
    return walk_forward_evaluate(market_frame(), config())


def test_records_prove_strictly_causal_windows(report) -> None:
    assert report.evaluated_rows > 100
    assert report.refit_count > 1
    for record in report.records:
        scored_at = pd.Timestamp(record.timestamp)
        assert pd.Timestamp(record.fit_ended_at) < scored_at
        assert pd.Timestamp(record.history_ended_at) < scored_at
        assert record.training_rows <= 100


def test_report_is_deterministic_and_privacy_preserving(report) -> None:
    second = walk_forward_evaluate(market_frame(), config())
    assert report.as_dict() == second.as_dict()
    encoded = json.dumps(report.as_dict())
    assert '"open"' not in encoded
    assert '"close"' not in encoded
    assert len(report.input_sha256) == 64
    assert len(report.config_sha256) == 64
    assert len(report.evidence_sha256) == 64


def test_evaluation_is_prefix_invariant() -> None:
    frame = market_frame(rows=300)
    short = walk_forward_evaluate(frame.iloc[:230], config())
    full = walk_forward_evaluate(frame, config())
    assert [item.as_dict() for item in short.records] == [
        item.as_dict() for item in full.records[: len(short.records)]
    ]


def test_future_mutation_cannot_rewrite_historical_scores() -> None:
    frame = market_frame(rows=300)
    changed = frame.copy()
    cutoff = 235
    changed.iloc[cutoff:, changed.columns.get_loc("close")] *= 1.25
    changed.iloc[cutoff:, changed.columns.get_loc("high")] *= 1.30
    changed.iloc[cutoff:, changed.columns.get_loc("low")] *= 0.80
    first = walk_forward_evaluate(frame, config())
    second = walk_forward_evaluate(changed, config())
    timestamp = frame.index[cutoff - 1].isoformat()
    first_prefix = [item.as_dict() for item in first.records if item.timestamp <= timestamp]
    second_prefix = [item.as_dict() for item in second.records if item.timestamp <= timestamp]
    assert first_prefix == second_prefix
    assert first.input_sha256 != second.input_sha256


def test_injected_events_produce_bounded_alert_evidence(report) -> None:
    assert 0 < report.anomaly_count < report.evaluated_rows
    assert report.anomaly_rate == pytest.approx(report.anomaly_count / report.evaluated_rows)
    assert all(0.0 <= item.score <= 1.0 for item in report.records)
    assert all(0.0 < item.threshold <= 1.0 for item in report.records)
    assert all(item.reasons for item in report.records if item.flagged)


@pytest.mark.parametrize(
    ("mutator", "code"),
    [
        (lambda value: value.set_axis(value.index.tz_localize(None)), "INDEX_NOT_TIMEZONE_AWARE"),
        (lambda value: value.sort_index(ascending=False), "TIMESTAMPS_NOT_SORTED"),
        (lambda value: value.assign(close=np.nan), "NON_FINITE_OHLCV"),
        (lambda value: value.assign(volume=0.0), "NON_POSITIVE_OHLCV"),
        (lambda value: value.assign(high=value["low"] - 1.0), "INVALID_HIGH_PRICE"),
        (lambda value: value.drop(columns="volume"), "MISSING_OHLCV_COLUMN"),
    ],
)
def test_market_contract_fails_closed(mutator, code: str) -> None:
    with pytest.raises(WalkForwardInputError, match=code):
        walk_forward_evaluate(mutator(market_frame()), config())


def test_duplicate_timestamps_have_specific_failure() -> None:
    frame = market_frame()
    duplicate = pd.concat([frame.iloc[:1], frame]).sort_index()
    with pytest.raises(WalkForwardInputError, match="DUPLICATE_TIMESTAMP"):
        walk_forward_evaluate(duplicate, config())


@pytest.mark.parametrize(
    "changes",
    [
        {"min_history": True},
        {"training_window": 50},
        {"refit_every": 101},
        {"contamination": 0.5},
        {"base_threshold": float("nan")},
        {"base_threshold": 0.9, "high_regime_multiplier": 1.2},
        {"high_regime_multiplier": 0.9},
        {"low_regime_multiplier": 1.1},
        {"seed": -1},
    ],
)
def test_configuration_rejects_unsafe_values(changes: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        config(**changes)


def test_row_budget_is_enforced() -> None:
    with pytest.raises(WalkForwardInputError, match="ROW_BUDGET_EXCEEDED"):
        walk_forward_evaluate(market_frame(), config(max_rows=200))


def test_configuration_digest_changes_with_policy(report) -> None:
    changed = walk_forward_evaluate(market_frame(), replace(config(), base_threshold=0.60))
    assert report.config_sha256 != changed.config_sha256
    assert report.evidence_sha256 != changed.evidence_sha256


def test_cli_writes_atomic_evidence_and_reports_malformed_input(tmp_path) -> None:
    output = tmp_path / "walk-forward.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.walk_forward",
            "--rows",
            "220",
            "--min-history",
            "60",
            "--training-window",
            "100",
            "--refit-every",
            "20",
            "--n-estimators",
            "20",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    malformed = subprocess.run(
        [sys.executable, "-m", "src.walk_forward", "--rows", "50"],
        check=False,
        capture_output=True,
        text=True,
    )
    naive_csv = tmp_path / "naive.csv"
    naive = market_frame(rows=220).tz_localize(None)
    naive.rename_axis("timestamp").reset_index().to_csv(naive_csv, index=False)
    naive_result = subprocess.run(
        [sys.executable, "-m", "src.walk_forward", "--input", str(naive_csv)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0
    assert json.loads(completed.stdout)["status"] == "completed"
    assert json.loads(output.read_text(encoding="utf-8"))["schema_version"] == 1
    assert not list(tmp_path.glob("*.tmp"))
    assert malformed.returncode == 3
    assert json.loads(malformed.stdout)["status"] == "malformed"
    assert naive_result.returncode == 3
    assert json.loads(naive_result.stdout)["reason_code"] == "TIMESTAMP_NOT_TIMEZONE_AWARE"

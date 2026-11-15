"""Metamorphic unit-invariance audit for market anomaly explanations."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .engine import MarketAnomalyEngine

REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
MAX_INPUT_BYTES = 50 * 1024 * 1024


class MarketAnalyzer(Protocol):
    def analyze(self, frame: pd.DataFrame, **kwargs: object) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class UnitScenario:
    name: str
    price_multiplier: float = 1.0
    volume_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or len(self.name) > 64:
            raise ValueError("scenario name must contain 1-64 characters")
        for field_name in ("price_multiplier", "volume_multiplier"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field_name} must be numeric")
            if not math.isfinite(value) or not 1e-9 <= value <= 1e9:
                raise ValueError(f"{field_name} must be finite and within [1e-9, 1e9]")


DEFAULT_SCENARIOS = (
    UnitScenario("price_to_cents", price_multiplier=100.0),
    UnitScenario("price_to_thousands", price_multiplier=0.001),
    UnitScenario("volume_to_lots", volume_multiplier=0.01),
    UnitScenario("volume_to_units", volume_multiplier=1_000.0),
)


@dataclass(frozen=True, slots=True)
class EngineConfig:
    robust_window: int = 50
    robust_threshold: float = 4.0
    contamination: float = 0.02
    seed: int = 42

    def __post_init__(self) -> None:
        if isinstance(self.robust_window, bool) or not isinstance(
            self.robust_window, int
        ):
            raise TypeError("robust_window must be an integer")
        if not 20 <= self.robust_window <= 10_000:
            raise ValueError("robust_window must be within [20, 10000]")
        if (
            isinstance(self.robust_threshold, bool)
            or not isinstance(self.robust_threshold, (int, float))
            or not math.isfinite(self.robust_threshold)
            or self.robust_threshold <= 0
        ):
            raise ValueError("robust_threshold must be finite and positive")
        if (
            isinstance(self.contamination, bool)
            or not isinstance(self.contamination, (int, float))
            or not math.isfinite(self.contamination)
            or not 0 < self.contamination <= 0.5
        ):
            raise ValueError("contamination must be finite and within (0, 0.5]")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")

    def kwargs(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class UnitInvariancePolicy:
    minimum_rows: int = 100
    maximum_rows: int = 100_000
    score_tolerance: float = 1e-10
    event_feature_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if isinstance(self.minimum_rows, bool) or not isinstance(
            self.minimum_rows, int
        ):
            raise TypeError("minimum_rows must be an integer")
        if isinstance(self.maximum_rows, bool) or not isinstance(
            self.maximum_rows, int
        ):
            raise TypeError("maximum_rows must be an integer")
        if self.minimum_rows < 2 or self.maximum_rows < self.minimum_rows:
            raise ValueError(
                "row bounds must satisfy 2 <= minimum_rows <= maximum_rows"
            )
        for name in ("score_tolerance", "event_feature_tolerance"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class UnitScenarioResult:
    scenario: str
    passed: bool
    finding_codes: tuple[str, ...]
    score_keys_match: bool
    max_absolute_score_delta: float | None
    flag_mismatches: int
    event_keys_match: bool
    event_contract_mismatches: int
    max_absolute_return_delta: float | None
    max_absolute_volume_change_delta: float | None


@dataclass(frozen=True, slots=True)
class UnitInvarianceReport:
    decision: str
    evidence_digest: str
    rows: int
    engine: EngineConfig
    policy: UnitInvariancePolicy
    scenarios: tuple[UnitScenarioResult, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate_frame(frame: pd.DataFrame, policy: UnitInvariancePolicy) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not policy.minimum_rows <= len(frame) <= policy.maximum_rows:
        raise ValueError(
            f"frame rows must be within [{policy.minimum_rows}, {policy.maximum_rows}]"
        )
    if not frame.columns.is_unique:
        raise ValueError("frame columns must be unique")
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"frame is missing required columns: {missing}")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError("frame index must be a DatetimeIndex")
    if (
        frame.index.hasnans
        or not frame.index.is_unique
        or not frame.index.is_monotonic_increasing
    ):
        raise ValueError(
            "frame index must contain valid, unique, monotonically increasing timestamps"
        )

    admitted = frame.loc[:, REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        if pd.api.types.is_bool_dtype(
            admitted[column]
        ) or not pd.api.types.is_numeric_dtype(admitted[column]):
            raise TypeError(f"column {column!r} must be numeric and non-boolean")
    values = admitted.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("frame values must be finite")
    if (values <= 0).any():
        raise ValueError("OHLCV values must be positive")
    return admitted.astype(float)


def _frame_digest(
    frame: pd.DataFrame,
    scenarios: Sequence[UnitScenario],
    config: EngineConfig,
    policy: UnitInvariancePolicy,
) -> str:
    digest = hashlib.sha256()
    metadata = {
        "schema_version": 1,
        "columns": list(REQUIRED_COLUMNS),
        "scenarios": [
            asdict(item) for item in sorted(scenarios, key=lambda item: item.name)
        ],
        "engine": asdict(config),
        "policy": asdict(policy),
    }
    digest.update(
        json.dumps(
            metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    )
    digest.update(np.asarray(frame.index.asi8, dtype="<i8").tobytes())
    digest.update(np.asarray(frame.to_numpy(dtype=float), dtype="<f8").tobytes())
    return digest.hexdigest()


def _maximum_delta(left: Sequence[float], right: Sequence[float]) -> float | None:
    if not left:
        return None
    return float(np.max(np.abs(np.asarray(left) - np.asarray(right))))


def _compare_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    scenario: UnitScenario,
    policy: UnitInvariancePolicy,
) -> UnitScenarioResult:
    baseline_scores = baseline["scores"]
    candidate_scores = candidate["scores"]
    score_keys_match = set(baseline_scores) == set(candidate_scores)
    common_scores = sorted(set(baseline_scores).intersection(candidate_scores))
    max_score_delta = _maximum_delta(
        [float(baseline_scores[key]) for key in common_scores],
        [float(candidate_scores[key]) for key in common_scores],
    )

    baseline_flags = baseline["flags"]
    candidate_flags = candidate["flags"]
    flag_keys = set(baseline_flags).union(candidate_flags)
    flag_mismatches = sum(
        baseline_flags.get(key) != candidate_flags.get(key) for key in flag_keys
    )

    baseline_events = {event["timestamp"]: event for event in baseline["events"]}
    candidate_events = {event["timestamp"]: event for event in candidate["events"]}
    event_keys_match = set(baseline_events) == set(candidate_events)
    event_contract_mismatches = 0
    return_left: list[float] = []
    return_right: list[float] = []
    volume_left: list[float] = []
    volume_right: list[float] = []
    for timestamp in sorted(set(baseline_events).intersection(candidate_events)):
        left = baseline_events[timestamp]
        right = candidate_events[timestamp]
        if (
            left["severity"] != right["severity"]
            or left["regime"] != right["regime"]
            or tuple(left["reasons"]) != tuple(right["reasons"])
        ):
            event_contract_mismatches += 1
        return_left.append(float(left["return_1"]))
        return_right.append(float(right["return_1"]))
        volume_left.append(float(left["volume_change"]))
        volume_right.append(float(right["volume_change"]))

    max_return_delta = _maximum_delta(return_left, return_right)
    max_volume_delta = _maximum_delta(volume_left, volume_right)
    findings: list[str] = []
    if not score_keys_match:
        findings.append("score_index_changed")
    if max_score_delta is None or max_score_delta > policy.score_tolerance:
        findings.append("score_value_changed")
    if flag_mismatches:
        findings.append("flag_decision_changed")
    if not event_keys_match:
        findings.append("event_set_changed")
    if event_contract_mismatches:
        findings.append("event_explanation_changed")
    if (
        max_return_delta is not None
        and max_return_delta > policy.event_feature_tolerance
    ) or (
        max_volume_delta is not None
        and max_volume_delta > policy.event_feature_tolerance
    ):
        findings.append("event_feature_changed")

    return UnitScenarioResult(
        scenario=scenario.name,
        passed=not findings,
        finding_codes=tuple(findings),
        score_keys_match=score_keys_match,
        max_absolute_score_delta=max_score_delta,
        flag_mismatches=flag_mismatches,
        event_keys_match=event_keys_match,
        event_contract_mismatches=event_contract_mismatches,
        max_absolute_return_delta=max_return_delta,
        max_absolute_volume_change_delta=max_volume_delta,
    )


def audit_unit_invariance(
    frame: pd.DataFrame,
    scenarios: Sequence[UnitScenario] = DEFAULT_SCENARIOS,
    *,
    config: EngineConfig | None = None,
    policy: UnitInvariancePolicy | None = None,
    analyzer: MarketAnalyzer | None = None,
) -> UnitInvarianceReport:
    """Run the detector under equivalent price/volume unit transformations."""

    config = config or EngineConfig()
    policy = policy or UnitInvariancePolicy()
    if not isinstance(config, EngineConfig):
        raise TypeError("config must be an EngineConfig")
    if not isinstance(policy, UnitInvariancePolicy):
        raise TypeError("policy must be a UnitInvariancePolicy")
    if isinstance(scenarios, (str, bytes)) or not scenarios:
        raise ValueError("scenarios must be a non-empty sequence")
    if len(scenarios) > 32:
        raise ValueError("at most 32 scenarios are allowed")
    if any(not isinstance(item, UnitScenario) for item in scenarios):
        raise TypeError("every scenario must be a UnitScenario")
    names = [item.name for item in scenarios]
    if len(names) != len(set(names)):
        raise ValueError("scenario names must be unique")

    admitted = _validate_frame(frame, policy)
    normalized_scenarios = tuple(sorted(scenarios, key=lambda item: item.name))
    engine = analyzer or MarketAnomalyEngine()
    baseline = engine.analyze(admitted.copy(), **config.kwargs())
    results: list[UnitScenarioResult] = []
    for scenario in normalized_scenarios:
        transformed = admitted.copy()
        price_columns = ["open", "high", "low", "close"]
        transformed.loc[:, price_columns] = (
            transformed.loc[:, price_columns] * scenario.price_multiplier
        )
        transformed.loc[:, "volume"] = (
            transformed.loc[:, "volume"] * scenario.volume_multiplier
        )
        candidate = engine.analyze(
            _validate_frame(transformed, policy), **config.kwargs()
        )
        results.append(_compare_results(baseline, candidate, scenario, policy))

    return UnitInvarianceReport(
        decision="pass" if all(item.passed for item in results) else "fail",
        evidence_digest=_frame_digest(admitted, normalized_scenarios, config, policy),
        rows=len(admitted),
        engine=config,
        policy=policy,
        scenarios=tuple(results),
    )


def _load_csv(path: Path) -> pd.DataFrame:
    if path.is_symlink():
        raise ValueError(f"refusing symlink input: {path}")
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ValueError(f"input exceeds {MAX_INPUT_BYTES} bytes: {path}")
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise ValueError("CSV must contain a timestamp column")
    timestamp = pd.to_datetime(frame.pop("timestamp"), utc=True, errors="raise")
    frame.index = pd.DatetimeIndex(timestamp)
    return frame


def _write_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit market anomaly unit invariance")
    parser.add_argument("input", type=Path, help="timestamped OHLCV CSV")
    parser.add_argument(
        "--output", type=Path, required=True, help="atomic JSON report path"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--score-tolerance", type=float, default=1e-10)
    parser.add_argument("--event-feature-tolerance", type=float, default=1e-10)
    args = parser.parse_args(argv)

    try:
        report = audit_unit_invariance(
            _load_csv(args.input),
            config=EngineConfig(seed=args.seed),
            policy=UnitInvariancePolicy(
                score_tolerance=args.score_tolerance,
                event_feature_tolerance=args.event_feature_tolerance,
            ),
        )
        _write_atomic(args.output, report.to_dict())
    except (OSError, TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        print(f"unit invariance audit failed: {exc}", file=sys.stderr)
        return 3
    return 0 if report.decision == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())

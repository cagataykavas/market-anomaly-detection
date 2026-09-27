from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from .detectors import build_market_features
from .regimes import regime_adjusted_threshold
from .synthetic import synthetic_ohlcv

FEATURE_COLUMNS = (
    "return_1",
    "abs_return",
    "realized_vol_20",
    "volume_change",
    "range_pct",
)
REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")
TIMEZONE_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})\Z", re.IGNORECASE)
MAX_INPUT_BYTES = 32_000_000


class WalkForwardInputError(ValueError):
    """Invalid configuration or market evidence with a stable reason code."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class WalkForwardConfig:
    min_history: int = 120
    training_window: int = 252
    refit_every: int = 20
    robust_window: int = 60
    regime_lookback: int = 120
    robust_threshold: float = 4.0
    contamination: float = 0.02
    base_threshold: float = 0.62
    high_regime_multiplier: float = 1.15
    low_regime_multiplier: float = 0.90
    n_estimators: int = 200
    seed: int = 42
    max_rows: int = 20_000

    def __post_init__(self) -> None:
        for name, value, lower, upper in (
            ("min_history", self.min_history, 40, 10_000),
            ("training_window", self.training_window, 40, 20_000),
            ("refit_every", self.refit_every, 1, 2_000),
            ("robust_window", self.robust_window, 10, 5_000),
            ("regime_lookback", self.regime_lookback, 20, 10_000),
            ("n_estimators", self.n_estimators, 10, 1_000),
            ("max_rows", self.max_rows, 100, 100_000),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or not lower <= value <= upper:
                raise ValueError(f"{name} must be an integer between {lower} and {upper}")
        if self.training_window < self.min_history:
            raise ValueError("training_window must cover min_history")
        if self.refit_every > self.training_window:
            raise ValueError("refit_every cannot exceed training_window")
        for name, value, lower, upper in (
            ("robust_threshold", self.robust_threshold, 0.1, 100.0),
            ("contamination", self.contamination, 1e-6, 0.499999),
            ("base_threshold", self.base_threshold, 1e-6, 1.0),
            ("high_regime_multiplier", self.high_regime_multiplier, 0.1, 10.0),
            ("low_regime_multiplier", self.low_regime_multiplier, 0.1, 10.0),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not lower <= float(value) <= upper
            ):
                raise ValueError(f"{name} must be finite and between {lower} and {upper}")
        if self.base_threshold * self.high_regime_multiplier > 1.0:
            raise ValueError("high-regime threshold cannot exceed one")
        if self.base_threshold * self.low_regime_multiplier > 1.0:
            raise ValueError("low-regime threshold cannot exceed one")
        if self.high_regime_multiplier < 1.0 or self.low_regime_multiplier > 1.0:
            raise ValueError("regime multipliers must widen thresholds in the declared direction")
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or not 0 <= self.seed < 2**32
        ):
            raise ValueError("seed must be a 32-bit unsigned integer")


@dataclass(frozen=True)
class WalkForwardRecord:
    timestamp: str
    fit_started_at: str
    fit_ended_at: str
    history_ended_at: str
    training_rows: int
    isolation_percentile: float
    return_robust_score: float
    volume_robust_score: float
    score: float
    threshold: float
    regime: str
    flagged: bool
    reasons: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["reasons"] = list(self.reasons)
        return result


@dataclass(frozen=True)
class WalkForwardReport:
    schema_version: int
    input_rows: int
    feature_rows: int
    evaluated_rows: int
    refit_count: int
    anomaly_count: int
    anomaly_rate: float
    config: dict[str, object]
    records: tuple[WalkForwardRecord, ...]
    input_sha256: str
    config_sha256: str
    evidence_sha256: str

    def body(self) -> dict[str, object]:
        result = asdict(self)
        result.pop("evidence_sha256")
        result["records"] = [record.as_dict() for record in self.records]
        return result

    def as_dict(self) -> dict[str, object]:
        return {**self.body(), "evidence_sha256": self.evidence_sha256}


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError) as exc:
        raise WalkForwardInputError("NON_CANONICAL_EVIDENCE") from exc


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _timestamp(value: object) -> str:
    return pd.Timestamp(value).isoformat()


def _validate_frame(frame: pd.DataFrame, config: WalkForwardConfig) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise WalkForwardInputError("FRAME_NOT_DATAFRAME")
    missing = [name for name in REQUIRED_COLUMNS if name not in frame.columns]
    if missing:
        raise WalkForwardInputError("MISSING_OHLCV_COLUMN")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise WalkForwardInputError("INDEX_NOT_DATETIME")
    if frame.index.tz is None:
        raise WalkForwardInputError("INDEX_NOT_TIMEZONE_AWARE")
    if not frame.index.is_monotonic_increasing:
        raise WalkForwardInputError("TIMESTAMPS_NOT_SORTED")
    if not frame.index.is_unique:
        raise WalkForwardInputError("DUPLICATE_TIMESTAMP")
    if len(frame) > config.max_rows:
        raise WalkForwardInputError("ROW_BUDGET_EXCEEDED")
    if len(frame) < config.min_history + 21:
        raise WalkForwardInputError("INSUFFICIENT_ROWS")

    values = frame.loc[:, REQUIRED_COLUMNS].copy()
    if any(pd.api.types.is_bool_dtype(values[name]) for name in REQUIRED_COLUMNS):
        raise WalkForwardInputError("NON_NUMERIC_OHLCV")
    try:
        values = values.astype(float)
    except (TypeError, ValueError) as exc:
        raise WalkForwardInputError("NON_NUMERIC_OHLCV") from exc
    numeric = values.to_numpy()
    if not np.isfinite(numeric).all():
        raise WalkForwardInputError("NON_FINITE_OHLCV")
    if (values[["open", "high", "low", "close", "volume"]] <= 0).any().any():
        raise WalkForwardInputError("NON_POSITIVE_OHLCV")
    if (values["high"] < values[["open", "close", "low"]].max(axis=1)).any():
        raise WalkForwardInputError("INVALID_HIGH_PRICE")
    if (values["low"] > values[["open", "close", "high"]].min(axis=1)).any():
        raise WalkForwardInputError("INVALID_LOW_PRICE")
    return values


def _causal_robust_score(series: pd.Series, position: int, window: int) -> float:
    history = series.iloc[max(0, position - window) : position]
    current = float(series.iloc[position])
    median = float(history.median())
    mad = float((history - median).abs().median())
    if not math.isfinite(median) or not math.isfinite(mad):
        raise WalkForwardInputError("INVALID_ROBUST_HISTORY")
    if mad <= 1e-15:
        return 0.0 if math.isclose(current, median, rel_tol=0.0, abs_tol=1e-15) else 1e6
    return abs(current - median) / (1.4826 * mad)


def _causal_regime(realized_volatility: pd.Series, position: int, lookback: int) -> str:
    history = realized_volatility.iloc[max(0, position - lookback) : position]
    current = float(realized_volatility.iloc[position])
    if len(history) < 20:
        return "normal"
    low = float(history.quantile(0.33))
    high = float(history.quantile(0.67))
    if current <= low:
        return "low"
    if current >= high:
        return "high"
    return "normal"


def _input_digest(frame: pd.DataFrame) -> str:
    rows = [
        {
            "timestamp": _timestamp(timestamp),
            **{name: float(row[name]) for name in REQUIRED_COLUMNS},
        }
        for timestamp, row in frame.iterrows()
    ]
    return _digest(rows)


def walk_forward_evaluate(
    frame: pd.DataFrame,
    config: WalkForwardConfig | None = None,
) -> WalkForwardReport:
    """Score each market row using only information available at that timestamp."""

    policy = config or WalkForwardConfig()
    values = _validate_frame(frame, policy)
    features = build_market_features(values).loc[:, FEATURE_COLUMNS].dropna()
    if len(features) <= policy.min_history:
        raise WalkForwardInputError("INSUFFICIENT_FEATURE_HISTORY")

    model: IsolationForest | None = None
    training_scores: np.ndarray | None = None
    fit_started_at = ""
    fit_ended_at = ""
    last_fit_position = -policy.refit_every
    refit_count = 0
    records: list[WalkForwardRecord] = []

    for position in range(policy.min_history, len(features)):
        if model is None or position - last_fit_position >= policy.refit_every:
            training = features.iloc[max(0, position - policy.training_window) : position]
            if len(training) < policy.min_history:
                raise WalkForwardInputError("INSUFFICIENT_TRAINING_WINDOW")
            model = IsolationForest(
                n_estimators=policy.n_estimators,
                contamination=policy.contamination,
                random_state=policy.seed,
                n_jobs=1,
            )
            model.fit(training)
            training_scores = -model.score_samples(training)
            if not np.isfinite(training_scores).all():
                raise WalkForwardInputError("NON_FINITE_MODEL_SCORE")
            fit_started_at = _timestamp(training.index[0])
            fit_ended_at = _timestamp(training.index[-1])
            last_fit_position = position
            refit_count += 1

        assert model is not None and training_scores is not None
        timestamp = features.index[position]
        current = features.iloc[[position]]
        raw_score = float(-model.score_samples(current)[0])
        if not math.isfinite(raw_score):
            raise WalkForwardInputError("NON_FINITE_MODEL_SCORE")
        isolation_percentile = float(
            (np.count_nonzero(training_scores <= raw_score) + 1) / (len(training_scores) + 1)
        )
        return_score = _causal_robust_score(features["return_1"], position, policy.robust_window)
        volume_score = _causal_robust_score(
            features["volume_change"], position, policy.robust_window
        )
        regime = _causal_regime(features["realized_vol_20"], position, policy.regime_lookback)
        return_normalized = min(return_score / policy.robust_threshold, 2.0) / 2.0
        volume_normalized = min(volume_score / policy.robust_threshold, 2.0) / 2.0
        score = float(
            0.45 * isolation_percentile + 0.35 * return_normalized + 0.20 * volume_normalized
        )
        threshold = float(
            regime_adjusted_threshold(
                policy.base_threshold,
                regime,
                high_multiplier=policy.high_regime_multiplier,
                low_multiplier=policy.low_regime_multiplier,
            )
        )
        flagged = score >= threshold
        reasons: list[str] = []
        if return_score >= policy.robust_threshold:
            reasons.append("extreme_causal_return_score")
        if volume_score >= policy.robust_threshold:
            reasons.append("extreme_causal_volume_score")
        if isolation_percentile >= 1.0 - policy.contamination:
            reasons.append("causal_isolation_outlier")
        if regime == "high":
            reasons.append("high_volatility_threshold")
        if flagged and not reasons:
            reasons.append("combined_causal_score")

        history_ended_at = _timestamp(features.index[position - 1])
        if not (
            pd.Timestamp(fit_ended_at) < timestamp and pd.Timestamp(history_ended_at) < timestamp
        ):
            raise WalkForwardInputError("NON_CAUSAL_WINDOW")
        records.append(
            WalkForwardRecord(
                timestamp=_timestamp(timestamp),
                fit_started_at=fit_started_at,
                fit_ended_at=fit_ended_at,
                history_ended_at=history_ended_at,
                training_rows=len(training_scores),
                isolation_percentile=isolation_percentile,
                return_robust_score=float(return_score),
                volume_robust_score=float(volume_score),
                score=score,
                threshold=threshold,
                regime=regime,
                flagged=flagged,
                reasons=tuple(reasons),
            )
        )

    anomaly_count = sum(record.flagged for record in records)
    config_body = asdict(policy)
    report = WalkForwardReport(
        schema_version=1,
        input_rows=len(values),
        feature_rows=len(features),
        evaluated_rows=len(records),
        refit_count=refit_count,
        anomaly_count=anomaly_count,
        anomaly_rate=anomaly_count / len(records) if records else 0.0,
        config=config_body,
        records=tuple(records),
        input_sha256=_input_digest(values),
        config_sha256=_digest(config_body),
        evidence_sha256="",
    )
    return replace(report, evidence_sha256=_digest(report.body()))


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            json.dump(payload, handle, allow_nan=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = handle.name
        os.replace(temporary, path)
    except OSError as exc:
        if temporary:
            Path(temporary).unlink(missing_ok=True)
        raise WalkForwardInputError("OUTPUT_WRITE_FAILED") from exc


def _load_frame(path: Path) -> pd.DataFrame:
    try:
        if path.stat().st_size > MAX_INPUT_BYTES:
            raise WalkForwardInputError("INPUT_BYTE_BUDGET_EXCEEDED")
        frame = pd.read_csv(path)
    except WalkForwardInputError:
        raise
    except (OSError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        raise WalkForwardInputError("INPUT_CSV_UNREADABLE") from exc
    if "timestamp" not in frame:
        raise WalkForwardInputError("MISSING_TIMESTAMP_COLUMN")
    raw_timestamps = frame["timestamp"].astype(str)
    if not raw_timestamps.map(lambda value: bool(TIMEZONE_SUFFIX.search(value))).all():
        raise WalkForwardInputError("TIMESTAMP_NOT_TIMEZONE_AWARE")
    try:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="raise")
    except (ValueError, TypeError) as exc:
        raise WalkForwardInputError("INVALID_TIMESTAMP") from exc
    return frame.set_index("timestamp")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Causal walk-forward market anomaly evaluation")
    parser.add_argument("--input", type=Path, help="CSV with timezone-aware OHLCV timestamps")
    parser.add_argument("--rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("artifacts/walk-forward.json"))
    parser.add_argument("--min-history", type=int, default=120)
    parser.add_argument("--training-window", type=int, default=252)
    parser.add_argument("--refit-every", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=200)
    args = parser.parse_args(argv)
    try:
        if args.input:
            frame = _load_frame(args.input)
        else:
            frame, _ = synthetic_ohlcv(rows=args.rows, seed=args.seed)
            frame.index = frame.index.tz_localize("UTC")
        config = WalkForwardConfig(
            min_history=args.min_history,
            training_window=args.training_window,
            refit_every=args.refit_every,
            n_estimators=args.n_estimators,
            seed=args.seed,
        )
        report = walk_forward_evaluate(frame, config)
        _write_json_atomic(args.output, report.as_dict())
    except (WalkForwardInputError, ValueError) as exc:
        code = exc.code if isinstance(exc, WalkForwardInputError) else "INVALID_CONFIGURATION"
        print(json.dumps({"status": "malformed", "reason_code": code}, sort_keys=True))
        return 3
    print(
        json.dumps(
            {
                "status": "completed",
                "evaluated_rows": report.evaluated_rows,
                "anomaly_count": report.anomaly_count,
                "refit_count": report.refit_count,
                "evidence_sha256": report.evidence_sha256,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

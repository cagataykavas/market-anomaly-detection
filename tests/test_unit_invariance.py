import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.synthetic import synthetic_ohlcv
from src.unit_invariance import (
    EngineConfig,
    UnitInvariancePolicy,
    UnitScenario,
    audit_unit_invariance,
    main,
)


class ScaleSensitiveAnalyzer:
    def analyze(self, frame: pd.DataFrame, **_: object) -> dict[str, object]:
        score = float(frame["close"].iloc[0] / 100.0)
        flagged = score > 1.0
        timestamp = str(frame.index[0])
        return {
            "scores": {timestamp: score},
            "flags": {timestamp: flagged},
            "events": [
                {
                    "timestamp": timestamp,
                    "score": score,
                    "severity": "high" if flagged else "low",
                    "regime": "normal",
                    "return_1": score,
                    "volume_change": 0.0,
                    "reasons": ["scale-sensitive" if flagged else "baseline"],
                }
            ],
        }


def small_frame(rows: int = 20) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=rows, freq="h", tz="UTC")
    close = np.linspace(100.0, 110.0, rows)
    return pd.DataFrame(
        {
            "open": close * 0.999,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "volume": np.linspace(1_000.0, 1_100.0, rows),
        },
        index=index,
    )


def test_real_engine_preserves_scores_flags_and_explanations_across_units() -> None:
    frame, _ = synthetic_ohlcv(rows=260, seed=17)
    scenarios = (
        UnitScenario("cents", price_multiplier=100.0),
        UnitScenario("lots", volume_multiplier=0.01),
    )

    report = audit_unit_invariance(frame, scenarios, config=EngineConfig(seed=17))

    assert report.decision == "pass"
    assert all(result.passed for result in report.scenarios)
    assert all(result.flag_mismatches == 0 for result in report.scenarios)
    assert len(report.evidence_digest) == 64


def test_scale_sensitive_analyzer_is_rejected() -> None:
    policy = UnitInvariancePolicy(minimum_rows=2)

    report = audit_unit_invariance(
        small_frame(),
        (UnitScenario("cents", price_multiplier=100.0),),
        policy=policy,
        analyzer=ScaleSensitiveAnalyzer(),
    )

    result = report.scenarios[0]
    assert report.decision == "fail"
    assert "score_value_changed" in result.finding_codes
    assert "event_explanation_changed" in result.finding_codes
    assert "event_feature_changed" in result.finding_codes


def test_digest_is_invariant_to_scenario_order() -> None:
    frame = small_frame()
    policy = UnitInvariancePolicy(minimum_rows=2)
    scenarios = (
        UnitScenario("cents", price_multiplier=100.0),
        UnitScenario("lots", volume_multiplier=0.01),
    )

    forward = audit_unit_invariance(
        frame, scenarios, policy=policy, analyzer=ScaleSensitiveAnalyzer()
    )
    reverse = audit_unit_invariance(
        frame,
        tuple(reversed(scenarios)),
        policy=policy,
        analyzer=ScaleSensitiveAnalyzer(),
    )

    assert forward.evidence_digest == reverse.evidence_digest


def test_digest_binds_engine_configuration() -> None:
    frame = small_frame()
    policy = UnitInvariancePolicy(minimum_rows=2)
    scenario = (UnitScenario("cents", price_multiplier=100.0),)

    first = audit_unit_invariance(
        frame,
        scenario,
        config=EngineConfig(seed=1),
        policy=policy,
        analyzer=ScaleSensitiveAnalyzer(),
    )
    second = audit_unit_invariance(
        frame,
        scenario,
        config=EngineConfig(seed=2),
        policy=policy,
        analyzer=ScaleSensitiveAnalyzer(),
    )

    assert first.evidence_digest != second.evidence_digest


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda frame: frame.drop(columns="volume"), "missing required columns"),
        (lambda frame: frame.assign(close=np.nan), "finite"),
        (lambda frame: frame.assign(volume=0.0), "positive"),
        (lambda frame: frame.iloc[::-1], "monotonically increasing"),
        (
            lambda frame: frame.set_axis(
                ["open", "high", "low", "close", "close"], axis="columns"
            ),
            "columns must be unique",
        ),
    ],
)
def test_malformed_frames_fail_closed(mutator, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        audit_unit_invariance(
            mutator(small_frame()),
            (UnitScenario("cents", price_multiplier=100.0),),
            policy=UnitInvariancePolicy(minimum_rows=2),
            analyzer=ScaleSensitiveAnalyzer(),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "price_multiplier": 1.0},
        {"name": "bad", "price_multiplier": 0.0},
        {"name": "bad", "volume_multiplier": float("nan")},
        {"name": "bad", "price_multiplier": True},
    ],
)
def test_invalid_scenarios_fail_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        UnitScenario(**kwargs)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: EngineConfig(robust_window=1),
        lambda: EngineConfig(contamination=0.0),
        lambda: UnitInvariancePolicy(minimum_rows=1),
        lambda: UnitInvariancePolicy(score_tolerance=float("nan")),
    ],
)
def test_invalid_policy_and_engine_config_fail_closed(factory) -> None:
    with pytest.raises((TypeError, ValueError)):
        factory()


def test_invalid_runtime_configuration_types_fail_closed() -> None:
    with pytest.raises(TypeError, match="config must be"):
        audit_unit_invariance(small_frame(), config="not-a-config")
    with pytest.raises(TypeError, match="policy must be"):
        audit_unit_invariance(small_frame(), policy="not-a-policy")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    output = frame.copy()
    output.insert(0, "timestamp", output.index)
    output.to_csv(path, index=False)


def test_cli_writes_passing_real_engine_report(tmp_path: Path) -> None:
    input_path = tmp_path / "market.csv"
    output_path = tmp_path / "report.json"
    frame, _ = synthetic_ohlcv(rows=220, seed=42)
    write_csv(input_path, frame)

    exit_code = main([str(input_path), "--output", str(output_path)])

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["decision"] == "pass"
    assert len(payload["scenarios"]) == 4


def test_cli_rejects_missing_timestamp_without_output(tmp_path: Path) -> None:
    input_path = tmp_path / "market.csv"
    output_path = tmp_path / "report.json"
    small_frame().to_csv(input_path, index=False)

    exit_code = main([str(input_path), "--output", str(output_path)])

    assert exit_code == 3
    assert not output_path.exists()

from __future__ import annotations

import json

import pytest

from src.engine import MarketAnomalyEngine
from src.explanation_audit import (
    ExplanationAuditError,
    ExplanationAuditPolicy,
    audit_explanations,
    main,
)
from src.synthetic import synthetic_ohlcv


def _event(
    *,
    timestamp: str = "2026-01-01T00:00:00Z",
    score: float = 0.75,
    threshold: float = 0.62,
    components: tuple[tuple[str, float, bool], ...] = (
        ("isolation_forest", 0.45, True),
        ("robust_return", 0.25, True),
        ("robust_volume", 0.05, False),
    ),
) -> dict[str, object]:
    return {
        "timestamp": timestamp,
        "score": score,
        "threshold": threshold,
        "explanation": {
            "method": "weighted-signal-decomposition-v1",
            "baseline": 0.0,
            "components": [
                {"signal": signal, "contribution": contribution, "reported": reported}
                for signal, contribution, reported in components
            ],
            "reported_components": [signal for signal, _, reported in components if reported],
        },
    }


def test_engine_explanations_pass_default_fidelity_policy():
    frame, _ = synthetic_ohlcv(rows=600, seed=42)
    artifact = MarketAnomalyEngine().analyze(frame, seed=42)

    report = audit_explanations(artifact)

    assert artifact["anomaly_count"] > 0
    assert report["passed"] is True
    assert report["summary"]["event_count"] == artifact["anomaly_count"]
    assert report["summary"]["max_additivity_error"] <= 1e-9
    assert report["summary"]["min_positive_coverage"] >= 0.90


def test_audit_detects_score_tampering_without_exposing_timestamp():
    artifact = {"events": [_event(score=0.80)]}

    report = audit_explanations(artifact)

    assert report["passed"] is False
    assert {finding["code"] for finding in report["findings"]} == {"additivity_mismatch"}
    assert "2026-01-01" not in json.dumps(report)


def test_audit_detects_omitted_material_component_and_low_coverage():
    artifact = {
        "events": [
            _event(
                components=(
                    ("isolation_forest", 0.40, True),
                    ("robust_return", 0.30, False),
                    ("robust_volume", 0.05, True),
                )
            )
        ]
    }

    report = audit_explanations(artifact, ExplanationAuditPolicy(material_component_share=0.20))

    assert report["passed"] is False
    assert {finding["code"] for finding in report["findings"]} == {
        "insufficient_positive_coverage",
        "material_component_omitted",
    }


def test_audit_detects_inconsistent_and_nonpositive_reported_components():
    inconsistent = _event(
        score=0.75,
        components=(
            ("isolation_forest", 0.80, True),
            ("robust_return", -0.05, True),
        ),
    )
    inconsistent["explanation"]["reported_components"] = ["isolation_forest"]
    nonpositive = _event(
        timestamp="2026-01-02T00:00:00Z",
        score=0.75,
        components=(
            ("isolation_forest", 0.80, True),
            ("robust_return", -0.05, True),
        ),
    )

    report = audit_explanations({"events": [inconsistent, nonpositive]})

    assert report["passed"] is False
    assert {finding["code"] for finding in report["findings"]} == {
        "inconsistent_reported_components",
        "nonpositive_component_reported",
    }


def test_audit_rejects_duplicate_component_names():
    artifact = {
        "events": [
            _event(
                components=(
                    ("isolation_forest", 0.40, True),
                    ("isolation_forest", 0.35, True),
                )
            )
        ]
    }

    with pytest.raises(ExplanationAuditError, match="duplicate signal"):
        audit_explanations(artifact)


def test_cli_writes_a_failure_report_and_returns_nonzero(tmp_path):
    artifact_path = tmp_path / "anomalies.json"
    report_path = tmp_path / "audit.json"
    artifact_path.write_text(json.dumps({"events": [_event(score=0.80)]}), encoding="utf-8")

    exit_code = main([str(artifact_path), "--output", str(report_path)])

    assert exit_code == 1
    assert json.loads(report_path.read_text(encoding="utf-8"))["passed"] is False

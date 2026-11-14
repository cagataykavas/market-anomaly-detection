from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from src.explanation_drift import (
    ExplanationDriftInputError,
    ExplanationEvent,
    evaluate_explanation_drift,
    jensen_shannon_divergence,
    load_explanation_events,
    main,
)


def _events(reason: str, count: int, *, regime: str = "normal") -> list[ExplanationEvent]:
    return [ExplanationEvent(regime, (reason,)) for _ in range(count)]


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_identical_explanations_pass_with_zero_uncertainty() -> None:
    baseline = _events("extreme volume change", 20)
    candidate = _events("extreme volume change", 20)

    report = evaluate_explanation_drift(
        baseline,
        candidate,
        minimum_events=20,
        maximum_jsd=0.01,
        bootstrap_samples=100,
        seed=7,
    )

    assert report.ok
    assert report.regimes[0].observed_jsd == 0.0
    assert report.regimes[0].bootstrap_upper_jsd == 0.0


def test_disjoint_reason_distributions_fail() -> None:
    report = evaluate_explanation_drift(
        _events("extreme volume change", 20),
        _events("multivariate Isolation Forest outlier", 20),
        minimum_events=20,
        maximum_jsd=0.10,
        bootstrap_samples=100,
    )

    assert not report.ok
    assert report.regimes[0].observed_jsd == 1.0
    assert report.regimes[0].bootstrap_upper_jsd == 1.0
    assert report.regimes[0].issues[0].code == "regime.drift_exceeds_threshold"
    assert report.regimes[0].largest_reason_shifts[0].delta in {-1.0, 1.0}


def test_fails_closed_when_a_regime_lacks_support() -> None:
    report = evaluate_explanation_drift(
        _events("reason", 20, regime="normal"),
        _events("reason", 5, regime="normal"),
        minimum_events=20,
        bootstrap_samples=100,
    )

    assert not report.ok
    assert report.regimes[0].observed_jsd is None
    assert report.regimes[0].issues[0].code == "regime.insufficient_support"


def test_evaluates_each_regime_independently() -> None:
    baseline = _events("stable", 10, regime="low") + _events("old", 10, regime="high")
    candidate = _events("stable", 10, regime="low") + _events("new", 10, regime="high")

    report = evaluate_explanation_drift(
        baseline,
        candidate,
        minimum_events=10,
        maximum_jsd=0.20,
        bootstrap_samples=100,
    )

    by_regime = {item.regime: item for item in report.regimes}
    assert by_regime["low"].ok
    assert not by_regime["high"].ok


def test_jensen_shannon_is_symmetric() -> None:
    from collections import Counter

    left = Counter({"a": 3, "b": 1})
    right = Counter({"a": 1, "c": 3})

    assert jensen_shannon_divergence(left, right) == pytest.approx(
        jensen_shannon_divergence(right, left)
    )


def test_bootstrap_is_reproducible_for_fixed_seed() -> None:
    baseline = [
        ExplanationEvent("normal", ("a",)) if index % 2 else ExplanationEvent("normal", ("b",))
        for index in range(20)
    ]
    candidate = [
        ExplanationEvent("normal", ("a",)) if index % 3 else ExplanationEvent("normal", ("b",))
        for index in range(20)
    ]

    first = evaluate_explanation_drift(
        baseline, candidate, minimum_events=20, maximum_jsd=1.0, bootstrap_samples=200, seed=9
    )
    second = evaluate_explanation_drift(
        baseline, candidate, minimum_events=20, maximum_jsd=1.0, bootstrap_samples=200, seed=9
    )

    assert first.as_dict() == second.as_dict()


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"regime":"low","regime":"high","reasons":["x"]}\n', encoding="utf-8")

    with pytest.raises(ExplanationDriftInputError, match="duplicate JSON key"):
        load_explanation_events(path)


def test_loader_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.jsonl"
    _write(target, [{"regime": "normal", "reasons": ["x"]}])
    link = tmp_path / "events.jsonl"
    os.symlink(target, link)

    with pytest.raises(ExplanationDriftInputError, match="non-symlink"):
        load_explanation_events(link)


def test_cli_writes_failure_report_and_returns_two(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    output = tmp_path / "report.json"
    _write(baseline, [{"regime": "normal", "reasons": ["old"]}] * 2)
    _write(candidate, [{"regime": "normal", "reasons": ["new"]}] * 2)

    exit_code = main(
        [
            "--baseline",
            str(baseline),
            "--candidate",
            str(candidate),
            "--minimum-events",
            "2",
            "--bootstrap-samples",
            "100",
            "--maximum-jsd",
            "0.2",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 2
    report = json.loads(output.read_text(encoding="utf-8"))
    assert not report["ok"]
    assert report["regimes"][0]["issues"][0]["code"] == "regime.drift_exceeds_threshold"

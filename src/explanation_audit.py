from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MAX_INPUT_BYTES = 8 * 1024 * 1024


class ExplanationAuditError(ValueError):
    """Raised when an anomaly artifact does not satisfy the audit schema."""


@dataclass(frozen=True)
class ExplanationAuditPolicy:
    additivity_atol: float = 1e-9
    material_component_share: float = 0.10
    min_positive_coverage: float = 0.90
    min_material_recall: float = 1.0
    min_events: int = 1

    def __post_init__(self) -> None:
        if self.additivity_atol < 0.0:
            raise ValueError("additivity_atol must be non-negative")
        for name in ("material_component_share", "min_positive_coverage", "min_material_recall"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.min_events < 1:
            raise ValueError("min_events must be at least 1")


@dataclass(frozen=True, order=True)
class AuditFinding:
    event_ref: str
    code: str
    detail: str


def _require_mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExplanationAuditError(f"{context} must be an object")
    return value


def _require_sequence(value: Any, context: str) -> Sequence[Any]:
    if not isinstance(value, list):
        raise ExplanationAuditError(f"{context} must be an array")
    return value


def _finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ExplanationAuditError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ExplanationAuditError(f"{context} must be finite")
    return result


def _event_ref(event: Mapping[str, Any], index: int) -> str:
    identity = str(event.get("timestamp", f"event-{index}"))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]


def _finding(event_ref: str, code: str, detail: str) -> AuditFinding:
    return AuditFinding(event_ref=event_ref, code=code, detail=detail)


def audit_explanations(
    artifact: Mapping[str, Any],
    policy: ExplanationAuditPolicy | None = None,
) -> dict[str, Any]:
    """Audit additive anomaly explanations without exposing raw event identifiers."""
    policy = policy or ExplanationAuditPolicy()
    events = _require_sequence(artifact.get("events"), "events")
    findings: list[AuditFinding] = []
    additivity_errors: list[float] = []
    coverages: list[float] = []
    material_recalls: list[float] = []
    failed_event_refs: set[str] = set()

    if len(events) < policy.min_events:
        findings.append(
            _finding(
                "artifact",
                "insufficient_events",
                f"found {len(events)} event(s); policy requires {policy.min_events}",
            )
        )

    for index, raw_event in enumerate(events):
        event = _require_mapping(raw_event, f"events[{index}]")
        event_ref = _event_ref(event, index)
        score = _finite_float(event.get("score"), f"events[{index}].score")
        threshold = _finite_float(event.get("threshold"), f"events[{index}].threshold")
        explanation = _require_mapping(event.get("explanation"), f"events[{index}].explanation")
        if explanation.get("method") != "weighted-signal-decomposition-v1":
            raise ExplanationAuditError(
                f"events[{index}].explanation.method must be weighted-signal-decomposition-v1"
            )
        baseline = _finite_float(explanation.get("baseline"), f"events[{index}].explanation.baseline")
        raw_components = _require_sequence(
            explanation.get("components"), f"events[{index}].explanation.components"
        )
        if not raw_components:
            raise ExplanationAuditError(f"events[{index}].explanation.components must not be empty")

        contributions: dict[str, float] = {}
        component_reported: set[str] = set()
        for component_index, raw_component in enumerate(raw_components):
            component = _require_mapping(
                raw_component, f"events[{index}].explanation.components[{component_index}]"
            )
            signal = component.get("signal")
            if not isinstance(signal, str) or not signal.strip():
                raise ExplanationAuditError(
                    f"events[{index}].explanation.components[{component_index}].signal must be a non-empty string"
                )
            if signal in contributions:
                raise ExplanationAuditError(f"events[{index}] contains duplicate signal {signal!r}")
            contributions[signal] = _finite_float(
                component.get("contribution"),
                f"events[{index}].explanation.components[{component_index}].contribution",
            )
            reported_flag = component.get("reported")
            if not isinstance(reported_flag, bool):
                raise ExplanationAuditError(
                    f"events[{index}].explanation.components[{component_index}].reported must be a boolean"
                )
            if reported_flag:
                component_reported.add(signal)

        raw_reported = _require_sequence(
            explanation.get("reported_components"),
            f"events[{index}].explanation.reported_components",
        )
        if not all(isinstance(value, str) and value for value in raw_reported):
            raise ExplanationAuditError(f"events[{index}].explanation.reported_components must contain strings")
        reported = set(raw_reported)
        if len(reported) != len(raw_reported):
            raise ExplanationAuditError(f"events[{index}].explanation.reported_components contains duplicates")

        event_findings: list[AuditFinding] = []
        if reported != component_reported:
            event_findings.append(
                _finding(event_ref, "inconsistent_reported_components", "component flags and summary disagree")
            )
        unknown = sorted(reported - contributions.keys())
        if unknown:
            event_findings.append(
                _finding(event_ref, "unknown_reported_component", f"unknown count={len(unknown)}")
            )

        reconstructed = baseline + math.fsum(contributions.values())
        additivity_error = abs(reconstructed - score)
        additivity_errors.append(additivity_error)
        if additivity_error > policy.additivity_atol:
            event_findings.append(
                _finding(
                    event_ref,
                    "additivity_mismatch",
                    f"absolute error={additivity_error:.6g}; tolerance={policy.additivity_atol:.6g}",
                )
            )
        if score + policy.additivity_atol < threshold:
            event_findings.append(
                _finding(event_ref, "score_below_threshold", "event score does not meet its applied threshold")
            )

        positive = {signal: max(value, 0.0) for signal, value in contributions.items()}
        positive_total = math.fsum(positive.values())
        if positive_total <= 0.0:
            coverage = 0.0
            material_recall = 0.0
            event_findings.append(
                _finding(event_ref, "no_positive_contribution", "reported anomaly has no positive component")
            )
        else:
            nonpositive_reported = [signal for signal in reported if contributions.get(signal, 0.0) <= 0.0]
            if nonpositive_reported:
                event_findings.append(
                    _finding(
                        event_ref,
                        "nonpositive_component_reported",
                        f"non-positive reported count={len(nonpositive_reported)}",
                    )
                )
            coverage = math.fsum(positive.get(signal, 0.0) for signal in reported) / positive_total
            material = {
                signal
                for signal, contribution in positive.items()
                if contribution / positive_total >= policy.material_component_share
            }
            material_recall = len(material & reported) / len(material) if material else 1.0
            if coverage + policy.additivity_atol < policy.min_positive_coverage:
                event_findings.append(
                    _finding(
                        event_ref,
                        "insufficient_positive_coverage",
                        f"coverage={coverage:.6f}; minimum={policy.min_positive_coverage:.6f}",
                    )
                )
            if material_recall + policy.additivity_atol < policy.min_material_recall:
                event_findings.append(
                    _finding(
                        event_ref,
                        "material_component_omitted",
                        f"recall={material_recall:.6f}; minimum={policy.min_material_recall:.6f}",
                    )
                )

        coverages.append(coverage)
        material_recalls.append(material_recall)
        if event_findings:
            failed_event_refs.add(event_ref)
            findings.extend(event_findings)

    findings.sort()
    return {
        "schema_version": 1,
        "passed": not findings,
        "policy": asdict(policy),
        "summary": {
            "event_count": len(events),
            "failed_event_count": len(failed_event_refs),
            "finding_count": len(findings),
            "max_additivity_error": max(additivity_errors, default=0.0),
            "min_positive_coverage": min(coverages, default=0.0),
            "mean_positive_coverage": math.fsum(coverages) / len(coverages) if coverages else 0.0,
            "min_material_recall": min(material_recalls, default=0.0),
        },
        "findings": [asdict(finding) for finding in findings],
    }


def _reject_constant(value: str) -> None:
    raise ExplanationAuditError(f"non-finite JSON number {value!r} is not allowed")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExplanationAuditError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def load_artifact(path: Path) -> Mapping[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ExplanationAuditError("input must be a regular, non-symlink file")
    if path.stat().st_size > MAX_INPUT_BYTES:
        raise ExplanationAuditError(f"input exceeds {MAX_INPUT_BYTES} bytes")
    try:
        parsed = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ExplanationAuditError(f"could not read valid JSON: {exc}") from exc
    return _require_mapping(parsed, "artifact")


def write_report(report: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit anomaly explanation fidelity")
    parser.add_argument("artifact", type=Path, help="anomalies.json produced by src.cli")
    parser.add_argument("--output", type=Path, default=Path("artifacts/explanation-audit.json"))
    parser.add_argument("--min-events", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        artifact = load_artifact(args.artifact)
        report = audit_explanations(artifact, ExplanationAuditPolicy(min_events=args.min_events))
        write_report(report, args.output)
    except (ExplanationAuditError, OSError, ValueError) as exc:
        parser.exit(2, f"explanation audit error: {exc}\n")
    print(json.dumps({"passed": report["passed"], **report["summary"]}, indent=2, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

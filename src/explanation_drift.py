from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import stat
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

MAX_INPUT_BYTES = 4_000_000
MAX_EVENTS = 10_000
MAX_REASONS_PER_EVENT = 32
MAX_LABEL_LENGTH = 200


class ExplanationDriftInputError(ValueError):
    """Raised when explanation evidence cannot be parsed safely."""


@dataclass(frozen=True)
class ExplanationEvent:
    regime: str
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class DriftIssue:
    code: str
    message: str


@dataclass(frozen=True)
class ReasonShift:
    reason: str
    baseline_share: float
    candidate_share: float
    delta: float


@dataclass(frozen=True)
class RegimeDrift:
    regime: str
    baseline_events: int
    candidate_events: int
    observed_jsd: float | None
    bootstrap_upper_jsd: float | None
    largest_reason_shifts: tuple[ReasonShift, ...]
    issues: tuple[DriftIssue, ...]

    @property
    def ok(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "ok": self.ok,
            "baseline_events": self.baseline_events,
            "candidate_events": self.candidate_events,
            "observed_jsd": self.observed_jsd,
            "bootstrap_upper_jsd": self.bootstrap_upper_jsd,
            "largest_reason_shifts": [asdict(shift) for shift in self.largest_reason_shifts],
            "issues": [asdict(issue) for issue in self.issues],
        }


@dataclass(frozen=True)
class ExplanationDriftReport:
    minimum_events: int
    maximum_jsd: float
    confidence: float
    bootstrap_samples: int
    seed: int
    regimes: tuple[RegimeDrift, ...]

    @property
    def ok(self) -> bool:
        return bool(self.regimes) and all(regime.ok for regime in self.regimes)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "parameters": {
                "minimum_events": self.minimum_events,
                "maximum_jsd": self.maximum_jsd,
                "confidence": self.confidence,
                "bootstrap_samples": self.bootstrap_samples,
                "seed": self.seed,
            },
            "regimes": [regime.as_dict() for regime in self.regimes],
        }


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ExplanationDriftInputError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _label(value: Any, field: str, line: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExplanationDriftInputError(f"line {line}: {field} must be a non-empty string")
    if len(value) > MAX_LABEL_LENGTH:
        raise ExplanationDriftInputError(
            f"line {line}: {field} exceeds {MAX_LABEL_LENGTH} characters"
        )
    return value


def _parse_event(payload: Any, line: int) -> ExplanationEvent:
    if not isinstance(payload, dict):
        raise ExplanationDriftInputError(f"line {line}: event must be a JSON object")
    regime = _label(payload.get("regime"), "regime", line)
    reasons_raw = payload.get("reasons")
    if not isinstance(reasons_raw, list) or not reasons_raw:
        raise ExplanationDriftInputError(f"line {line}: reasons must be a non-empty array")
    if len(reasons_raw) > MAX_REASONS_PER_EVENT:
        raise ExplanationDriftInputError(
            f"line {line}: reasons exceeds {MAX_REASONS_PER_EVENT} entries"
        )
    reasons = tuple(_label(reason, "reasons", line) for reason in reasons_raw)
    if len(set(reasons)) != len(reasons):
        raise ExplanationDriftInputError(f"line {line}: reasons must not contain duplicates")
    return ExplanationEvent(regime=regime, reasons=reasons)


def load_explanation_events(path: str | Path) -> list[ExplanationEvent]:
    source = Path(path)
    try:
        metadata = source.lstat()
    except OSError as exc:
        raise ExplanationDriftInputError(f"cannot inspect input: {exc}") from exc
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ExplanationDriftInputError("input must be a regular, non-symlink file")
    if metadata.st_size > MAX_INPUT_BYTES:
        raise ExplanationDriftInputError(f"input exceeds {MAX_INPUT_BYTES} bytes")

    try:
        with source.open("rb") as handle:
            raw_bytes = handle.read(MAX_INPUT_BYTES + 1)
        if len(raw_bytes) > MAX_INPUT_BYTES:
            raise ExplanationDriftInputError(f"input exceeds {MAX_INPUT_BYTES} bytes")
        text = raw_bytes.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise ExplanationDriftInputError(f"cannot read UTF-8 input: {exc}") from exc

    events: list[ExplanationEvent] = []
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        if not raw_line.strip():
            raise ExplanationDriftInputError(f"line {line_number}: blank lines are not allowed")
        if len(events) >= MAX_EVENTS:
            raise ExplanationDriftInputError(f"input exceeds {MAX_EVENTS} events")
        try:
            payload = json.loads(
                raw_line,
                object_pairs_hook=_strict_object,
                parse_constant=lambda value, current_line=line_number: (
                    _ for _ in ()
                ).throw(
                    ExplanationDriftInputError(
                        f"line {current_line}: non-finite number {value}"
                    )
                ),
            )
        except json.JSONDecodeError as exc:
            raise ExplanationDriftInputError(
                f"line {line_number}: invalid JSON: {exc.msg}"
            ) from exc
        events.append(_parse_event(payload, line_number))
    if not events:
        raise ExplanationDriftInputError("input contains no explanation events")
    return events


def _reason_counts(events: list[ExplanationEvent]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for event in events:
        counts.update(event.reasons)
    return counts


def _distribution(counts: Counter[str]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {}
    return {reason: count / total for reason, count in counts.items()}


def jensen_shannon_divergence(
    baseline: Counter[str], candidate: Counter[str]
) -> float:
    """Return base-2 JSD in [0, 1] for two reason occurrence distributions."""

    left = _distribution(baseline)
    right = _distribution(candidate)
    keys = set(left) | set(right)
    if not keys:
        return 0.0

    def divergence(distribution: dict[str, float]) -> float:
        value = 0.0
        for key in keys:
            probability = distribution.get(key, 0.0)
            if probability == 0:
                continue
            midpoint = (left.get(key, 0.0) + right.get(key, 0.0)) / 2
            value += probability * math.log2(probability / midpoint)
        return value

    return (divergence(left) + divergence(right)) / 2


def _empirical_upper(values: list[float], confidence: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(confidence * len(ordered)) - 1)
    return ordered[index]


def _bootstrap_upper_jsd(
    baseline: list[ExplanationEvent],
    candidate: list[ExplanationEvent],
    *,
    samples: int,
    confidence: float,
    seed: int,
) -> float:
    generator = random.Random(seed)
    values: list[float] = []
    for _ in range(samples):
        baseline_sample = [generator.choice(baseline) for _ in baseline]
        candidate_sample = [generator.choice(candidate) for _ in candidate]
        values.append(
            jensen_shannon_divergence(
                _reason_counts(baseline_sample), _reason_counts(candidate_sample)
            )
        )
    return _empirical_upper(values, confidence)


def _regime_seed(seed: int, regime: str) -> int:
    digest = hashlib.sha256(regime.encode("utf-8")).digest()
    return seed ^ int.from_bytes(digest[:8], "big")


def _largest_reason_shifts(
    baseline: Counter[str], candidate: Counter[str], *, limit: int = 10
) -> tuple[ReasonShift, ...]:
    left = _distribution(baseline)
    right = _distribution(candidate)
    shifts = [
        ReasonShift(
            reason=reason,
            baseline_share=round(left.get(reason, 0.0), 6),
            candidate_share=round(right.get(reason, 0.0), 6),
            delta=round(right.get(reason, 0.0) - left.get(reason, 0.0), 6),
        )
        for reason in set(left) | set(right)
    ]
    return tuple(sorted(shifts, key=lambda item: (-abs(item.delta), item.reason))[:limit])


def evaluate_explanation_drift(
    baseline: list[ExplanationEvent],
    candidate: list[ExplanationEvent],
    *,
    minimum_events: int = 20,
    maximum_jsd: float = 0.10,
    bootstrap_samples: int = 500,
    confidence: float = 0.95,
    seed: int = 42,
) -> ExplanationDriftReport:
    if isinstance(minimum_events, bool) or not isinstance(minimum_events, int):
        raise ExplanationDriftInputError("minimum_events must be an integer")
    if not 2 <= minimum_events <= MAX_EVENTS:
        raise ExplanationDriftInputError(f"minimum_events must be between 2 and {MAX_EVENTS}")
    if (
        isinstance(maximum_jsd, bool)
        or not isinstance(maximum_jsd, (int, float))
        or not math.isfinite(maximum_jsd)
        or not 0 <= maximum_jsd <= 1
    ):
        raise ExplanationDriftInputError("maximum_jsd must be between 0 and 1")
    if isinstance(bootstrap_samples, bool) or not isinstance(bootstrap_samples, int):
        raise ExplanationDriftInputError("bootstrap_samples must be an integer")
    if not 100 <= bootstrap_samples <= 10_000:
        raise ExplanationDriftInputError("bootstrap_samples must be between 100 and 10000")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(confidence)
        or not 0.5 < confidence < 1
    ):
        raise ExplanationDriftInputError("confidence must be greater than 0.5 and less than 1")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ExplanationDriftInputError("seed must be an integer in [0, 2^32)")

    by_baseline: dict[str, list[ExplanationEvent]] = defaultdict(list)
    by_candidate: dict[str, list[ExplanationEvent]] = defaultdict(list)
    for event in baseline:
        by_baseline[event.regime].append(event)
    for event in candidate:
        by_candidate[event.regime].append(event)

    results: list[RegimeDrift] = []
    for regime in sorted(set(by_baseline) | set(by_candidate)):
        baseline_events = by_baseline[regime]
        candidate_events = by_candidate[regime]
        issues: list[DriftIssue] = []
        if len(baseline_events) < minimum_events or len(candidate_events) < minimum_events:
            issues.append(
                DriftIssue(
                    "regime.insufficient_support",
                    f"requires at least {minimum_events} events in each cohort; found "
                    f"baseline={len(baseline_events)}, candidate={len(candidate_events)}",
                )
            )
            results.append(
                RegimeDrift(
                    regime,
                    len(baseline_events),
                    len(candidate_events),
                    None,
                    None,
                    (),
                    tuple(issues),
                )
            )
            continue

        baseline_counts = _reason_counts(baseline_events)
        candidate_counts = _reason_counts(candidate_events)
        observed = jensen_shannon_divergence(baseline_counts, candidate_counts)
        upper = _bootstrap_upper_jsd(
            baseline_events,
            candidate_events,
            samples=bootstrap_samples,
            confidence=confidence,
            seed=_regime_seed(seed, regime),
        )
        if upper > maximum_jsd:
            issues.append(
                DriftIssue(
                    "regime.drift_exceeds_threshold",
                    f"bootstrap upper JSD {upper:.6f} exceeds {maximum_jsd:.6f}",
                )
            )
        results.append(
            RegimeDrift(
                regime=regime,
                baseline_events=len(baseline_events),
                candidate_events=len(candidate_events),
                observed_jsd=round(observed, 6),
                bootstrap_upper_jsd=round(upper, 6),
                largest_reason_shifts=_largest_reason_shifts(
                    baseline_counts, candidate_counts
                ),
                issues=tuple(issues),
            )
        )
    return ExplanationDriftReport(
        minimum_events=minimum_events,
        maximum_jsd=maximum_jsd,
        confidence=confidence,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
        regimes=tuple(results),
    )


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gate regime-conditioned anomaly explanation drift."
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--minimum-events", type=int, default=20)
    parser.add_argument("--maximum-jsd", type=float, default=0.10)
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)

    try:
        report = evaluate_explanation_drift(
            load_explanation_events(args.baseline),
            load_explanation_events(args.candidate),
            minimum_events=args.minimum_events,
            maximum_jsd=args.maximum_jsd,
            bootstrap_samples=args.bootstrap_samples,
            confidence=args.confidence,
            seed=args.seed,
        ).as_dict()
    except ExplanationDriftInputError as exc:
        report = {
            "ok": False,
            "regimes": [],
            "issues": [{"code": "input.invalid", "message": str(exc)}],
        }
    if args.output is None:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _write_report(args.output, report)
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

# Regime-conditioned explanation drift

An anomaly detector can preserve its alert rate while changing *why* it says
events are unusual. `src.explanation_drift` compares reason-code distributions
between a reviewed baseline and a candidate release separately for each
volatility regime.

The input is JSONL. Each row needs the regime and the event's unique reason
codes; additional event fields may be retained for lineage elsewhere.

```json
{"timestamp":"2026-01-02T10:00:00Z","regime":"high","reasons":["extreme volume change","multivariate Isolation Forest outlier"]}
```

Run the release gate with prespecified thresholds:

```bash
python -m src.explanation_drift \
  --baseline artifacts/reasons-baseline.jsonl \
  --candidate artifacts/reasons-candidate.jsonl \
  --minimum-events 20 \
  --maximum-jsd 0.10 \
  --bootstrap-samples 500 \
  --confidence 0.95 \
  --seed 42 \
  --output artifacts/explanation-drift.json
```

The command exits `0` only when every observed regime passes. Invalid evidence,
insufficient support, or excessive drift exits `2`. Reports are deterministic
for a fixed seed and written atomically.

## Method

For each regime, the evaluator:

1. counts reason occurrences across anomaly events;
2. computes base-2 Jensen-Shannon divergence, bounded to `[0, 1]`;
3. independently resamples baseline and candidate events with replacement;
4. gates the configured empirical upper confidence quantile, rather than only
   the observed point estimate;
5. reports the ten largest signed reason-share shifts for review.

Both cohorts must meet the minimum event count in every regime present in either
cohort. Input is bounded to 4 MB, 10,000 events, and 32 unique reasons per event;
duplicate JSON keys, non-finite numbers, blank lines, oversized labels, and
symlink inputs are rejected.

## Scope and limitations

The bootstrap treats events as independent. Market anomalies can cluster in
time, so the interval may be optimistic under serial dependence. JSD also
detects distributional change but does not establish whether a new explanation
is more faithful. Use this global monitoring gate alongside local fidelity and
data-lineage controls.

The next increment should retain session/day blocks and use a block bootstrap
that preserves temporal dependence while binding both cohorts to exact detector
and feature-pipeline versions.

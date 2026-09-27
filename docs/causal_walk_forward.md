# Causal walk-forward anomaly evaluation

The original retrospective engine fits Isolation Forest and percentile-ranks its scores over the complete input series. That is useful for descriptive analysis, but it is not valid evidence of historical alert performance: future observations influence earlier model fits and score ranks.

`src.walk_forward` provides a deployment-shaped evaluation path in which each row is scored only after a model has been fit on earlier rows.

## Causality contract

For every scored timestamp, the evaluator records:

- the first and last observations used to fit Isolation Forest;
- the last observation used by rolling robust and volatility-regime statistics;
- the training sample count, score, threshold, regime and reason codes.

Both history endpoints must be strictly earlier than the scored timestamp. Isolation scores are mapped to percentiles using only the active training window, never the complete evaluation series. Refitting follows a deterministic cadence and uses a bounded trailing window.

The resulting prefix-invariance property is tested directly: evaluating a prefix produces identical records to the same timestamps in a longer series. Appending or changing future data therefore cannot rewrite historical alert evidence.

## Run

```bash
python -m src.walk_forward \
  --rows 500 \
  --min-history 120 \
  --training-window 252 \
  --refit-every 20 \
  --output artifacts/walk-forward.json
```

Custom CSV input must contain strictly ordered, unique timestamps plus positive `open`, `high`, `low`, `close` and `volume` values. Timestamps are normalized to UTC by the CLI. Invalid price envelopes, non-finite values, insufficient history and row-budget violations fail closed.

The JSON artifact is written atomically and includes canonical SHA-256 identities for the input, configuration and complete evidence. It does not include raw OHLCV values.

## Scope and limitations

This path evaluates end-of-period anomaly alerts; using it for intraday decisions would require feature-availability timestamps and exchange-calendar handling. It does not remove survivorship bias, corporate-action errors, missing-session bias or label uncertainty. Isolation Forest is refit deterministically, but the fitted estimator is not yet serialized or signed.

The next step is to bind a versioned market-data manifest and exchange calendar to the evidence, then compare alert precision and delay using the repository's one-to-one temporal event matcher.

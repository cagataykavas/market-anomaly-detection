# Feature-unit invariance audit

An explanation should not change merely because an economically identical market
series is represented in cents instead of dollars, or individual shares instead
of lots. A unit-sensitive anomaly pipeline can silently change flags and reason
codes after a feed, vendor or currency-normalization migration.

`src.unit_invariance` runs the actual anomaly engine on the admitted OHLCV frame
and on controlled price/volume rescalings. It compares:

- the complete score index and score values;
- every anomaly flag;
- the set of emitted event timestamps;
- severity and volatility regime;
- ordered explanation reason codes; and
- reported return and volume-change features.

The default scenarios convert prices to cents and thousands, and volume to lots
and units. Multipliers are finite, positive and bounded. The frame must have a
unique, increasing timestamp index and finite positive OHLCV values. Inputs and
scenario counts are bounded before model execution.

## CLI

```bash
python -m src.unit_invariance market.csv \
  --output artifacts/unit_invariance.json \
  --seed 42
```

The CSV must contain `timestamp,open,high,low,close,volume`. The report is
written atomically and includes a SHA-256 digest over canonical timestamps,
OHLCV values, scenarios, engine configuration, policy and schema version.

Exit codes are:

- `0`: every equivalent-unit scenario passed;
- `2`: admitted evidence exposed a unit-sensitive result;
- `3`: input, policy or engine evidence was malformed.

## Interpretation boundary

Passing establishes invariance only for the declared transformations and
tolerances. It does not prove explanation faithfulness, robustness to data
revisions, currency conversion with changing FX rates, split/corporate-action
handling, or invariance to nonlinear feature engineering. A production pipeline
should run this audit against representative versioned data before feed or unit
migrations and combine it with fidelity, drift and point-in-time data checks.

# Market-data admission

The engine validates the complete OHLCV frame before calculating a rolling feature or
fitting Isolation Forest. The same validation therefore covers direct Python use, the
CLI, and API requests.

## Contract

An admitted frame has:

- 80–100,000 rows;
- the required `open`, `high`, `low`, `close`, and `volume` columns;
- a unique, strictly increasing `DatetimeIndex` without missing timestamps;
- finite numeric values, positive OHLC prices, and non-negative volume; and
- candle bounds where `high` is not below open, low, or close and `low` is not above
  open, high, or close.

The API additionally bounds the timestamp text and total request rows before building
the frame.

## Why fail closed?

Duplicate timestamps make score lookups ambiguous and can turn a scalar into multiple
rows during event construction. Non-finite values and malformed candles can otherwise
flow into rolling features or model fitting and produce plausible-looking anomaly
evidence. Validation runs once, before either path begins.

## Limitations

This contract establishes structural integrity, not data provenance or market
correctness. It does not prove source authority, exchange-calendar completeness,
corporate-action adjustment, point-in-time vendor semantics, or whether a missing
interval should exist. Those checks require instrument and venue metadata from the
upstream data contract.

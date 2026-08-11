# Market Anomaly Detection

A finance-focused anomaly-detection lab for identifying unusual behavior in returns, realized volatility, volume, spreads and cross-asset relationships.

The public baseline uses synthetic market data so the full pipeline is reproducible without proprietary feeds.

## Core methods

- rolling robust z-scores using median/MAD
- volatility and volume spike detection
- Isolation Forest on multivariate market features
- event-window extraction around detected anomalies
- precision-friendly thresholding for sparse anomaly regimes
- walk-forward evaluation to avoid look-ahead leakage

## Architecture

```text
OHLCV / market features
        |
        v
cleaning + returns + rolling statistics
        |
        +--> robust univariate detectors
        |
        +--> multivariate Isolation Forest
        |
        v
anomaly score fusion
        |
        v
event windows + diagnostics + plots
```

## Quick start

```bash
pip install -r requirements.txt
python -m src.demo
```

## Portfolio focus

This repository emphasizes finance-aware evaluation rather than naive random train/test splitting. Market anomalies are sparse, time-dependent and regime-sensitive, so the code uses chronological windows and explicit event extraction.

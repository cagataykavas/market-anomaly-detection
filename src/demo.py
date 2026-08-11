from __future__ import annotations

import numpy as np
import pandas as pd

from .detectors import event_windows, isolation_forest, robust_zscore


def synthetic_market(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0002, 0.01, n)
    shock_idx = np.array([350, 720, 1120, 1300])
    returns[shock_idx] += np.array([-0.09, 0.08, -0.11, 0.07])
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + rng.uniform(0.001, 0.02, n))
    low = close * (1 - rng.uniform(0.001, 0.02, n))
    volume = rng.lognormal(13.0, 0.35, n)
    volume[shock_idx] *= np.array([4.0, 3.0, 5.0, 3.5])
    index = pd.date_range("2022-01-01", periods=n, freq="B")
    return pd.DataFrame({"close": close, "high": high, "low": low, "volume": volume}, index=index)


def main() -> None:
    market = synthetic_market()
    returns = market["close"].pct_change().fillna(0.0)
    rz = robust_zscore(returns, window=60, threshold=4.0)
    iso = isolation_forest(market, contamination=0.02)

    print(f"robust anomalies: {int(rz.flags.sum())}")
    print(f"isolation-forest anomalies: {int(iso.flags.sum())}")
    print(f"event windows: {len(event_windows(market, iso.flags, radius=3))}")


if __name__ == "__main__":
    main()

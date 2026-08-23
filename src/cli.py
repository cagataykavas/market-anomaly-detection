from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .engine import MarketAnomalyEngine
from .evaluation import event_detection_metrics
from .report import render_report
from .synthetic import synthetic_ohlcv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Market anomaly detection reference project")
    parser.add_argument("--input", type=Path, help="CSV with timestamp,open,high,low,close,volume")
    parser.add_argument("--rows", type=int, default=900)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--json-output", type=Path, default=Path("artifacts/anomalies.json"))
    parser.add_argument("--html-output", type=Path, default=Path("artifacts/anomalies.html"))
    args = parser.parse_args(argv)

    labels = None
    if args.input:
        frame = pd.read_csv(args.input, parse_dates=["timestamp"]).set_index("timestamp").sort_index()
    else:
        frame, labels = synthetic_ohlcv(args.rows, args.seed)

    result = MarketAnomalyEngine().analyze(frame, seed=args.seed)
    metrics = None
    if labels is not None:
        flags = pd.Series({pd.Timestamp(key): value for key, value in result["flags"].items()}).sort_index()
        metrics = event_detection_metrics(flags, labels)
        result["evaluation"] = metrics

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    render_report(result, args.html_output, metrics)
    print(json.dumps({
        "anomalies": result["anomaly_count"],
        "anomaly_rate": result["anomaly_rate"],
        "evaluation": metrics,
        "json": str(args.json_output),
        "html": str(args.html_output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments import Baseline_yfinance as baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and prepare market features for DVC.")
    parser.add_argument("--tickers", nargs="+", default=baseline.TICKERS)
    parser.add_argument("--start", default=baseline.START_DATE)
    parser.add_argument("--end", default=baseline.END_DATE)
    parser.add_argument("--output", type=Path, default=Path("data/processed.csv"))
    args = parser.parse_args()

    data = baseline.download_data(args.tickers, args.start, args.end)
    data = baseline.add_technical_indicators(data)
    data = baseline.add_turbulence(data)
    data["datadate"] = pd.to_datetime(data["datadate"]).dt.strftime("%Y-%m-%d")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Saved {len(data):,} rows and {len(data.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()

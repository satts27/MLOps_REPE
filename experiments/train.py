from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import pandas as pd

from experiments import Baseline_yfinance as baseline


def finite_metric(value: float) -> float | None:
    return float(value) if pd.notna(value) and value not in (float("inf"), float("-inf")) else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RL ensemble and track it with MLflow.")
    parser.add_argument("--data", type=Path, default=Path("data/processed.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/baseline"))
    parser.add_argument("--experiment", default="yfinance-rl-baseline")
    parser.add_argument("--a2c-timesteps", type=int, default=baseline.A2C_TIMESTEPS)
    parser.add_argument("--ppo-timesteps", type=int, default=baseline.PPO_TIMESTEPS)
    parser.add_argument("--ddpg-timesteps", type=int, default=baseline.DDPG_TIMESTEPS)
    parser.add_argument("--rebalance-window", type=int, default=baseline.REBALANCE_WINDOW)
    parser.add_argument("--validation-window", type=int, default=baseline.VALIDATION_WINDOW)
    parser.add_argument("--test-window", type=int, default=baseline.TEST_WINDOW)
    parser.add_argument("--seed", type=int, default=baseline.SEED)
    parser.add_argument("--quick", action="store_true", help="Use small timesteps for a smoke run.")
    args = parser.parse_args()

    if args.quick:
        args.a2c_timesteps = min(args.a2c_timesteps, 200)
        args.ppo_timesteps = min(args.ppo_timesteps, 300)
        args.ddpg_timesteps = min(args.ddpg_timesteps, 200)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(args.data, parse_dates=["datadate"])

    baseline.OUTPUT_DIR = args.output_dir
    baseline.MODEL_DIR = args.model_dir
    baseline.A2C_TIMESTEPS = args.a2c_timesteps
    baseline.PPO_TIMESTEPS = args.ppo_timesteps
    baseline.DDPG_TIMESTEPS = args.ddpg_timesteps
    baseline.REBALANCE_WINDOW = args.rebalance_window
    baseline.VALIDATION_WINDOW = args.validation_window
    baseline.TEST_WINDOW = args.test_window
    baseline.SEED = args.seed

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run() as run:
        mlflow.log_params({
            "a2c_timesteps": args.a2c_timesteps,
            "ppo_timesteps": args.ppo_timesteps,
            "ddpg_timesteps": args.ddpg_timesteps,
            "rebalance_window": args.rebalance_window,
            "validation_window": args.validation_window,
            "test_window": args.test_window,
            "seed": args.seed,
            "rows": len(data),
            "tickers": data["tic"].nunique(),
        })
        mlflow.log_artifact(str(args.data), artifact_path="data")

        summary = baseline.run_ensemble_strategy(data)
        summary_path = args.output_dir / "ensemble_summary.csv"
        mlflow.log_artifact(str(summary_path), artifact_path="outputs")
        if args.model_dir.exists():
            mlflow.log_artifacts(str(args.model_dir), artifact_path="models")

        if not summary.empty:
            for column in ("a2c_sharpe", "ppo_sharpe", "ddpg_sharpe"):
                metric = finite_metric(summary[column].mean())
                if metric is not None:
                    mlflow.log_metric(f"mean_{column}", metric)
            mlflow.log_metric("windows_completed", len(summary))
            mlflow.set_tag("best_model", summary["best_model"].mode().iat[0])

        print(f"MLflow run: {run.info.run_id}")
        print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()

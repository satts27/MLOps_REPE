#!/usr/bin/env python3
"""
Native RL Trading Pipeline using:
- yfinance
- gymnasium
- stable-baselines3
- ta (technical indicators)

Implements:
1. Data download from Yahoo Finance
2. Technical indicator generation
3. Turbulence index calculation
4. Custom Gym trading environment
5. A2C / PPO / DDPG training
6. Ensemble selection using Sharpe Ratio
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timedelta

import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from ta.trend import MACD, CCIIndicator, ADXIndicator
from ta.momentum import RSIIndicator

import matplotlib.pyplot as plt


# =========================================================
# CONFIG
# =========================================================

TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META"
]

END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = (
    datetime.now() - timedelta(days=365)
).strftime("%Y-%m-%d")

INITIAL_BALANCE = 1_000_000
HMAX_NORMALIZE = 100
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4

A2C_TIMESTEPS = 5_000
PPO_TIMESTEPS = 10_000
DDPG_TIMESTEPS = 5_000
PROGRESS_PRINT_EVERY = 1_000

REBALANCE_WINDOW = 15
VALIDATION_WINDOW = 15
TEST_WINDOW = 30

SEED = 42

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("models") / "baseline"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class TrainingProgressCallback(BaseCallback):

    def __init__(self, model_name, total_timesteps, print_every):

        super().__init__()

        self.model_name = model_name
        self.total_timesteps = total_timesteps
        self.print_every = print_every
        self.next_print = print_every

    def _on_training_start(self):

        print(
            f"[{self.model_name}] started training "
            f"for {self.total_timesteps:,} timesteps",
            flush=True
        )

    def _on_step(self):

        if self.num_timesteps >= self.next_print:

            percent = (
                self.num_timesteps /
                self.total_timesteps *
                100
            )

            print(
                f"[{self.model_name}] "
                f"{self.num_timesteps:,}/{self.total_timesteps:,} "
                f"timesteps ({percent:.1f}%)",
                flush=True
            )

            self.next_print += self.print_every

        return True

    def _on_training_end(self):

        print(
            f"[{self.model_name}] finished training",
            flush=True
        )


# =========================================================
# DOWNLOAD DATA
# =========================================================

def download_data(tickers, start, end):

    all_data = []

    for ticker in tickers:

        print(f"Downloading {ticker}")

        df = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=False
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            print(f"Skipping {ticker}: no downloaded rows")
            continue

        df = df.reset_index()

        df["tic"] = ticker

        df = df.rename(columns={
            "Date": "datadate",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjcp",
            "Volume": "volume"
        })

        df["datadate"] = pd.to_datetime(df["datadate"])

        all_data.append(df)

    final_df = pd.concat(all_data, ignore_index=True)

    final_df = final_df.sort_values(
        ["datadate", "tic"]
    ).reset_index(drop=True)

    return final_df


# =========================================================
# TECHNICAL INDICATORS
# =========================================================

def add_technical_indicators(df):

    dfs = []

    for ticker in df.tic.unique():

        ticker_df = df[df.tic == ticker].copy()

        ticker_df["macd"] = MACD(
            close=ticker_df["adjcp"]
        ).macd()

        ticker_df["rsi"] = RSIIndicator(
            close=ticker_df["adjcp"]
        ).rsi()

        ticker_df["cci"] = CCIIndicator(
            high=ticker_df["high"],
            low=ticker_df["low"],
            close=ticker_df["adjcp"]
        ).cci()

        ticker_df["adx"] = ADXIndicator(
            high=ticker_df["high"],
            low=ticker_df["low"],
            close=ticker_df["adjcp"]
        ).adx()

        dfs.append(ticker_df)

    final_df = pd.concat(dfs)

    final_df = final_df.sort_values(
        ["datadate", "tic"]
    ).reset_index(drop=True)

    final_df = final_df.bfill()

    return final_df


# =========================================================
# TURBULENCE INDEX
# =========================================================

def calculate_turbulence(df):

    pivot = df.pivot(
        index="datadate",
        columns="tic",
        values="adjcp"
    )

    unique_dates = pivot.index

    turbulence_window = min(
        REBALANCE_WINDOW,
        max(1, len(unique_dates) - 1)
    )

    turbulence = [0] * turbulence_window

    for i in range(turbulence_window, len(unique_dates)):

        current_price = pivot.iloc[i]

        hist_price = pivot.iloc[:i]

        cov = hist_price.cov()

        diff = current_price - hist_price.mean()

        temp = diff.values.T @ np.linalg.pinv(cov.values) @ diff.values

        turbulence.append(temp)

    turbulence_df = pd.DataFrame({
        "datadate": unique_dates,
        "turbulence": turbulence
    })

    return turbulence_df


def add_turbulence(df):

    turbulence_df = calculate_turbulence(df)

    df = df.merge(
        turbulence_df,
        on="datadate"
    )

    return df


# =========================================================
# DATA SPLIT
# =========================================================

def data_split(df, start, end):

    data = df[
        (df.datadate >= start) &
        (df.datadate < end)
    ]

    data = data.sort_values(
        ["datadate", "tic"]
    )

    data.index = data.datadate.factorize()[0]

    return data


# =========================================================
# ENVIRONMENT
# =========================================================

class StockTradingEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df,
        turbulence_threshold=None
    ):

        super().__init__()

        self.df = df
        self.stock_dim = len(df.tic.unique())

        self.day = 0

        self.data = self.df.loc[self.day, :]

        self.turbulence_threshold = turbulence_threshold

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.stock_dim,),
            dtype=np.float32
        )

        self.state_dim = (
            1 +
            self.stock_dim +
            self.stock_dim +
            self.stock_dim * 4
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32
        )

        self.state = self._initiate_state()

        self.asset_memory = [INITIAL_BALANCE]

    def _initiate_state(self):

        return np.array(

            [INITIAL_BALANCE] +

            self.data.adjcp.values.tolist() +

            [0] * self.stock_dim +

            self.data.macd.values.tolist() +

            self.data.rsi.values.tolist() +

            self.data.cci.values.tolist() +

            self.data.adx.values.tolist()

        )

    def _buy_stock(self, index, action):

        available_amount = self.state[0] // self.state[index + 1]

        buy_num_shares = min(
            available_amount,
            action
        )

        buy_amount = (
            self.state[index + 1] *
            buy_num_shares
        )

        self.state[0] -= buy_amount * (
            1 + TRANSACTION_FEE_PERCENT
        )

        self.state[
            self.stock_dim + 1 + index
        ] += buy_num_shares

    def _sell_stock(self, index, action):

        sell_num_shares = min(
            abs(action),
            self.state[self.stock_dim + 1 + index]
        )

        sell_amount = (
            self.state[index + 1] *
            sell_num_shares
        )

        self.state[0] += sell_amount * (
            1 - TRANSACTION_FEE_PERCENT
        )

        self.state[
            self.stock_dim + 1 + index
        ] -= sell_num_shares

    def step(self, actions):

        terminal = self.day >= len(
            self.df.index.unique()
        ) - 1

        if terminal:

            return self.state, 0, True, False, {}

        actions = actions * HMAX_NORMALIZE

        begin_total_asset = (
            self.state[0] +
            np.sum(
                np.array(
                    self.state[1:1+self.stock_dim]
                ) *
                np.array(
                    self.state[
                        self.stock_dim+1:
                        self.stock_dim*2+1
                    ]
                )
            )
        )

        argsort_actions = np.argsort(actions)

        sell_index = argsort_actions[
            :np.where(actions < 0)[0].shape[0]
        ]

        buy_index = argsort_actions[::-1][
            :np.where(actions > 0)[0].shape[0]
        ]

        for index in sell_index:
            self._sell_stock(index, actions[index])

        for index in buy_index:
            self._buy_stock(index, actions[index])

        self.day += 1

        self.data = self.df.loc[self.day, :]

        self.state = np.array(

            [self.state[0]] +

            self.data.adjcp.values.tolist() +

            list(
                self.state[
                    self.stock_dim+1:
                    self.stock_dim*2+1
                ]
            ) +

            self.data.macd.values.tolist() +

            self.data.rsi.values.tolist() +

            self.data.cci.values.tolist() +

            self.data.adx.values.tolist()

        )

        end_total_asset = (
            self.state[0] +
            np.sum(
                np.array(
                    self.state[1:1+self.stock_dim]
                ) *
                np.array(
                    self.state[
                        self.stock_dim+1:
                        self.stock_dim*2+1
                    ]
                )
            )
        )

        reward = (
            end_total_asset -
            begin_total_asset
        ) * REWARD_SCALING

        self.asset_memory.append(end_total_asset)

        return self.state, reward, False, False, {}

    def reset(self, seed=None, options=None):

        super().reset(seed=seed)

        self.day = 0

        self.data = self.df.loc[self.day, :]

        self.state = self._initiate_state()

        self.asset_memory = [INITIAL_BALANCE]

        return self.state, {}

    def render(self):
        return self.state


# =========================================================
# SHARPE RATIO
# =========================================================

def calculate_sharpe(asset_memory):

    returns = pd.Series(asset_memory).pct_change().dropna()

    if len(returns) == 0 or returns.std() == 0:
        return -np.inf

    sharpe = (
        np.sqrt(252) *
        returns.mean() /
        returns.std()
    )

    return sharpe


# =========================================================
# TRAINING
# =========================================================

def train_model(model_name, env):

    if model_name == "A2C":

        total_timesteps = A2C_TIMESTEPS

        model = A2C(
            "MlpPolicy",
            env,
            verbose=0,
            seed=SEED
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=TrainingProgressCallback(
                model_name,
                total_timesteps,
                PROGRESS_PRINT_EVERY
            )
        )

    elif model_name == "PPO":

        total_timesteps = PPO_TIMESTEPS

        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=SEED
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=TrainingProgressCallback(
                model_name,
                total_timesteps,
                PROGRESS_PRINT_EVERY
            )
        )

    elif model_name == "DDPG":

        total_timesteps = DDPG_TIMESTEPS

        n_actions = env.action_space.shape[-1]

        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.5 * np.ones(n_actions)
        )

        model = DDPG(
            "MlpPolicy",
            env,
            action_noise=action_noise,
            verbose=0,
            seed=SEED
        )

        model.learn(
            total_timesteps=total_timesteps,
            callback=TrainingProgressCallback(
                model_name,
                total_timesteps,
                PROGRESS_PRINT_EVERY
            )
        )

    return model


# =========================================================
# VALIDATION
# =========================================================

def validate_model(model, validation_env):

    obs, _ = validation_env.reset()

    done = False

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=True
        )

        obs, reward, done, _, _ = validation_env.step(action)

    sharpe = calculate_sharpe(
        validation_env.asset_memory
    )

    return sharpe, validation_env.asset_memory[-1]


# =========================================================
# MAIN PIPELINE
# =========================================================

def run_ensemble_strategy(df):

    unique_trade_dates = df.datadate.unique()

    summary = []

    test_start = unique_trade_dates[-TEST_WINDOW]
    test_end = unique_trade_dates[-1]

    rolling_points = list(range(
        REBALANCE_WINDOW + VALIDATION_WINDOW + REBALANCE_WINDOW,
        len(unique_trade_dates) - TEST_WINDOW,
        REBALANCE_WINDOW
    ))

    print(
        f"Running {len(rolling_points)} rolling windows. "
        f"Progress summary will be saved to "
        f"{OUTPUT_DIR / 'ensemble_summary.csv'}",
        flush=True
    )

    print(
        f"Holding out final {TEST_WINDOW} trading days for testing: "
        f"{test_start} -> {test_end}",
        flush=True
    )

    for window_number, i in enumerate(rolling_points, start=1):

        print("=" * 60)
        print(
            f"Window {window_number}/{len(rolling_points)} "
            f"(iteration {i})",
            flush=True
        )

        validation_start = unique_trade_dates[
            i - REBALANCE_WINDOW - VALIDATION_WINDOW
        ]

        validation_end = unique_trade_dates[
            i - REBALANCE_WINDOW
        ]

        trade_end = unique_trade_dates[i]

        print("Validation:", validation_start, validation_end)
        print("Trade End:", trade_end)

        train_df = data_split(
            df,
            unique_trade_dates[0],
            validation_start
        )

        validation_df = data_split(
            df,
            validation_start,
            validation_end
        )

        if train_df.empty or validation_df.empty:
            print("Skipping window: not enough train/validation data")
            continue

        train_env = DummyVecEnv([
            lambda: StockTradingEnv(train_df)
        ])

        validation_env = StockTradingEnv(validation_df)

        sharpe_scores = {}
        final_values = {}
        model_paths = {}

        for model_name in ["A2C", "PPO", "DDPG"]:

            print(
                f"Training {model_name} "
                f"for window {window_number}/{len(rolling_points)}",
                flush=True
            )

            model = train_model(
                model_name,
                train_env
            )

            sharpe, final_value = validate_model(
                model,
                validation_env
            )

            sharpe_scores[model_name] = sharpe
            final_values[model_name] = final_value

            model_dir = MODEL_DIR / f"window_{i}"
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / f"{model_name}.zip"
            model.save(model_path)
            model_paths[model_name] = str(model_path)

            print(
                f"{model_name} Sharpe: {sharpe:.4f}; "
                f"final value: {final_value:.2f}; "
                f"saved model: {model_path}",
                flush=True
            )

        best_model = max(
            sharpe_scores,
            key=sharpe_scores.get
        )

        print(f"Best Model: {best_model}", flush=True)

        summary.append({
            "iteration": i,
            "window": window_number,
            "validation_start": validation_start,
            "validation_end": validation_end,
            "trade_end": trade_end,
            "test_start": test_start,
            "test_end": test_end,
            "best_model": best_model,
            "a2c_sharpe": sharpe_scores["A2C"],
            "ppo_sharpe": sharpe_scores["PPO"],
            "ddpg_sharpe": sharpe_scores["DDPG"],
            "a2c_final_value": final_values["A2C"],
            "ppo_final_value": final_values["PPO"],
            "ddpg_final_value": final_values["DDPG"],
            "a2c_model_path": model_paths["A2C"],
            "ppo_model_path": model_paths["PPO"],
            "ddpg_model_path": model_paths["DDPG"],
        })

        pd.DataFrame(summary).to_csv(
            OUTPUT_DIR / "ensemble_summary.csv",
            index=False
        )

        print(
            f"Saved progress after window "
            f"{window_number}/{len(rolling_points)}",
            flush=True
        )

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        OUTPUT_DIR / "ensemble_summary.csv",
        index=False
    )

    print(summary_df)
    return summary_df


# =========================================================
# MAIN
# =========================================================

def main():

    print("Downloading data...")

    df = download_data(
        TICKERS,
        START_DATE,
        END_DATE
    )

    print("Adding technical indicators...")

    df = add_technical_indicators(df)

    print("Calculating turbulence index...")

    df = add_turbulence(df)

    print(df.head())

    print(df.columns)

    run_ensemble_strategy(df)


if __name__ == "__main__":
    main()

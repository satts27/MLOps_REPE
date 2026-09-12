from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import yfinance as yf
import finnhub
from dotenv import load_dotenv

from pathlib import Path
from datetime import datetime, timedelta

load_dotenv()

import gymnasium as gym
from gymnasium import spaces

import torch
from stable_baselines3 import PPO, A2C, DDPG
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise

from ta.trend import MACD, CCIIndicator, ADXIndicator
from ta.momentum import RSIIndicator

from transformers import pipeline


# =========================================================
# CONFIG
# =========================================================

TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
]

END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = (
    datetime.now() - timedelta(days=365)
).strftime("%Y-%m-%d")

INITIAL_BALANCE = 1_000_000
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = 1e-4
HMAX_NORMALIZE = 100

A2C_TIMESTEPS = 5_000
PPO_TIMESTEPS = 10_000
DDPG_TIMESTEPS = 5_000
PROGRESS_PRINT_EVERY = 1_000

REBALANCE_WINDOW = 15
VALIDATION_WINDOW = 15
TEST_WINDOW = 30

SEED = 42
FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_REVISION = "7db323f79b751944bcfa66298ec06977e4518306"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
CACHED_DATASET_FILE = OUTPUT_DIR / "final_dataset.csv"
MODEL_DIR = Path("models") / "bert"
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

            shown_timesteps = min(
                self.num_timesteps,
                self.total_timesteps
            )

            percent = (
                shown_timesteps /
                self.total_timesteps *
                100
            )

            print(
                f"[{self.model_name}] "
                f"{shown_timesteps:,}/{self.total_timesteps:,} "
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
# FINNHUB CLIENT
# =========================================================

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

finnhub_client = finnhub.Client(
    api_key=FINNHUB_API_KEY
)


# =========================================================
# FINBERT
# =========================================================

print("Loading FinBERT...")

FINBERT_DEVICE = 0 if torch.cuda.is_available() else -1

finbert = pipeline(
    "sentiment-analysis",
    model=FINBERT_MODEL,
    revision=FINBERT_REVISION,
    device=FINBERT_DEVICE,
    model_kwargs={"use_safetensors": True}
)

print(
    "FinBERT device:",
    "cuda:0" if FINBERT_DEVICE == 0 else "cpu",
    flush=True
)


# =========================================================
# DOWNLOAD PRICE DATA
# =========================================================


def download_price_data():

    all_data = []

    for ticker_number, ticker in enumerate(TICKERS, start=1):

        print(
            f"Downloading prices for {ticker} "
            f"({ticker_number}/{len(TICKERS)})",
            flush=True
        )

        df = yf.download(
            ticker,
            start=START_DATE,
            end=END_DATE,
            auto_adjust=False
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        if df.empty:
            print(f"Skipping {ticker}: no downloaded rows", flush=True)
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

        print(
            f"Downloaded {len(df):,} price rows for {ticker}",
            flush=True
        )

    if len(all_data) == 0:
        raise RuntimeError("No price data was downloaded from yfinance.")

    final_df = pd.concat(all_data)

    final_df = final_df.sort_values(
        ["datadate", "tic"]
    ).reset_index(drop=True)

    return final_df


# =========================================================
# FINBERT SENTIMENT
# =========================================================


def sentiment_to_score(result):

    label = result["label"].lower()
    score = result["score"]

    if label == "positive":
        return score

    elif label == "negative":
        return -score

    return 0


# =========================================================
# HISTORICAL NEWS EXTRACTION
# =========================================================


def fetch_news_for_ticker(ticker):

    print(f"Fetching news for {ticker}", flush=True)

    news = finnhub_client.company_news(
        ticker,
        _from=START_DATE,
        to=END_DATE
    )

    print(
        f"Fetched {len(news):,} news articles for {ticker}",
        flush=True
    )

    records = []

    for article_number, article in enumerate(news, start=1):

        headline = article.get("headline", "")

        timestamp = article.get("datetime", None)

        if timestamp is None:
            continue

        if len(headline.strip()) == 0:
            continue

        date = pd.to_datetime(
            timestamp,
            unit="s"
        ).date()

        try:

            if (
                article_number == 1 or
                article_number % 25 == 0 or
                article_number == len(news)
            ):
                print(
                    f"Analyzing {ticker} news "
                    f"{article_number}/{len(news)}",
                    flush=True
                )

            result = finbert(headline)[0]

            sentiment_score = sentiment_to_score(result)

            records.append({
                "datadate": pd.to_datetime(date),
                "tic": ticker,
                "headline": headline,
                "sentiment": sentiment_score
            })

        except Exception as e:
            print(e)

    print(
        f"Built {len(records):,} sentiment records for {ticker}",
        flush=True
    )

    return pd.DataFrame(records)


# =========================================================
# BUILD DAILY SENTIMENT DATASET
# =========================================================


def build_sentiment_dataset():

    all_sentiment = []

    for ticker_number, ticker in enumerate(TICKERS, start=1):

        print(
            f"Sentiment ticker {ticker_number}/{len(TICKERS)}: {ticker}",
            flush=True
        )

        df = fetch_news_for_ticker(ticker)

        if not df.empty:
            all_sentiment.append(df)

    if len(all_sentiment) == 0:
        print(
            "No sentiment records found; all sentiment values will be 0.",
            flush=True
        )

        return pd.DataFrame(
            columns=["datadate", "tic", "sentiment"]
        )

    sentiment_df = pd.concat(all_sentiment)

    sentiment_df = (
        sentiment_df
        .groupby(["datadate", "tic"])["sentiment"]
        .mean()
        .reset_index()
    )

    sentiment_df.to_csv(
        OUTPUT_DIR / "daily_sentiment.csv",
        index=False
    )

    print(
        f"Saved daily sentiment rows: {len(sentiment_df):,}",
        flush=True
    )

    return sentiment_df


# =========================================================
# TECHNICAL INDICATORS
# =========================================================


def add_technical_indicators(df):

    dfs = []

    tickers = df.tic.unique()

    for ticker_number, ticker in enumerate(tickers, start=1):

        print(
            f"Generating indicators for {ticker} "
            f"({ticker_number}/{len(tickers)})",
            flush=True
        )

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

    turbulence = [0] * 30

    for i in range(30, len(unique_dates)):

        if i == 30 or i % 50 == 0 or i == len(unique_dates) - 1:
            print(
                f"Calculating turbulence "
                f"{i}/{len(unique_dates) - 1}",
                flush=True
            )

        current_price = pivot.iloc[i]

        hist_price = pivot.iloc[:i]

        cov = hist_price.cov()

        diff = current_price - hist_price.mean()

        temp = (
            diff.values.T
            @ np.linalg.pinv(cov.values)
            @ diff.values
        )

        turbulence.append(temp)

    turbulence_df = pd.DataFrame({
        "datadate": unique_dates,
        "turbulence": turbulence
    })

    return turbulence_df


# =========================================================
# MERGE EVERYTHING
# =========================================================


def build_final_dataset():

    print("Downloading prices...")

    price_df = download_price_data()

    print("Generating technical indicators...")

    price_df = add_technical_indicators(price_df)

    print("Building historical sentiment dataset...")

    sentiment_df = build_sentiment_dataset()

    print("Merging sentiment with prices...")

    final_df = price_df.merge(
        sentiment_df,
        on=["datadate", "tic"],
        how="left"
    )

    final_df["sentiment"] = final_df[
        "sentiment"
    ].fillna(0).astype(float)

    print("Calculating turbulence...")

    turbulence_df = calculate_turbulence(final_df)

    final_df = final_df.merge(
        turbulence_df,
        on="datadate",
        how="left"
    )

    final_df.to_csv(
        CACHED_DATASET_FILE,
        index=False
    )

    print(
        f"Saved final dataset to {CACHED_DATASET_FILE}",
        flush=True
    )

    return final_df


def load_or_build_final_dataset(refresh):

    if CACHED_DATASET_FILE.exists() and not refresh:
        print(
            f"Using cached dataset: {CACHED_DATASET_FILE}",
            flush=True
        )

        final_df = pd.read_csv(CACHED_DATASET_FILE)
        final_df["datadate"] = pd.to_datetime(final_df["datadate"])

        return final_df

    return build_final_dataset()


# =========================================================
# DATA SPLIT
# =========================================================


def data_split(df, start, end):

    data = df[
        (df.datadate >= start)
        &
        (df.datadate < end)
    ]

    data = data.sort_values(
        ["datadate", "tic"]
    )

    data.index = data.datadate.factorize()[0]

    return data


# =========================================================
# RL ENVIRONMENT
# =========================================================


class StockTradingEnv(gym.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(self, df):

        super().__init__()

        self.df = df.sort_values(
            ["datadate", "tic"]
        ).copy()

        self.df.index = self.df.datadate.factorize()[0]

        self.stock_dim = len(df.tic.unique())

        self.day = 0

        self.data = self.df.loc[self.day, :]

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
            self.stock_dim * 5
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

            [INITIAL_BALANCE]

            + self.data.adjcp.values.tolist()

            + [0] * self.stock_dim

            + self.data.macd.values.tolist()

            + self.data.rsi.values.tolist()

            + self.data.cci.values.tolist()

            + self.data.adx.values.tolist()

            + self.data.sentiment.values.tolist()
            ,
            dtype=np.float32
        )

    def _buy_stock(self, index, action):

        available_amount = (
            self.state[0] //
            self.state[index + 1]
        )

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
            self.state[
                self.stock_dim + 1 + index
            ]
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

        terminal = (
            self.day >=
            len(self.df.index.unique()) - 1
        )

        if terminal:
            return self.state, 0, True, False, {}

        actions = actions * HMAX_NORMALIZE

        begin_total_asset = (
            self.state[0]
            + np.sum(
                np.array(
                    self.state[1:1+self.stock_dim]
                )
                * np.array(
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

            [self.state[0]]

            + self.data.adjcp.values.tolist()

            + list(
                self.state[
                    self.stock_dim+1:
                    self.stock_dim*2+1
                ]
            )

            + self.data.macd.values.tolist()

            + self.data.rsi.values.tolist()

            + self.data.cci.values.tolist()

            + self.data.adx.values.tolist()

            + self.data.sentiment.values.tolist()
            ,
            dtype=np.float32
        )

        end_total_asset = (
            self.state[0]
            + np.sum(
                np.array(
                    self.state[1:1+self.stock_dim]
                )
                * np.array(
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


# =========================================================
# ENSEMBLE HELPERS
# =========================================================


def calculate_sharpe(asset_memory):

    returns = pd.Series(asset_memory).pct_change().dropna()

    if len(returns) == 0 or returns.std() == 0:
        return -np.inf

    return np.sqrt(252) * returns.mean() / returns.std()


def train_model(model_name, env, timesteps):

    if model_name == "A2C":

        model = A2C(
            "MlpPolicy",
            env,
            verbose=0,
            seed=SEED
        )

    elif model_name == "PPO":

        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=SEED
        )

    elif model_name == "DDPG":

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

    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.learn(
        total_timesteps=timesteps,
        callback=TrainingProgressCallback(
            model_name,
            timesteps,
            PROGRESS_PRINT_EVERY
        )
    )

    return model


def validate_model(model, env):

    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

    return calculate_sharpe(env.asset_memory), env.asset_memory[-1]


def run_ensemble_strategy(
    df,
    a2c_timesteps,
    ppo_timesteps,
    ddpg_timesteps
):

    unique_dates = df.datadate.sort_values().unique()

    test_start = unique_dates[-TEST_WINDOW]
    test_end = unique_dates[-1]

    rolling_points = list(range(
        REBALANCE_WINDOW + VALIDATION_WINDOW + REBALANCE_WINDOW,
        len(unique_dates) - TEST_WINDOW,
        REBALANCE_WINDOW
    ))

    model_timesteps = {
        "A2C": a2c_timesteps,
        "PPO": ppo_timesteps,
        "DDPG": ddpg_timesteps,
    }

    summary = []
    summary_file = OUTPUT_DIR / "bert_ensemble_summary.csv"

    if summary_file.exists():
        summary_file.unlink()

    print(
        f"Running BERT ensemble over "
        f"{len(rolling_points)} rolling windows",
        flush=True
    )

    print(
        f"Holding out final {TEST_WINDOW} trading days for testing: "
        f"{test_start} -> {test_end}",
        flush=True
    )

    for window_number, i in enumerate(rolling_points, start=1):

        validation_start = unique_dates[
            i - REBALANCE_WINDOW - VALIDATION_WINDOW
        ]
        validation_end = unique_dates[i - REBALANCE_WINDOW]
        trade_end = unique_dates[i]

        print("=" * 60, flush=True)
        print(
            f"Window {window_number}/{len(rolling_points)} "
            f"iteration {i}",
            flush=True
        )
        print(
            f"Validation: {validation_start} -> {validation_end}; "
            f"Trade end marker: {trade_end}",
            flush=True
        )

        train_df = data_split(
            df,
            unique_dates[0],
            validation_start
        )

        validation_df = data_split(
            df,
            validation_start,
            validation_end
        )

        if train_df.empty or validation_df.empty:
            print("Skipping window: not enough data", flush=True)
            continue

        sharpe_scores = {}
        final_values = {}
        model_paths = {}

        for model_name in ["A2C", "PPO", "DDPG"]:

            print(
                f"Training {model_name} for "
                f"window {window_number}/{len(rolling_points)}",
                flush=True
            )

            train_env = DummyVecEnv([
                lambda data=train_df: StockTradingEnv(data)
            ])

            model = train_model(
                model_name,
                train_env,
                model_timesteps[model_name]
            )

            validation_env = StockTradingEnv(validation_df)

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
            summary_file,
            index=False
        )

        print(
            f"Best model: {best_model}. "
            f"Saved {summary_file}",
            flush=True
        )

    return pd.DataFrame(summary)


# =========================================================
# MAIN
# =========================================================


def main():

    parser = argparse.ArgumentParser(
        description="Run the FinBERT sentiment trading experiment."
    )

    parser.add_argument(
        "--timesteps",
        type=int,
        default=PPO_TIMESTEPS,
        help="Number of PPO training timesteps."
    )

    parser.add_argument(
        "--a2c-timesteps",
        type=int,
        default=A2C_TIMESTEPS,
        help="Number of A2C training timesteps per rolling window."
    )

    parser.add_argument(
        "--ddpg-timesteps",
        type=int,
        default=DDPG_TIMESTEPS,
        help="Number of DDPG training timesteps per rolling window."
    )

    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Redownload prices/news instead of using outputs/final_dataset.csv."
    )

    args = parser.parse_args()

    print(
        f"Building FinBERT dataset from {START_DATE} to {END_DATE}",
        flush=True
    )

    final_df = load_or_build_final_dataset(args.refresh_data)

    print(final_df.head())

    print(final_df.columns)

    ensemble_summary = run_ensemble_strategy(
        final_df,
        args.a2c_timesteps,
        args.timesteps,
        args.ddpg_timesteps
    )

    print(ensemble_summary)

    print("Training completed.", flush=True)


if __name__ == "__main__":
    main()

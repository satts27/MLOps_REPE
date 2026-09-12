from __future__ import annotations

import argparse
import os
import numpy as np
import pandas as pd
import yfinance as yf
import finnhub
import torch
import torch.nn as nn
from dotenv import load_dotenv

from pathlib import Path
from datetime import datetime, timedelta

load_dotenv()

from transformers import (
    AutoTokenizer,
    AutoModel
)

from torch.utils.data import Dataset, DataLoader

from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv

import gymnasium as gym
from gymnasium import spaces

from ta.trend import MACD, CCIIndicator, ADXIndicator
from ta.momentum import RSIIndicator


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

WINDOW_SIZE = 15
PRETRAIN_EPOCHS = 10
A2C_TIMESTEPS = 5_000
PPO_TIMESTEPS = 10_000
DDPG_TIMESTEPS = 5_000
PROGRESS_PRINT_EVERY = 1_000
REBALANCE_WINDOW = 15
VALIDATION_WINDOW = 15
TEST_WINDOW = 30
INITIAL_BALANCE = 1_000_000
TRANSACTION_FEE_PERCENT = 0.001

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

PRICE_FEATURES = [
    "open",
    "high",
    "low",
    "adjcp",
    "volume",
    "macd",
    "rsi",
    "cci",
    "adx"
]

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")

FINBERT_MODEL = "ProsusAI/finbert"
FINBERT_REVISION = "7db323f79b751944bcfa66298ec06977e4518306"

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR = Path("models") / "multimodal"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CACHED_DATASET_FILE = (
    OUTPUT_DIR /
    "multimodal_final_dataset.pkl"
)


# =========================================================
# CALLBACK
# =========================================================

class TrainingProgressCallback(BaseCallback):

    def __init__(
        self,
        model_name,
        total_timesteps,
        print_every
    ):

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


# =========================================================
# FINNHUB
# =========================================================

finnhub_client = finnhub.Client(
    api_key=FINNHUB_API_KEY
)


# =========================================================
# FINBERT
# =========================================================

print("Loading FinBERT...", flush=True)

finbert_tokenizer = AutoTokenizer.from_pretrained(
    FINBERT_MODEL,
    revision=FINBERT_REVISION
)

finbert_model = AutoModel.from_pretrained(
    FINBERT_MODEL,
    revision=FINBERT_REVISION,
    use_safetensors=True
).to(DEVICE)

finbert_model.eval()

print(f"FinBERT device: {DEVICE}", flush=True)


# =========================================================
# DOWNLOAD PRICE DATA
# =========================================================


def download_price_data():

    all_data = []

    for ticker_number, ticker in enumerate(TICKERS, start=1):

        print(
            f"Downloading {ticker} "
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

    final_df = pd.concat(all_data)

    final_df = final_df.sort_values(
        ["datadate", "tic"]
    )

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

    # =====================================================
    # FIX 1: REMOVE INF VALUES
    # =====================================================

    final_df = final_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # =====================================================
    # FIX 2: REMOVE UNSTABLE EARLY ROWS
    # =====================================================

    final_df = final_df.dropna(
        subset=PRICE_FEATURES
    )

    final_df = final_df.sort_values(
        ["datadate", "tic"]
    )

    print(
        "Remaining NaNs after cleanup:",
        final_df[PRICE_FEATURES].isna().sum().sum(),
        flush=True
    )

    return final_df


# =========================================================
# HISTORICAL NEWS
# =========================================================


def fetch_news_for_ticker(ticker):

    print(f"Fetching news for {ticker}", flush=True)

    news = finnhub_client.company_news(
        ticker,
        _from=START_DATE,
        to=END_DATE
    )

    print(
        f"Fetched {len(news):,} articles",
        flush=True
    )

    records = []

    for article in news:

        headline = article.get("headline", "")

        timestamp = article.get("datetime", None)

        if timestamp is None:
            continue

        if len(headline.strip()) == 0:
            continue

        date = pd.to_datetime(
            timestamp,
            unit="s"
        ).normalize()

        records.append({
            "datadate": date,
            "tic": ticker,
            "headline": headline
        })

    return pd.DataFrame(records)


# =========================================================
# FINBERT EMBEDDINGS
# =========================================================


def generate_finbert_embedding(text):

    inputs = finbert_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(DEVICE)

    with torch.no_grad():

        outputs = finbert_model(**inputs)

    cls_embedding = outputs.last_hidden_state[:, 0, :]

    return cls_embedding.squeeze(0).cpu().numpy()


# =========================================================
# BUILD NEWS EMBEDDINGS
# =========================================================


def build_news_embeddings():

    all_news = []

    for ticker in TICKERS:

        news_df = fetch_news_for_ticker(ticker)

        if not news_df.empty:
            all_news.append(news_df)

    if len(all_news) == 0:

        return pd.DataFrame(
            columns=["datadate", "tic", "embedding"]
        )

    news_df = pd.concat(all_news)

    embeddings = []

    grouped = news_df.groupby([
        "datadate",
        "tic"
    ])

    for (date, ticker), group in grouped:

        headline_embeddings = []

        for headline in group["headline"]:

            emb = generate_finbert_embedding(headline)

            headline_embeddings.append(emb)

        daily_embedding = np.mean(
            headline_embeddings,
            axis=0
        )

        embeddings.append({
            "datadate": date,
            "tic": ticker,
            "embedding": daily_embedding
        })

    return pd.DataFrame(embeddings)


# =========================================================
# FINAL DATASET
# =========================================================


def build_final_dataset():

    price_df = download_price_data()

    price_df = add_technical_indicators(price_df)

    news_embedding_df = build_news_embeddings()

    final_df = price_df.merge(
        news_embedding_df,
        on=["datadate", "tic"],
        how="left"
    )

    zero_embedding = np.zeros(
        768,
        dtype=np.float32
    )

    final_df["embedding"] = final_df[
        "embedding"
    ].apply(
        lambda value: (
            value
            if isinstance(value, np.ndarray)
            else zero_embedding
        )
    )

    final_df.to_pickle(CACHED_DATASET_FILE)

    print(
        f"Saved dataset to {CACHED_DATASET_FILE}",
        flush=True
    )

    return final_df


# =========================================================
# POSITIONAL ENCODING
# =========================================================


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=500):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(
            0,
            max_len
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2)
            * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


# =========================================================
# PRICE ENCODER
# =========================================================


class PriceEncoder(nn.Module):

    def __init__(
        self,
        input_dim,
        d_model,
        n_heads,
        n_layers
    ):

        super().__init__()

        self.input_projection = nn.Linear(
            input_dim,
            d_model
        )

        self.position = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers
        )

    def forward(self, x):

        x = self.input_projection(x)

        x = self.position(x)

        x = self.transformer(x)

        return x


# =========================================================
# NEWS ENCODER
# =========================================================


class NewsEncoder(nn.Module):

    def __init__(
        self,
        input_dim=768,
        d_model=128
    ):

        super().__init__()

        self.projection = nn.Linear(
            input_dim,
            d_model
        )

    def forward(self, x):

        return self.projection(x)


# =========================================================
# CROSS ATTENTION
# =========================================================


class CrossAttentionFusion(nn.Module):

    def __init__(self, d_model, n_heads):

        super().__init__()

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            batch_first=True
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        price_features,
        news_features
    ):

        attended, attention_weights = (
            self.cross_attention(
                query=price_features,
                key=news_features,
                value=news_features
            )
        )

        fused = self.layer_norm(
            price_features + attended
        )

        return fused, attention_weights


# =========================================================
# MULTIMODAL TRANSFORMER
# =========================================================


class MultiModalTransformer(nn.Module):

    def __init__(
        self,
        price_input_dim,
        d_model=128,
        n_heads=4,
        n_layers=2,
        latent_dim=128
    ):

        super().__init__()

        self.price_encoder = PriceEncoder(
            input_dim=price_input_dim,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers
        )

        self.news_encoder = NewsEncoder(
            input_dim=768,
            d_model=d_model
        )

        self.cross_attention = CrossAttentionFusion(
            d_model=d_model,
            n_heads=n_heads
        )

        self.output_head = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )

        self.return_prediction = nn.Linear(
            latent_dim,
            1
        )

    def forward(
        self,
        price_sequence,
        news_sequence
    ):

        price_sequence = torch.nan_to_num(
            price_sequence,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        news_sequence = torch.nan_to_num(
            news_sequence,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        price_features = self.price_encoder(
            price_sequence
        )

        news_features = self.news_encoder(
            news_sequence
        )

        fused, attention_weights = (
            self.cross_attention(
                price_features,
                news_features
            )
        )

        pooled = fused.mean(dim=1)

        latent_embedding = self.output_head(
            pooled
        )

        latent_embedding = torch.nan_to_num(
            latent_embedding,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        prediction = self.return_prediction(
            latent_embedding
        )

        prediction = torch.nan_to_num(
            prediction,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        return (
            latent_embedding,
            prediction,
            attention_weights
        )


# =========================================================
# DATASET
# =========================================================


class MarketDataset(Dataset):

    def __init__(
        self,
        df,
        ticker,
        window_size
    ):

        self.window_size = window_size

        self.df = df[
            df.tic == ticker
        ].sort_values("datadate")

        raw_price_data = self.df[
            PRICE_FEATURES
        ].values.astype(np.float32)

        raw_price_data = np.nan_to_num(
            raw_price_data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        feature_mean = raw_price_data.mean(
            axis=0,
            keepdims=True
        )

        feature_std = raw_price_data.std(
            axis=0,
            keepdims=True
        )

        feature_std = np.where(
            feature_std < 1e-8,
            1.0,
            feature_std
        )

        self.price_data = (
            (raw_price_data - feature_mean) /
            feature_std
        ).astype(np.float32)

        self.price_data = np.nan_to_num(
            self.price_data,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        self.news_data = self.df[
            "embedding"
        ].values

        self.news_data = np.array([
            np.nan_to_num(
                np.asarray(embedding, dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0
            )
            for embedding in self.news_data
        ])

        # =====================================================
        # FIX 4: CLEAN TARGET RETURNS
        # =====================================================

        returns = (
            self.df["adjcp"]
            .pct_change()
            .shift(-1)
        )

        returns = returns.replace(
            [np.inf, -np.inf],
            np.nan
        )

        returns = returns.fillna(0)

        returns = returns.clip(-0.2, 0.2)

        self.targets = returns.values.astype(np.float32)

    def __len__(self):

        return (
            len(self.df)
            - self.window_size
        )

    def __getitem__(self, idx):

        price_window = self.price_data[
            idx:idx+self.window_size
        ]

        news_window = np.stack(
            self.news_data[
                idx:idx+self.window_size
            ]
        ).astype(np.float32)

        price_window = np.nan_to_num(
            price_window,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        news_window = np.nan_to_num(
            news_window,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        target = self.targets[
            idx+self.window_size
        ]

        return (
            torch.tensor(price_window),
            torch.tensor(news_window),
            torch.tensor(target)
        )


# =========================================================
# PRETRAIN TRANSFORMER
# =========================================================


def pretrain_transformer(
    model,
    dataloader,
    epochs=PRETRAIN_EPOCHS
):

    # =====================================================
    # FIX 6: LOWER LEARNING RATE
    # =====================================================

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-5
    )

    criterion = nn.MSELoss()

    model.train()

    progress_file = OUTPUT_DIR / "multimodal_pretrain_progress.csv"

    if progress_file.exists():
        progress_file.unlink()

    for epoch in range(epochs):

        epoch_loss = 0
        valid_batches = 0

        print(
            f"Transformer epoch {epoch+1}/{epochs}",
            flush=True
        )

        for batch_number, (
            price_batch,
            news_batch,
            targets
        ) in enumerate(dataloader, start=1):

            price_batch = price_batch.to(DEVICE)
            news_batch = news_batch.to(DEVICE)
            targets = targets.to(DEVICE)

            (
                latent_embedding,
                prediction,
                attention_weights
            ) = model(
                price_batch,
                news_batch
            )

            loss = criterion(
                prediction.squeeze(),
                targets
            )

            # =================================================
            # FIX 7: SKIP NaN LOSSES
            # =================================================

            if torch.isnan(loss):

                print(
                    "NaN loss detected — skipping batch",
                    flush=True
                )

                continue

            optimizer.zero_grad()

            loss.backward()

            # =================================================
            # FIX 8: GRADIENT CLIPPING
            # =================================================

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            epoch_loss += loss.item()
            valid_batches += 1

            if (
                batch_number == 1 or
                batch_number % 10 == 0 or
                batch_number == len(dataloader)
            ):

                print(
                    f"Epoch {epoch+1} "
                    f"Batch {batch_number}/{len(dataloader)} "
                    f"Loss {loss.item():.6f}",
                    flush=True
                )

        average_loss = (
            epoch_loss / valid_batches
            if valid_batches > 0
            else float("nan")
        )

        print(
            f"Epoch {epoch+1} Average Loss: "
            f"{average_loss:.6f}",
            flush=True
        )

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "average_loss": average_loss,
            },
            OUTPUT_DIR / "multimodal_transformer_latest.pt"
        )

        pd.DataFrame([{
            "epoch": epoch + 1,
            "average_loss": average_loss,
            "valid_batches": valid_batches,
            "total_batches": len(dataloader),
        }]).to_csv(
            progress_file,
            mode="a",
            index=False,
            header=not progress_file.exists()
        )


# =========================================================
# RL ENVIRONMENT
# =========================================================


class EmbeddingTradingEnv(gym.Env):

    def __init__(
        self,
        transformer,
        dataset
    ):

        super().__init__()

        self.transformer = transformer
        self.dataset = dataset
        self.current_step = 0
        self.portfolio_value = INITIAL_BALANCE
        self.previous_position = 0.0
        self.asset_memory = [INITIAL_BALANCE]

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(128,),
            dtype=np.float32
        )

    def get_embedding(self):

        (
            price_window,
            news_window,
            target
        ) = self.dataset[self.current_step]

        price_window = (
            price_window
            .unsqueeze(0)
            .to(DEVICE)
        )

        news_window = (
            news_window
            .unsqueeze(0)
            .to(DEVICE)
        )

        with torch.no_grad():

            embedding, _, _ = self.transformer(
                price_window,
                news_window
            )

        obs = embedding.squeeze(0).cpu().numpy().astype(np.float32)

        return np.nan_to_num(
            obs,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

    def reset(self, seed=None, options=None):

        self.current_step = 0
        self.portfolio_value = INITIAL_BALANCE
        self.previous_position = 0.0
        self.asset_memory = [INITIAL_BALANCE]

        obs = self.get_embedding()

        return obs, {}

    def step(self, action):

        current_return = float(
            self.dataset.targets[
                self.current_step + self.dataset.window_size
            ]
        )

        position = float(np.clip(action[0], -1, 1))

        turnover = abs(position - self.previous_position)

        strategy_return = (
            position * current_return -
            turnover * TRANSACTION_FEE_PERCENT
        )

        self.portfolio_value *= max(
            0.0,
            1.0 + strategy_return
        )

        self.asset_memory.append(self.portfolio_value)

        reward = strategy_return

        self.previous_position = position

        self.current_step += 1

        done = (
            self.current_step
            >= len(self.dataset)-1
        )

        obs = self.get_embedding()

        return obs, reward, done, False, {}


# =========================================================
# ENSEMBLE EVALUATION
# =========================================================


def calculate_sharpe(asset_memory):

    returns = pd.Series(asset_memory).pct_change().dropna()

    if len(returns) == 0 or returns.std() == 0:
        return -np.inf

    return np.sqrt(252) * returns.mean() / returns.std()


def train_rl_model(model_name, env, timesteps):

    if model_name == "A2C":
        model = A2C(
            "MlpPolicy",
            env,
            verbose=0,
            seed=42
        )

    elif model_name == "PPO":
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            seed=42
        )

    elif model_name == "DDPG":
        n_actions = env.action_space.shape[-1]

        action_noise = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.2 * np.ones(n_actions)
        )

        model = DDPG(
            "MlpPolicy",
            env,
            action_noise=action_noise,
            verbose=0,
            seed=42
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


def validate_rl_model(model, env):

    obs, _ = env.reset()
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

    return calculate_sharpe(env.asset_memory), env.asset_memory[-1]


def build_ticker_dataset(df, ticker, start_date, end_date):

    sliced_df = df[
        (df.datadate >= start_date) &
        (df.datadate < end_date)
    ]

    if sliced_df.empty:
        return None

    dataset = MarketDataset(
        sliced_df,
        ticker=ticker,
        window_size=WINDOW_SIZE
    )

    if len(dataset) <= 1:
        return None

    return dataset


def run_multimodal_ensemble(
    transformer,
    df,
    ticker,
    a2c_timesteps,
    ppo_timesteps,
    ddpg_timesteps
):

    ticker_dates = (
        df[df.tic == ticker]
        .sort_values("datadate")
        .datadate
        .unique()
    )

    test_start = ticker_dates[-TEST_WINDOW]
    test_end = ticker_dates[-1]

    rolling_points = list(range(
        WINDOW_SIZE + VALIDATION_WINDOW + REBALANCE_WINDOW,
        len(ticker_dates) - TEST_WINDOW,
        REBALANCE_WINDOW
    ))

    summary = []
    summary_file = OUTPUT_DIR / "multimodal_ensemble_summary.csv"

    if summary_file.exists():
        summary_file.unlink()

    model_timesteps = {
        "A2C": a2c_timesteps,
        "PPO": ppo_timesteps,
        "DDPG": ddpg_timesteps,
    }

    print(
        f"Running multimodal ensemble over "
        f"{len(rolling_points)} rolling windows",
        flush=True
    )

    print(
        f"Holding out final {TEST_WINDOW} trading days for testing: "
        f"{test_start} -> {test_end}",
        flush=True
    )

    for window_number, i in enumerate(rolling_points, start=1):

        validation_start_index = (
            i - VALIDATION_WINDOW - REBALANCE_WINDOW
        )

        validation_context_index = max(
            0,
            validation_start_index - WINDOW_SIZE
        )

        validation_start = ticker_dates[validation_start_index]
        validation_context_start = ticker_dates[validation_context_index]
        validation_end = ticker_dates[i - REBALANCE_WINDOW]
        trade_end = ticker_dates[i]

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

        train_dataset = build_ticker_dataset(
            df,
            ticker,
            ticker_dates[0],
            validation_start
        )

        validation_dataset = build_ticker_dataset(
            df,
            ticker,
            validation_context_start,
            validation_end
        )

        if train_dataset is None or validation_dataset is None:
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
                lambda dataset=train_dataset: EmbeddingTradingEnv(
                    transformer,
                    dataset
                )
            ])

            model = train_rl_model(
                model_name,
                train_env,
                model_timesteps[model_name]
            )

            validation_env = EmbeddingTradingEnv(
                transformer,
                validation_dataset
            )

            sharpe, final_value = validate_rl_model(
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
        description="Run multimodal transformer + PPO trading experiment."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=PRETRAIN_EPOCHS,
        help="Number of transformer pretraining epochs."
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
        "--ticker",
        default="AAPL",
        help="Ticker to use for the embedding-based RL ensemble."
    )

    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Rebuild the price/news dataset instead of using cache."
    )

    args = parser.parse_args()

    print(
        f"Running from {START_DATE} to {END_DATE}",
        flush=True
    )

    if CACHED_DATASET_FILE.exists() and not args.refresh_data:
        print(
            f"Using cached dataset: {CACHED_DATASET_FILE}",
            flush=True
        )

        final_df = pd.read_pickle(CACHED_DATASET_FILE)
    else:
        final_df = build_final_dataset()

    dataset = MarketDataset(
        final_df,
        ticker=args.ticker,
        window_size=WINDOW_SIZE
    )

    dataloader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True
    )

    model = MultiModalTransformer(
        price_input_dim=len(PRICE_FEATURES)
    ).to(DEVICE)

    print("Pretraining transformer...", flush=True)

    pretrain_transformer(
        model,
        dataloader,
        epochs=args.epochs
    )

    print("Starting multimodal RL ensemble...", flush=True)

    ensemble_summary = run_multimodal_ensemble(
        model,
        final_df,
        args.ticker,
        args.a2c_timesteps,
        args.timesteps,
        args.ddpg_timesteps
    )

    print(ensemble_summary)

    print("Training completed", flush=True)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yfinance as yf

from stable_baselines3 import A2C, DDPG, PPO

from experiments import MultimodalTransformer as multimodal


MODEL_LOADERS = {
    "A2C": A2C,
    "PPO": PPO,
    "DDPG": DDPG,
}


def infer_algorithm(model_path):

    algorithm = Path(model_path).stem.upper()

    if algorithm not in MODEL_LOADERS:
        raise ValueError(
            "Could not infer algorithm from model filename. "
            "Use a model named A2C.zip, PPO.zip, or DDPG.zip, "
            "or pass --algorithm."
        )

    return algorithm


def download_recent_prices(ticker, end_date, lookback_days):

    end_timestamp = pd.to_datetime(end_date)
    start_timestamp = end_timestamp - pd.Timedelta(days=lookback_days)

    df = yf.download(
        ticker,
        start=start_timestamp.strftime("%Y-%m-%d"),
        end=(end_timestamp + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise RuntimeError(f"No yfinance price data found for {ticker}.")

    df = df.reset_index()
    df["tic"] = ticker

    df = df.rename(columns={
        "Date": "datadate",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adjcp",
        "Volume": "volume",
    })

    df["datadate"] = pd.to_datetime(df["datadate"]).dt.normalize()

    return df


def fetch_recent_news(ticker, start_date, end_date):

    news = multimodal.finnhub_client.company_news(
        ticker,
        _from=pd.to_datetime(start_date).strftime("%Y-%m-%d"),
        to=pd.to_datetime(end_date).strftime("%Y-%m-%d")
    )

    records = []

    for article in news:

        headline = article.get("headline", "")
        timestamp = article.get("datetime", None)

        if timestamp is None or len(headline.strip()) == 0:
            continue

        records.append({
            "datadate": pd.to_datetime(
                timestamp,
                unit="s"
            ).normalize(),
            "tic": ticker,
            "headline": headline,
        })

    return pd.DataFrame(records)


def build_recent_news_embeddings(ticker, start_date, end_date):

    news_df = fetch_recent_news(
        ticker,
        start_date,
        end_date
    )

    if news_df.empty:
        return pd.DataFrame(
            columns=["datadate", "tic", "embedding"]
        )

    rows = []

    grouped = news_df.groupby(["datadate", "tic"])

    for (date, symbol), group in grouped:

        headline_embeddings = [
            multimodal.generate_finbert_embedding(headline)
            for headline in group["headline"]
        ]

        rows.append({
            "datadate": date,
            "tic": symbol,
            "embedding": np.mean(headline_embeddings, axis=0),
        })

    return pd.DataFrame(rows)


def build_prediction_dataset(ticker, end_date, lookback_days):

    price_df = download_recent_prices(
        ticker,
        end_date,
        lookback_days
    )

    price_df = multimodal.add_technical_indicators(price_df)

    news_start = price_df["datadate"].min()
    news_end = pd.to_datetime(end_date)

    news_df = build_recent_news_embeddings(
        ticker,
        news_start,
        news_end
    )

    final_df = price_df.merge(
        news_df,
        on=["datadate", "tic"],
        how="left"
    )

    zero_embedding = np.zeros(
        768,
        dtype=np.float32
    )

    final_df["embedding"] = final_df["embedding"].apply(
        lambda value: (
            value
            if isinstance(value, np.ndarray)
            else zero_embedding
        )
    )

    final_df = final_df.sort_values("datadate")

    dataset = multimodal.MarketDataset(
        final_df,
        ticker=ticker,
        window_size=multimodal.WINDOW_SIZE
    )

    if len(dataset) <= 0:
        raise RuntimeError(
            "Not enough cleaned price rows to build a prediction window. "
            f"Need more than {multimodal.WINDOW_SIZE} rows after indicators."
        )

    return final_df, dataset


def load_transformer(checkpoint_path):

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Transformer checkpoint not found: {checkpoint_path}"
        )

    transformer = multimodal.MultiModalTransformer(
        price_input_dim=len(multimodal.PRICE_FEATURES)
    ).to(multimodal.DEVICE)

    state = torch.load(
        checkpoint_path,
        map_location=multimodal.DEVICE
    )

    transformer.load_state_dict(state["model_state_dict"])
    transformer.eval()

    return transformer


def get_latest_observation(transformer, dataset):

    price_window, news_window, target = dataset[len(dataset) - 1]

    price_window = price_window.unsqueeze(0).to(multimodal.DEVICE)
    news_window = news_window.unsqueeze(0).to(multimodal.DEVICE)

    with torch.no_grad():
        embedding, predicted_return, attention = transformer(
            price_window,
            news_window
        )

    observation = embedding.squeeze(0).cpu().numpy().astype(np.float32)
    observation = np.nan_to_num(
        observation,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    return observation, float(predicted_return.squeeze().cpu().item())


def action_to_signal(action_value, buy_threshold, sell_threshold):

    if action_value >= buy_threshold:
        return "BUY"

    if action_value <= sell_threshold:
        return "SELL"

    return "HOLD"


def parse_tickers(ticker_argument):

    ticker_argument = ticker_argument.strip()

    if ticker_argument.upper() == "ALL":
        return list(multimodal.TICKERS)

    tickers = [
        ticker.strip().upper()
        for ticker in ticker_argument.split(",")
        if ticker.strip()
    ]

    if len(tickers) == 0:
        raise ValueError("No tickers provided.")

    return tickers


def predict_ticker(
    ticker,
    model,
    model_path,
    algorithm,
    transformer,
    args
):

    final_df, dataset = build_prediction_dataset(
        ticker=ticker,
        end_date=args.date,
        lookback_days=args.lookback_days
    )

    observation, transformer_return = get_latest_observation(
        transformer,
        dataset
    )

    action, _ = model.predict(
        observation,
        deterministic=True
    )

    action_value = float(np.asarray(action).reshape(-1)[0])

    signal = action_to_signal(
        action_value,
        args.buy_threshold,
        args.sell_threshold
    )

    latest_row = final_df.iloc[-1]

    return {
        "Ticker": ticker,
        "Signal": signal,
        "RL Action": round(action_value, 4),
        "Transformer Return": round(transformer_return, 6),
        "Latest Price Date": latest_row["datadate"].date(),
        "Latest Adj Close": round(float(latest_row["adjcp"]), 2),
        "Algorithm": algorithm,
        "Model Path": str(model_path),
    }


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Load a saved multimodal RL model and predict a "
            "BUY/SELL/HOLD signal from recent prices and news."
        )
    )

    parser.add_argument(
        "--model-path",
        default="models/multimodal/window_180/A2C.zip",
        help="Path to saved A2C/PPO/DDPG multimodal model."
    )

    parser.add_argument(
        "--algorithm",
        choices=["A2C", "PPO", "DDPG"],
        default=None,
        help="Algorithm type. If omitted, inferred from model filename."
    )

    parser.add_argument(
        "--ticker",
        default="ALL",
        help=(
            "Ticker to predict for. Use ALL for the study tickers, "
            "or pass a comma list like AAPL,MSFT."
        )
    )

    parser.add_argument(
        "--date",
        default=datetime.now().strftime("%Y-%m-%d"),
        help="Prediction date. Uses latest available daily price up to this date."
    )

    parser.add_argument(
        "--lookback-days",
        type=int,
        default=120,
        help="Calendar days of price/news history to fetch."
    )

    parser.add_argument(
        "--transformer-checkpoint",
        default=str(multimodal.OUTPUT_DIR / "multimodal_transformer_latest.pt"),
        help="Path to trained multimodal transformer checkpoint."
    )

    parser.add_argument(
        "--buy-threshold",
        type=float,
        default=0.10,
        help="Action value at or above this becomes BUY."
    )

    parser.add_argument(
        "--sell-threshold",
        type=float,
        default=-0.10,
        help="Action value at or below this becomes SELL."
    )

    parser.add_argument(
        "--output-path",
        default="outputs/multimodal_daily_predictions.csv",
        help="CSV path where all predictions are saved."
    )

    args = parser.parse_args()

    model_path = Path(args.model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"RL model not found: {model_path}")

    algorithm = args.algorithm or infer_algorithm(model_path)
    model = MODEL_LOADERS[algorithm].load(model_path)

    transformer = load_transformer(
        Path(args.transformer_checkpoint)
    )

    tickers = parse_tickers(args.ticker)
    rows = []

    print("\n================================================")
    print("MULTIMODAL DAILY PREDICTIONS")
    print("================================================")
    print(f"Model: {model_path}")
    print(f"Algorithm: {algorithm}")
    print(f"Prediction date requested: {args.date}")
    print(f"Tickers: {', '.join(tickers)}")
    print("================================================")

    for ticker in tickers:

        print(f"Predicting {ticker}...", flush=True)

        try:
            rows.append(
                predict_ticker(
                    ticker,
                    model,
                    model_path,
                    algorithm,
                    transformer,
                    args
                )
            )

        except Exception as error:
            rows.append({
                "Ticker": ticker,
                "Signal": "ERROR",
                "RL Action": np.nan,
                "Transformer Return": np.nan,
                "Latest Price Date": "",
                "Latest Adj Close": np.nan,
                "Algorithm": algorithm,
                "Model Path": str(model_path),
                "Error": str(error),
            })

    predictions_df = pd.DataFrame(rows)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(output_path, index=False)

    display_columns = [
        "Ticker",
        "Signal",
        "RL Action",
        "Transformer Return",
        "Latest Price Date",
        "Latest Adj Close",
    ]

    print()
    print(predictions_df[display_columns].to_string(index=False))
    print(f"\nSaved predictions to {output_path}")
    print("================================================\n")


if __name__ == "__main__":
    main()

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from pymongo import MongoClient
from backend.metrics import instrument_app

try:
    from flask_cors import CORS
except ImportError:
    CORS = None

try:
    import finnhub
except ImportError:
    finnhub = None

from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, CCIIndicator, MACD


# =========================================================
# CONFIG
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")
OUTPUT_DIR = BASE_DIR / "outputs"
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "multimodal" / "window_60" / "A2C.zip"
DEFAULT_TRANSFORMER_PATH = OUTPUT_DIR / "multimodal_transformer_latest.pt"

TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
]

SUMMARY_FILES = {
    "baseline": OUTPUT_DIR / "ensemble_summary.csv",
    "bert": OUTPUT_DIR / "bert_ensemble_summary.csv",
    "multimodal": OUTPUT_DIR / "multimodal_ensemble_summary.csv",
}

BENCHMARK_RESULTS_PATH = OUTPUT_DIR / "benchmark_comparison_results.csv"
CURVES_PATH = OUTPUT_DIR / "benchmark_equity_drawdown_curves.csv"
PREDICTIONS_PATH = OUTPUT_DIR / "multimodal_daily_predictions.csv"

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
MONGODB_URI = os.getenv("MONGODB_URI")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "mlops_repe")

app = Flask(__name__)
instrument_app(app)

if CORS is not None:
    CORS(app)


@app.after_request
def add_cors_headers(response):

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type, Authorization"
    )
    response.headers["Access-Control-Allow-Methods"] = (
        "GET, POST, OPTIONS"
    )

    return response


_prediction_module = None
_model_cache = {}
_transformer_cache = {}
_mongo_client = None
_mongo_database = None


# =========================================================
# JSON HELPERS
# =========================================================

def clean_value(value):

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if np.isnan(value) or np.isinf(value):
            return None
        return float(value)

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d")

    if pd.isna(value):
        return None

    return value


def frame_to_records(df):

    return [
        {
            column: clean_value(value)
            for column, value in row.items()
        }
        for row in df.to_dict(orient="records")
    ]


def get_payload():

    if request.is_json:
        return request.get_json(silent=True) or {}

    payload = request.form.to_dict()
    payload.update(request.args.to_dict())

    return payload


def parse_tickers(value):

    if value is None or str(value).strip().upper() == "ALL":
        return TICKERS

    tickers = [
        ticker.strip().upper()
        for ticker in str(value).split(",")
        if ticker.strip()
    ]

    return tickers or TICKERS


def read_csv(path):

    if not path.exists():
        return pd.DataFrame()

    return pd.read_csv(path)


def get_mongo_database():

    global _mongo_client, _mongo_database

    if not MONGODB_URI:
        return None

    if _mongo_database is None:
        _mongo_client = MongoClient(
            MONGODB_URI,
            serverSelectionTimeoutMS=3000,
        )
        _mongo_client.admin.command("ping")
        _mongo_database = _mongo_client[MONGODB_DATABASE]

    return _mongo_database


def read_stored_market(ticker, lookback_days):

    database = get_mongo_database()
    if database is None:
        return None

    start_date = datetime.utcnow() - pd.Timedelta(days=int(lookback_days))
    rows = list(
        database.market_data.find(
            {
                "ticker": ticker,
                "trading_date": {"$gte": start_date},
            },
            {"_id": 0},
        ).sort("trading_date", 1)
    )

    if not rows:
        return None

    history = []
    for row in rows:
        indicators = row.get("indicators", {})
        history.append({
            "date": row.get("trading_date"),
            "open": row.get("open"),
            "high": row.get("high"),
            "low": row.get("low"),
            "close": row.get("close"),
            "adj_close": row.get("adj_close"),
            "volume": row.get("volume"),
            "macd": indicators.get("macd"),
            "rsi": indicators.get("rsi"),
            "cci": indicators.get("cci"),
            "adx": indicators.get("adx"),
            "daily_return": None,
            "volatility_15d": None,
            "sma_15": None,
            "sma_30": None,
        })

    return history


def read_stored_news(ticker, days):

    database = get_mongo_database()
    if database is None:
        return None

    start_date = datetime.utcnow() - pd.Timedelta(days=int(days))
    rows = database.news.find(
        {
            "ticker": ticker,
            "published_at": {"$gte": start_date},
        },
        {"_id": 0},
    ).sort("published_at", -1).limit(100)

    return [
        {
            "datetime": row.get("published_at"),
            "headline": row.get("headline"),
            "source": row.get("source"),
            "url": row.get("url"),
            "summary": row.get("summary"),
        }
        for row in rows
    ]


# =========================================================
# MARKET DATA
# =========================================================

def download_market_data(ticker, lookback_days):

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=int(lookback_days))

    df = yf.download(
        ticker,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        raise RuntimeError(f"No yfinance data found for {ticker}.")

    df = df.reset_index()

    df = df.rename(columns={
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume",
    })

    df["date"] = pd.to_datetime(df["date"]).dt.normalize()

    return df


def add_dashboard_indicators(df):

    df = df.copy()

    df["macd"] = MACD(
        close=df["adj_close"]
    ).macd()

    df["rsi"] = RSIIndicator(
        close=df["adj_close"]
    ).rsi()

    df["cci"] = CCIIndicator(
        high=df["high"],
        low=df["low"],
        close=df["adj_close"]
    ).cci()

    df["adx"] = ADXIndicator(
        high=df["high"],
        low=df["low"],
        close=df["adj_close"]
    ).adx()

    df["daily_return"] = df["adj_close"].pct_change()
    df["volatility_15d"] = (
        df["daily_return"].rolling(15).std() *
        np.sqrt(252)
    )
    df["sma_15"] = df["adj_close"].rolling(15).mean()
    df["sma_30"] = df["adj_close"].rolling(30).mean()

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def fetch_news(ticker, days):

    if finnhub is None or not FINNHUB_API_KEY:
        return []

    client = finnhub.Client(api_key=FINNHUB_API_KEY)

    end_date = pd.Timestamp.today().normalize()
    start_date = end_date - pd.Timedelta(days=int(days))

    articles = client.company_news(
        ticker,
        _from=start_date.strftime("%Y-%m-%d"),
        to=end_date.strftime("%Y-%m-%d")
    )

    rows = []

    for article in articles:
        rows.append({
            "datetime": pd.to_datetime(
                article.get("datetime"),
                unit="s"
            ).strftime("%Y-%m-%d")
            if article.get("datetime") is not None
            else None,
            "headline": article.get("headline"),
            "source": article.get("source"),
            "url": article.get("url"),
            "summary": article.get("summary"),
        })

    return rows


# =========================================================
# PREDICTION
# =========================================================

def load_prediction_module():

    global _prediction_module

    if _prediction_module is None:
        from backend import MultiModal_Prediction as predictor

        _prediction_module = predictor

    return _prediction_module


def load_cached_model(predictor, model_path, algorithm):

    cache_key = (str(model_path), algorithm)

    if cache_key not in _model_cache:
        _model_cache[cache_key] = predictor.MODEL_LOADERS[
            algorithm
        ].load(model_path)

    return _model_cache[cache_key]


def load_cached_transformer(predictor, checkpoint_path):

    cache_key = str(checkpoint_path)

    if cache_key not in _transformer_cache:
        _transformer_cache[cache_key] = predictor.load_transformer(
            checkpoint_path
        )

    return _transformer_cache[cache_key]


def run_predictions(payload):

    predictor = load_prediction_module()

    model_path = Path(
        payload.get("model_path", DEFAULT_MODEL_PATH)
    )

    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path

    checkpoint_path = Path(
        payload.get("transformer_checkpoint", DEFAULT_TRANSFORMER_PATH)
    )

    if not checkpoint_path.is_absolute():
        checkpoint_path = BASE_DIR / checkpoint_path

    algorithm = payload.get("algorithm")

    if algorithm is None or algorithm == "":
        algorithm = predictor.infer_algorithm(model_path)

    algorithm = algorithm.upper()

    model = load_cached_model(
        predictor,
        model_path,
        algorithm
    )

    transformer = load_cached_transformer(
        predictor,
        checkpoint_path
    )

    tickers = parse_tickers(payload.get("ticker", "ALL"))

    class Args:
        pass

    args = Args()
    args.date = payload.get(
        "date",
        datetime.now().strftime("%Y-%m-%d")
    )
    args.lookback_days = int(payload.get("lookback_days", 120))
    args.buy_threshold = float(payload.get("buy_threshold", 0.10))
    args.sell_threshold = float(payload.get("sell_threshold", -0.10))

    rows = []

    for ticker in tickers:
        try:
            rows.append(
                predictor.predict_ticker(
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
                "RL Action": None,
                "Transformer Return": None,
                "Latest Price Date": None,
                "Latest Adj Close": None,
                "Algorithm": algorithm,
                "Model Path": str(model_path),
                "Error": str(error),
            })

    predictions_df = pd.DataFrame(rows)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    return predictions_df


# =========================================================
# ROUTES
# =========================================================

@app.get("/api/health")
def health():

    return jsonify({
        "status": "ok",
        "tickers": TICKERS,
        "default_model_path": str(DEFAULT_MODEL_PATH),
        "default_transformer_path": str(DEFAULT_TRANSFORMER_PATH),
    })


@app.get("/api/dashboard")
def dashboard_snapshot():

    predictions = read_csv(PREDICTIONS_PATH)
    benchmark = read_csv(BENCHMARK_RESULTS_PATH)

    summaries = {
        name: frame_to_records(read_csv(path).tail(1))
        for name, path in SUMMARY_FILES.items()
    }

    return jsonify({
        "predictions": frame_to_records(predictions),
        "benchmark": frame_to_records(benchmark),
        "summaries": summaries,
    })


@app.route("/api/predict", methods=["GET", "POST"])
def predict():

    if not FINNHUB_API_KEY:
        return jsonify({
            "error": "FINNHUB_API_KEY is not configured. Set it in .env or the environment."
        }), 503

    payload = get_payload()
    predictions_df = run_predictions(payload)

    return jsonify({
        "rows": frame_to_records(predictions_df),
        "saved_to": str(PREDICTIONS_PATH),
    })


@app.get("/api/market/<ticker>")
def market(ticker):

    lookback_days = int(request.args.get("lookback_days", 120))
    include_news = request.args.get("include_news", "true").lower() != "false"
    news_days = int(request.args.get("news_days", 7))
    source = request.args.get("source", "mongo").lower()

    if source == "mongo":
        try:
            stored_history = read_stored_market(ticker.upper(), lookback_days)
            if stored_history:
                stored_news = (
                    read_stored_news(ticker.upper(), news_days)
                    if include_news
                    else []
                )
                return jsonify({
                    "ticker": ticker.upper(),
                    "source": "mongodb",
                    "latest": stored_history[-1],
                    "history": stored_history[-90:],
                    "news": stored_news or [],
                })
        except Exception as error:
            app.logger.warning("MongoDB read failed; using live data: %s", error)

    price_df = download_market_data(
        ticker.upper(),
        lookback_days
    )

    price_df = add_dashboard_indicators(price_df)

    latest = price_df.iloc[-1].to_dict()

    return jsonify({
        "ticker": ticker.upper(),
        "source": "yfinance",
        "latest": {
            key: clean_value(value)
            for key, value in latest.items()
        },
        "history": frame_to_records(price_df.tail(90)),
        "news": fetch_news(ticker.upper(), news_days)
        if include_news
        else [],
    })


@app.get("/api/news/<ticker>")
def news(ticker):

    days = int(request.args.get("days", 7))

    try:
        stored_news = read_stored_news(ticker.upper(), days)
        if stored_news:
            return jsonify({
                "ticker": ticker.upper(),
                "source": "mongodb",
                "news": stored_news,
            })
    except Exception as error:
        app.logger.warning("MongoDB news read failed; using live data: %s", error)

    return jsonify({
        "ticker": ticker.upper(),
        "source": "finnhub",
        "news": fetch_news(ticker.upper(), days),
    })


@app.get("/api/ingestion/status")
def ingestion_status():

    try:
        database = get_mongo_database()
        if database is None:
            return jsonify({"error": "MONGODB_URI is not configured."}), 503
        run = database.ingestion_runs.find_one(
            {}, {"_id": 0}, sort=[("started_at", -1)]
        )
        return jsonify({"source": "mongodb", "latest_run": run})
    except Exception as error:
        return jsonify({"error": str(error)}), 503


@app.get("/api/summaries")
def summaries():

    return jsonify({
        name: frame_to_records(read_csv(path))
        for name, path in SUMMARY_FILES.items()
    })


@app.get("/api/benchmark")
def benchmark():

    return jsonify({
        "results": frame_to_records(read_csv(BENCHMARK_RESULTS_PATH)),
        "curves": frame_to_records(read_csv(CURVES_PATH)),
    })


@app.get("/api/files")
def files():

    return jsonify({
        "summaries": {
            name: str(path)
            for name, path in SUMMARY_FILES.items()
        },
        "benchmark_results": str(BENCHMARK_RESULTS_PATH),
        "benchmark_curves": str(CURVES_PATH),
        "predictions": str(PREDICTIONS_PATH),
    })


if __name__ == "__main__":
    app.run(
        host=os.getenv("BACKEND_HOST", "0.0.0.0"),
        port=int(os.getenv("BACKEND_PORT", "5000")),
        debug=os.getenv("FLASK_DEBUG", "0") == "1"
    )

from __future__ import annotations

import argparse
import hashlib
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import finnhub
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from pymongo import MongoClient, UpdateOne
from pymongo.collection import Collection
from ta.momentum import RSIIndicator
from ta.trend import ADXIndicator, CCIIndicator, MACD

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META"]
DEFAULT_DATABASE = "mlops_repe"
PROFILE_DEFAULTS = {
    "bootstrap": {"lookback_days": 120, "news_days": 7},
    "nightly": {"lookback_days": 5, "news_days": 1},
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def finite_value(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    numeric = float(value)
    return numeric if pd.notna(numeric) else None


def download_prices(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    indicator_warmup_start = start_date - timedelta(days=60)
    frame = yf.download(
        ticker,
        start=indicator_warmup_start.strftime("%Y-%m-%d"),
        end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )

    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    if frame.empty:
        return pd.DataFrame()

    frame = frame.reset_index().rename(
        columns={
            "Date": "trading_date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )
    frame["trading_date"] = pd.to_datetime(frame["trading_date"]).dt.date
    frame["macd"] = MACD(close=frame["adj_close"]).macd()
    frame["rsi"] = RSIIndicator(close=frame["adj_close"]).rsi()
    frame["cci"] = CCIIndicator(
        high=frame["high"], low=frame["low"], close=frame["adj_close"]
    ).cci()
    frame["adx"] = ADXIndicator(
        high=frame["high"], low=frame["low"], close=frame["adj_close"]
    ).adx()
    return frame[
        frame["trading_date"] >= start_date.date()
    ].reset_index(drop=True)


def market_documents(ticker: str, frame: pd.DataFrame, collected_at: datetime) -> list[dict[str, Any]]:
    documents = []
    for row in frame.to_dict(orient="records"):
        documents.append(
            {
                "ticker": ticker,
                "trading_date": datetime.combine(row["trading_date"], datetime.min.time(), tzinfo=timezone.utc),
                "open": finite_value(row.get("open")),
                "high": finite_value(row.get("high")),
                "low": finite_value(row.get("low")),
                "close": finite_value(row.get("close")),
                "adj_close": finite_value(row.get("adj_close")),
                "volume": finite_value(row.get("volume")),
                "indicators": {
                    name: finite_value(row.get(name))
                    for name in ("macd", "rsi", "cci", "adx")
                },
                "source": "yfinance",
                "collected_at": collected_at,
            }
        )
    return documents


def news_documents(
    ticker: str,
    client: finnhub.Client,
    start_date: datetime,
    end_date: datetime,
    collected_at: datetime,
) -> list[dict[str, Any]]:
    articles = client.company_news(
        ticker,
        _from=start_date.strftime("%Y-%m-%d"),
        to=end_date.strftime("%Y-%m-%d"),
    )
    documents = []
    for article in articles:
        headline = str(article.get("headline", "")).strip()
        published_timestamp = article.get("datetime")
        if not headline or published_timestamp is None:
            continue
        article_id = str(article.get("id") or hashlib.sha256(
            f"{ticker}:{published_timestamp}:{headline}".encode("utf-8")
        ).hexdigest())
        documents.append(
            {
                "ticker": ticker,
                "article_id": article_id,
                "headline": headline,
                "summary": article.get("summary", ""),
                "source": article.get("source", ""),
                "url": article.get("url", ""),
                "published_at": datetime.fromtimestamp(
                    int(published_timestamp), tz=timezone.utc
                ),
                "collected_at": collected_at,
            }
        )
    return documents


def upsert_documents(collection: Collection, documents: list[dict[str, Any]], key_fields: list[str]) -> int:
    if not documents:
        return 0
    operations = [
        UpdateOne(
            {field: document[field] for field in key_fields},
            {"$set": document},
            upsert=True,
        )
        for document in documents
    ]
    result = collection.bulk_write(operations, ordered=False)
    return result.upserted_count + result.modified_count


def ensure_indexes(database) -> None:
    database.market_data.create_index(
        [("ticker", 1), ("trading_date", 1)], unique=True
    )
    database.news.create_index(
        [("ticker", 1), ("article_id", 1)], unique=True
    )
    database.ingestion_runs.create_index("started_at")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect market data and news into MongoDB Atlas.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--profile", choices=PROFILE_DEFAULTS, default="nightly")
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--news-days", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="Fetch and validate without writing to MongoDB.")
    args = parser.parse_args()
    profile = PROFILE_DEFAULTS[args.profile]
    lookback_days = args.lookback_days or profile["lookback_days"]
    news_days = args.news_days or profile["news_days"]

    collected_at = utc_now()
    end_date = collected_at.replace(hour=0, minute=0, second=0, microsecond=0)
    start_date = end_date - timedelta(days=lookback_days)
    news_start = end_date - timedelta(days=news_days)
    mongo_client = None
    database = None
    run_id = None

    if not args.dry_run:
        mongo_uri = os.getenv("MONGODB_URI")
        if not mongo_uri:
            raise RuntimeError("MONGODB_URI is not configured in .env or the environment.")
        database_name = os.getenv("MONGODB_DATABASE", DEFAULT_DATABASE)
        mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
        mongo_client.admin.command("ping")
        database = mongo_client[database_name]
        ensure_indexes(database)
        run = database.ingestion_runs.insert_one(
            {
                "started_at": collected_at,
                "status": "running",
                "tickers": args.tickers,
                "profile": args.profile,
                "lookback_days": lookback_days,
                "news_days": news_days,
            }
        )
        run_id = run.inserted_id

    news_client = None
    finnhub_key = os.getenv("FINNHUB_API_KEY")
    if finnhub_key:
        news_client = finnhub.Client(api_key=finnhub_key)

    market_count = 0
    news_count = 0
    try:
        for ticker in args.tickers:
            ticker = ticker.upper()
            prices = download_prices(ticker, start_date, end_date)
            market_docs = market_documents(ticker, prices, collected_at)
            market_count += len(market_docs)
            if database is not None:
                upsert_documents(database.market_data, market_docs, ["ticker", "trading_date"])

            if news_client is not None:
                articles = news_documents(ticker, news_client, news_start, end_date, collected_at)
                news_count += len(articles)
                if database is not None:
                    upsert_documents(database.news, articles, ["ticker", "article_id"])
            print(f"{ticker}: {len(market_docs)} market rows; news collected: {news_client is not None}")

        if database is not None:
            database.ingestion_runs.update_one(
                {"_id": run_id},
                {"$set": {
                    "finished_at": utc_now(),
                    "status": "completed",
                    "market_rows": market_count,
                    "news_articles": news_count,
                }},
            )
    except Exception as error:
        if database is not None:
            database.ingestion_runs.update_one(
                {"_id": run_id},
                {"$set": {"finished_at": utc_now(), "status": "failed", "error": str(error)}},
            )
        raise
    finally:
        if mongo_client is not None:
            mongo_client.close()

    mode = "dry-run" if args.dry_run else "MongoDB"
    print(
        f"Completed {mode} ({args.profile}): "
        f"{market_count} market rows and {news_count} news articles."
    )


if __name__ == "__main__":
    main()

# MLOps pipeline

This project now has a DVC pipeline around the yfinance RL baseline and MLflow tracking for each training run.

## Project layout

```text
backend/      Flask API and multimodal prediction serving
experiments/  Baseline, BERT, multimodal, and MLflow training code
ingestion/    MongoDB market/news collector
frontend/     Web application
data/         DVC-managed input data
models/       DVC-managed model artifacts
outputs/      Training and prediction outputs
```

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
dvc init
```

The `dvc init` command is only needed once, and creates the local `.dvc/` metadata directory.

## Run a fast demo

```powershell
python -m experiments.prepare_data --start 2024-01-01 --end 2024-04-30
python -m experiments.train --quick
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

Open the MLflow URL printed by the command, usually `http://127.0.0.1:5000`.

## Run with DVC

```powershell
dvc repro
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

DVC tracks the prepared dataset and training outputs. MLflow records parameters, Sharpe metrics, the selected model, the summary CSV, and trained model artifacts. Use `dvc dag` to show the stage graph and `dvc metrics show` to inspect the generated summary.

The BERT and multimodal scripts remain separate because they require external FinBERT/Finnhub resources and should be promoted to their own DVC stages after their credentials and data contracts are configured.

## MongoDB collection lifecycle

Run the one-time historical bootstrap first:

```powershell
python -m ingestion.collect_market_data --profile bootstrap
```

This loads 120 calendar days of market data and 7 days of news. Schedule the nightly profile afterward:

```powershell
python -m ingestion.collect_market_data --profile nightly
```

The nightly profile collects a short overlap window and upserts by `ticker + trading_date` for prices and `ticker + article_id` for news. That makes reruns safe and lets late data corrections update existing records instead of creating duplicates.

The nightly GitHub Actions job installs only `requirements-collector.txt`. The full `requirements.txt` remains for model training and MLflow, so data ingestion does not install the heavier deep-learning stack.

## Docker

Docker Compose runs the application services with these ports:

```text
Backend:  http://127.0.0.1:5000
Frontend: http://127.0.0.1:5173
MLflow:   http://127.0.0.1:5001
```

Create `.env` in the project root with local credentials, then start the stack:

```powershell
docker compose up --build
```

Run the collector manually in Docker:

```powershell
docker compose --profile collector run --rm collector
```

Models and outputs remain outside the backend image and are mounted at runtime. MongoDB Atlas remains external; Docker does not start a local MongoDB service.

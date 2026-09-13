# MLOps_REPE

An end-to-end stock analytics project with a Flask API, React/TanStack frontend, MongoDB Atlas ingestion, DVC data versioning, MLflow experiment tracking, and Docker Compose services.

## Prerequisites

- Python 3.11 recommended
- Node.js 20 or newer
- Docker Desktop, if using containers
- Git and DVC
- MongoDB Atlas account for ingestion and stored dashboard data
- Finnhub API key for news and multimodal predictions

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

## Clone And Configure

```powershell
git clone https://github.com/satts27/MLOps_REPE.git
cd MLOps_REPE
```

Create the local environment file from the template:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and provide local credentials:

```env
FINNHUB_API_KEY=your_finnhub_api_key
MONGODB_URI=mongodb+srv://username:password@host/
MONGODB_DATABASE=mlops_repe
```

Never commit `.env`. It is ignored by Git. Add the same values as GitHub repository secrets named `FINNHUB_API_KEY` and `MONGODB_URI` for Actions.

## Python Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

The full requirements file is for training and MLflow. For ingestion only, install the smaller set:

```powershell
python -m pip install -r requirements-collector.txt
```

The repository already contains DVC metadata. Only run `dvc init` when creating a new checkout that does not contain `.dvc/`.

## Local Application

Start the backend in one terminal:

```powershell
.\.venv\Scripts\Activate.ps1
python -m backend.dashboard_backend
```

Start the frontend in a second terminal:

```powershell
cd frontend
npm ci
npx vite dev --host 127.0.0.1 --port 5173
```

Open the application at `http://127.0.0.1:5173`. The backend health endpoint is `http://127.0.0.1:5000/api/health`.

To build the frontend:

```powershell
cd frontend
npm run build
```

## Run a fast demo

```powershell
python -m experiments.prepare_data --start 2024-01-01 --end 2024-04-30
python -m experiments.train --quick
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

Open MLflow at `http://127.0.0.1:5001`.

## Run with DVC

```powershell
dvc repro
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5001
```

DVC tracks the prepared dataset and training outputs. MLflow records parameters, Sharpe metrics, the selected model, the summary CSV, and trained model artifacts. Use `dvc dag` to show the stage graph and `dvc metrics show` to inspect the generated summary.

The BERT and multimodal scripts require external FinBERT/Finnhub resources and are kept under `experiments/`.

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

## GitHub Actions

The workflow at `.github/workflows/nightly-ingestion.yml` runs the nightly collector at `01:15 UTC` and can also be started manually from the repository's **Actions** tab.

Add these repository secrets before running it:

```text
MONGODB_URI
FINNHUB_API_KEY
```

The workflow executes:

```text
python -m ingestion.collect_market_data --profile nightly
```

## Docker

Docker Compose runs the application services with these ports:

```text
Backend:  http://127.0.0.1:5000
Frontend: http://127.0.0.1:5173
MLflow:   http://127.0.0.1:5001
```

After creating `.env`, start the application stack:

```powershell
docker compose up --build
```

Services:

```text
Frontend: http://127.0.0.1:5173
Backend:  http://127.0.0.1:5000
MLflow:   http://127.0.0.1:5001
```

Run the collector manually in Docker:

```powershell
docker compose --profile collector run --rm collector
```

The backend image mounts `models/` and `outputs/` at runtime. MongoDB Atlas remains external; Docker does not start a local MongoDB service.

## Useful Commands

```powershell
dvc dag
dvc status
dvc repro
python -m ingestion.collect_market_data --profile nightly --dry-run
docker compose ps
docker compose logs backend
docker compose down
```

For another machine, restore DVC-managed artifacts before starting the backend:

```powershell
dvc pull
```

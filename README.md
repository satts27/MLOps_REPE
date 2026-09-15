# MLOps_REPE

An end-to-end stock analytics project with a Flask API, React/TanStack frontend, MongoDB Atlas ingestion, DVC data versioning, MLflow experiment tracking, Prometheus metrics, Grafana dashboards, and Docker Compose services.

## Prerequisites

- Python 3.11 recommended
- Node.js 22 or newer (the frontend Docker image uses Node.js 22)
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
monitoring/   Prometheus configuration and provisioned Grafana dashboard
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

## Quick start with Docker Desktop

From the project root, configure `.env` as above and make sure Docker Desktop is running. Python and Node do not need to be installed on the host for this route.

Prepare the mounted directories and SQLite file, then start all application and monitoring services:

```powershell
New-Item -ItemType Directory -Force models, outputs, mlruns | Out-Null
if (-not (Test-Path -LiteralPath mlflow.db)) {
    New-Item -ItemType File -Path mlflow.db | Out-Null
}
docker compose up -d --build
docker compose ps
```

Open the frontend at http://127.0.0.1:5173, MLflow at http://127.0.0.1:5001, Grafana at http://127.0.0.1:3000 (default login `admin` / `admin`), and Prometheus at http://127.0.0.1:9090. The backend health URL is http://127.0.0.1:5000/api/health. The first build can take several minutes.

Prices can fall back to live data before MongoDB is populated. To populate MongoDB, run the one-time bootstrap:

```powershell
docker compose --profile collector run --rm collector python -m ingestion.collect_market_data --profile bootstrap
```

Prediction serving requires both `models/multimodal/window_60/A2C.zip` and `outputs/multimodal_transformer_latest.pt`, plus the Finnhub key. These artifacts are not bundled in published images. Restore them from your artifact store, or train the multimodal model and select the matching checkpoint/model paths. Baseline training produces different models and does not supply these multimodal artifacts. The dashboard can show prices and news before predictions are available.

Use `docker compose logs --tail 50 backend frontend mlflow prometheus grafana` to inspect startup failures and `docker compose down` to stop the stack.

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

DVC tracks the prepared dataset and training outputs. MLflow records parameters, Sharpe metrics, the selected model, the summary CSV, and trained model artifacts. Use `dvc dag` to show the stage graph and inspect `outputs/ensemble_summary.csv` for the generated summary.

The BERT and multimodal scripts require external FinBERT/Finnhub resources and are kept under `experiments/`.

## Quarterly baseline experiments

Run the nine quarters from Q2 2024 through Q2 2026:

Activate the Python environment from **Python Setup** first. These runs are tracked in MLflow; they are not created with `dvc exp run` and will not appear as DVC experiments in the editor extension.

```powershell
python -m experiments.run_quarters
```

Each quarter uses the five configured tickers and trains A2C (5,000 steps), PPO (10,000 steps), and DDPG (5,000 steps), with seed 42. The quarterly runs use 15-day rebalance and validation windows and a holdout of up to 15 trading days, shortened when needed to leave at least one rolling training window. The default 30-day holdout leaves no rolling training windows in a single quarter. MLflow records the holdout length for each run. The final holdout is excluded from training/validation, but the current baseline does not evaluate it. Reported Sharpe scores are validation scores.

The runner downloads 90 calendar days before each quarter for feature warmup, then restricts training data to the quarter. Quarter ends are inclusive; Yahoo Finance requests use the first day of the next quarter as their exclusive end. MLflow experiments are named `baseline-2024Q2`, `baseline-2024Q3`, and so on. Data, logs, and summaries go under `outputs/quarterly/<quarter>/`, models under `models/quarterly/<quarter>/`, and the aggregate results to `outputs/quarterly/comparison.csv`. Completed quarters have a `completed.json` marker and are skipped when restarting the runner. Existing default datasets and dashboard models are preserved.

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

The workflow at `.github/workflows/nightly-ingestion.yml` runs the nightly collector at `01:15 UTC` (`06:45 IST`) and can also be started manually from the repository's **Actions** tab. Keep the workflow on the default branch for scheduled runs.

Add these repository secrets before running it:

```text
MONGODB_URI
FINNHUB_API_KEY
```

The workflow executes:

```text
python -m ingestion.collect_market_data --profile nightly
```

## Publish images to Docker Hub

The workflow `.github/workflows/docker-publish.yml` builds and pushes all three application images on every push to `main`, including documentation-only pushes. It also supports **Actions > Publish Docker images > Run workflow** on `main`.

Add repository Actions secrets `DOCKERHUB_USERNAME` (your Docker ID) and `DOCKERHUB_TOKEN` (a Docker Hub personal access token with Read and Write permissions). The workflow uses the username secret as the image namespace, so no username needs to be hardcoded.

Add them under **GitHub repository > Settings > Secrets and variables > Actions > New repository secret**. Store the token there, not in `.env` or source code. These publishing secrets are separate from the ingestion secrets `MONGODB_URI` and `FINNHUB_API_KEY`.

Create these repositories in that Docker Hub account:

- `<username>/mlops-repe-backend`
- `<username>/mlops-repe-frontend`
- `<username>/mlops-repe-collector`

Each image receives `latest` and `sha-<full Git commit SHA>` tags. Builds target `linux/amd64` and use a separate GitHub Actions build cache per service. Each push starts its own run. When builds overlap, `latest` refers to the last build that finishes publishing; use the commit SHA tag to deploy a specific version. The three images publish independently, so check that all three jobs succeed for the chosen commit.

Check the three build jobs under **Actions > Publish Docker images**, then inspect the **Tags** tab in each Docker Hub repository. This workflow publishes images; deployment and model training remain separate. Prometheus and Grafana continue to use their official images. The frontend image currently runs the Vite development server, matching the existing Dockerfile.

Credentials and local environment files are excluded from Docker build contexts. Supply application credentials at runtime; model artifacts remain mounted separately. The local Compose stack still builds from source.

The workflow uses the official [Docker GitHub Actions](https://docs.docker.com/build/ci/github-actions/).

## Docker

Docker Compose runs the application services with these ports:

```text
Backend:  http://127.0.0.1:5000
Frontend: http://127.0.0.1:5173
MLflow:   http://127.0.0.1:5001
Prometheus: http://127.0.0.1:9090
Grafana:    http://127.0.0.1:3000
```

After creating `.env`, start the application stack:

```powershell
docker compose up -d --build
```

Services:

```text
Frontend: http://127.0.0.1:5173
Backend:  http://127.0.0.1:5000
MLflow:   http://127.0.0.1:5001
Prometheus: http://127.0.0.1:9090
Grafana:    http://127.0.0.1:3000
```

Run the collector manually in Docker:

```powershell
docker compose --profile collector run --rm collector
```

The backend image mounts `models/` and `outputs/` at runtime. MongoDB Atlas remains external; Docker does not start a local MongoDB service.

## Prometheus and Grafana

Prometheus collects backend metrics, and Grafana displays them in a dashboard. Compose uses the official prebuilt images `prom/prometheus:v3.5.0` and `grafana/grafana:12.1.0`; neither monitoring service needs a custom Dockerfile. Adding the services to Compose configures them. The following command downloads missing images, builds the backend, and creates and starts the containers:

```powershell
docker compose up -d --build backend prometheus grafana
```

To start the complete application and monitoring stack in the background:

```powershell
docker compose up -d --build
```

For later starts when the backend image is already built and current:

```powershell
docker compose up -d prometheus grafana
```

Compose also starts the backend dependency. Run commands from the project root with Docker Desktop running and `.env` configured.

- Grafana: http://127.0.0.1:3000 — log in with `admin` / `admin` by default.
- Prometheus: http://127.0.0.1:9090 — inspect the `backend` target under **Status > Targets**.
- Backend metrics: http://127.0.0.1:5000/metrics

You can set `GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD` in `.env` before the first start. Grafana stores its initial credentials in its persistent volume; changing these environment variables later does not reset an existing account.

Verify the containers and generate an API request:

```powershell
docker compose ps
docker compose logs --tail 50 backend prometheus grafana
Invoke-RestMethod http://127.0.0.1:5000/api/health
Invoke-WebRequest http://127.0.0.1:5000/metrics
```

In Prometheus, confirm the `backend` target is **UP** and try the query `up{job="backend"}` (expected value: `1`). If it is down, check the backend logs and metrics endpoint. The Grafana Prometheus datasource is provisioned automatically at `http://prometheus:9090`, using the Compose service name.

Open **Dashboards > MLOps REPE > MLOps REPE — Backend**. The provisioned dashboard shows backend availability, request rate by route, HTTP 5xx percentage, p95 response time, status codes, process memory, and CPU usage. Generate traffic by opening the frontend or requesting `/api/health`; rate panels need at least two scrapes (about 30 seconds). Error rate can show no data until the first server error.

Prometheus scrapes `backend:5000/metrics` every 15 seconds using Docker's internal network. Route labels use templates such as `/api/market/<ticker>` to keep the number of series bounded. Scrape requests are excluded from API metrics. The Python client also exports process metrics in the Linux backend container. This monitors API HTTP failures; a prediction row with `Signal=ERROR` inside an HTTP 200 response is not an HTTP failure.

Configuration lives under `monitoring/`: Prometheus scrape configuration, Grafana datasource/provider YAML, and the dashboard JSON. Restart Prometheus after editing its configuration. Grafana checks the dashboard files every 30 seconds; edit the JSON to persist dashboard changes. Prometheus retains 15 days of samples. Named Docker volumes retain monitoring data across restarts and `docker compose down`; `docker compose down -v` deletes those volumes.

Instrumentation currently supports the single-process backend used by this Compose setup. Multiple workers require Prometheus Python multiprocess configuration. Monitoring covers the backend and Prometheus itself; the collector, MLflow, MongoDB, and container resource metrics do not have exporters configured.

The setup follows the official [Prometheus Flask integration](https://prometheus.github.io/client_python/exporting/http/flask/) and [Grafana provisioning documentation](https://grafana.com/docs/grafana/latest/administration/provisioning/).

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

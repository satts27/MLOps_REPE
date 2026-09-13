# MLOps_REPE

An end-to-end stock analytics project with a Flask API, React/TanStack frontend, MongoDB Atlas ingestion, DVC data versioning, MLflow experiment tracking, Prometheus metrics, Grafana dashboards, and Docker Compose services.

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
Prometheus: http://127.0.0.1:9090
Grafana:    http://127.0.0.1:3000
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

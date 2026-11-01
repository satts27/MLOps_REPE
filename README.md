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

## Jenkins and Kubernetes

`Jenkinsfile` provides CI/CD for the `main` branch: it compiles the Python modules, builds the frontend, builds and publishes the backend, collector, and frontend images to Docker Hub with both `sha-<commit>` and `latest` tags, then deploys the commit-tagged backend and frontend to Kubernetes. Other branches run validation only.

Configure a Jenkins Multibranch Pipeline for this repository. The Jenkins agent must have Git, Python 3.11, Node.js 22 with npm, Docker, `kubectl`, and network access to the Docker daemon and target cluster. Add these Jenkins credentials:

| Credential ID | Jenkins type | Used for |
| --- | --- | --- |
| `dockerhub-credentials` | Username with password | Docker Hub username and access token |
| `kubeconfig` | Secret file | Kubernetes cluster credentials |

Make sure the Docker Hub image repositories from **Publish images to Docker Hub** exist. The cluster must be able to pull them (public repositories need no additional setup). Install an NGINX Ingress Controller and configure DNS or a local hosts entry for `mlops-repe.local` to resolve to its address.

Before the first deployment, create the application secret and restore model artifacts to the persistent volumes. Create `k8s/app-secrets.env` with only these keys and values (do not commit it):

```env
MONGODB_URI=your_mongodb_connection_string
FINNHUB_API_KEY=your_finnhub_api_key
```

Apply the secret once:

```sh
kubectl create namespace mlops-repe
kubectl create secret generic mlops-repe-secrets \
  --namespace mlops-repe \
  --from-env-file=k8s/app-secrets.env
```

The pipeline creates two 5 GiB persistent volume claims for `models/` and `outputs/`; the cluster needs a default StorageClass capable of provisioning `ReadWriteOnce` volumes. Copy `models/multimodal/window_60/A2C.zip` into the models volume and `outputs/multimodal_transformer_latest.pt` into the outputs volume before relying on predictions. The API and frontend can start without these artifacts, but prediction serving cannot. Keep the environment file secure and remove it after creating/updating the Kubernetes Secret. To rotate credentials, rerun the secret creation command with `--dry-run=client -o yaml | kubectl apply -f -` after editing the file, then restart the backend deployment with `kubectl rollout restart deployment/mlops-repe-backend --namespace mlops-repe`.

The Kubernetes manifests are in [k8s/application.yaml](./k8s/application.yaml). They expose the frontend and backend internally and route `/api` and `/metrics` to the backend through Ingress; `/` serves the frontend. The frontend image is built with an empty `VITE_API_BASE` so browser requests use the same Ingress host. The Compose build instead sets the API base to `http://127.0.0.1:5000`.

### Kubernetes availability and request-rate scaling

Both Deployments use startup, readiness, and liveness probes. A startup probe allows the
application to initialize before liveness checks begin; readiness removes a pod from its
Service when it cannot serve traffic; liveness restarts an unhealthy pod for automatic
self-healing.

The backend HorizontalPodAutoscaler scales between one and five replicas using:

- backend requests per minute, targeting 30 requests per pod per minute; and
- CPU utilization, targeting 70%.

Install Prometheus Operator and metrics-server first, then install the Prometheus Adapter
with the provided custom metric mapping:

```sh
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update
helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --create-namespace
helm upgrade --install prometheus-adapter prometheus-community/prometheus-adapter \
  --namespace monitoring \
  --values k8s/prometheus-adapter-values.yaml
```

Confirm that the custom metric is available:

```sh
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta2" | head
kubectl get hpa -n mlops-repe --watch
kubectl get pods -n mlops-repe -l app=mlops-repe-backend --watch
```

Generate load to observe request-rate scaling:

```sh
for i in $(seq 1 1000); do
  curl -s "http://mlops-repe.local/api/health" >/dev/null
done
```

After the Helm stack is installed, apply [k8s/prometheus-hpa-alerts.yaml](./k8s/prometheus-hpa-alerts.yaml). That `PrometheusRule` is separate from [k8s/application.yaml](./k8s/application.yaml) so a Jenkins deploy still works if the Prometheus Operator CRDs are not on the cluster.

```sh
kubectl apply -f k8s/prometheus-hpa-alerts.yaml
```

`BackendHpaScaledOut` fires when the backend HPA desired replica count stays above 1 for one minute (scaled out) and resolves when it returns to 1. It does not notify on every 2 to 3 to 4 step. Open the cluster Alertmanager UI with:

```sh
kubectl port-forward -n monitoring svc/prometheus-kube-prometheus-alertmanager 9093:9093
```

Then browse http://127.0.0.1:9093. Compose Alertmanager on Docker is a different process and does not see this HPA metric.

Compose email comes from `.env`. Cluster email is a Helm values file: copy [k8s/kube-prometheus-alertmanager-values.example.yaml](./k8s/kube-prometheus-alertmanager-values.example.yaml) to `k8s/kube-prometheus-alertmanager-values.yaml`, paste SMTP settings and the `to:` list, then:

```sh
helm upgrade prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring --reuse-values \
  --values k8s/kube-prometheus-alertmanager-values.yaml
```

Do not commit the filled `k8s/kube-prometheus-alertmanager-values.yaml` (it is gitignored).

## Prometheus and Grafana

Prometheus collects backend metrics, and Grafana displays them in a dashboard. Compose uses the official prebuilt images `prom/prometheus:v3.5.0` and `grafana/grafana:12.1.0`; neither monitoring service needs a custom Dockerfile. Adding the services to Compose configures them. The following command downloads missing images, builds the backend, and creates and starts the containers:

```powershell
docker compose up -d --build backend prometheus alertmanager grafana
```

To start the complete application and monitoring stack in the background:

```powershell
docker compose up -d --build
```

For later starts when the backend image is already built and current:

```powershell
docker compose up -d prometheus alertmanager grafana
```

Compose also starts the backend dependency. Run commands from the project root with Docker Desktop running and `.env` configured.

- Grafana: http://127.0.0.1:3000 — log in with `admin` / `admin` by default.
- Prometheus: http://127.0.0.1:9090 — inspect the `backend` target under **Status > Targets**.
- Alertmanager: http://127.0.0.1:9093 — firing alerts after Prometheus evaluates the rules.
- Backend metrics: http://127.0.0.1:5000/metrics

You can set `GRAFANA_ADMIN_USER` and `GRAFANA_ADMIN_PASSWORD` in `.env` before the first start. Grafana stores its initial credentials in its persistent volume; changing these environment variables later does not reset an existing account.

Verify the containers and generate an API request:

```powershell
docker compose ps
docker compose logs --tail 50 backend prometheus alertmanager grafana
Invoke-RestMethod http://127.0.0.1:5000/api/health
Invoke-WebRequest http://127.0.0.1:5000/metrics
```

In Prometheus, confirm the `backend` target is **UP** and try the query `up{job="backend"}` (expected value: `1`). If it is down, check the backend logs and metrics endpoint. The Grafana Prometheus datasource is provisioned automatically at `http://prometheus:9090`, using the Compose service name.

Open **Dashboards > MLOps REPE > MLOps REPE — Backend**. The provisioned dashboard shows backend availability, request rate by route, HTTP 5xx percentage, p95 response time, status codes, process memory, and CPU usage. Generate traffic by opening the frontend or requesting `/api/health`; rate panels need at least two scrapes (about 30 seconds). Error rate can show no data until the first server error.

Prometheus scrapes `backend:5000/metrics` every 15 seconds using Docker's internal network. Route labels use templates such as `/api/market/<ticker>` to keep the number of series bounded. Scrape requests are excluded from API metrics. The Python client also exports process metrics in the Linux backend container. This monitors API HTTP failures; a prediction row with `Signal=ERROR` inside an HTTP 200 response is not an HTTP failure.

## Prometheus alerting

Prometheus evaluates `monitoring/prometheus/alerts.yml` every 15 seconds and sends firing alerts to Alertmanager on the Compose network (`alertmanager:9093`). `Dockerfile.alertmanager` wraps `prom/alertmanager:v0.28.1` so SMTP settings can come from `.env`. The Alertmanager UI is published at `127.0.0.1:9093`.

Rules:

- `BackendDown` — `up{job="backend"} == 0` for 30 seconds
- `BackendHighErrorRate` — HTTP 5xx share above 5% for 2 minutes

Alertmanager is configured in `monitoring/alertmanager/alertmanager.yml`. It always shows alerts in the UI at http://127.0.0.1:9093. Prometheus alerts are at http://127.0.0.1:9090/alerts.

Optional email: set these in `.env` (never commit them), then rebuild Alertmanager:

```env
ALERT_EMAIL_TO=you@example.com
SMTP_FROM=you@example.com
SMTP_USERNAME=you@example.com
SMTP_PASSWORD=your_app_password
SMTP_SMARTHOST=smtp.gmail.com:587
```

```powershell
docker compose up -d --build alertmanager
```

If those values are empty, Alertmanager stays UI-only. For Gmail, turn on 2-Step Verification and create an [App Password](https://support.google.com/accounts/answer/185833); do not use your normal account password. Other providers work if you set `SMTP_SMARTHOST` (host:port). Firing `BackendDown` then sends mail to `ALERT_EMAIL_TO` and a resolve mail when the backend comes back.

To fire `BackendDown` in a demo:

```powershell
docker compose stop backend
```

Wait about 30 seconds. The alert should move from **Pending** to **Firing**. Then:

```powershell
docker compose start backend
```

The alert resolves after the next successful scrape. `BackendHighErrorRate` stays inactive until the API returns 5xx responses. Restart Prometheus after editing `alerts.yml` or `prometheus.yml`.

Configuration lives under `monitoring/`: Prometheus scrape configuration, alert rules, Alertmanager YAML, Grafana datasource/provider YAML, and the dashboard JSON. Restart Prometheus after editing its scrape or rule files. Grafana checks the dashboard files every 30 seconds; edit the JSON to persist dashboard changes. Prometheus retains 15 days of samples. Named Docker volumes retain monitoring data across restarts and `docker compose down`; `docker compose down -v` deletes those volumes.

Instrumentation currently supports the single-process backend used by this Compose setup. Multiple workers require Prometheus Python multiprocess configuration. Monitoring covers the backend and Prometheus itself; the collector, MLflow, MongoDB, and container resource metrics do not have exporters configured.

The setup follows the official [Prometheus Flask integration](https://prometheus.github.io/client_python/exporting/http/flask/), [alerting and Alertmanager](https://prometheus.io/docs/alerting/latest/overview/), and [Grafana provisioning documentation](https://grafana.com/docs/grafana/latest/administration/provisioning/).

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

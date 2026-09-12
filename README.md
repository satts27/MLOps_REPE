# MLOps pipeline

This project now has a DVC pipeline around the yfinance RL baseline and MLflow tracking for each training run.

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
python prepare_data.py --start 2024-01-01 --end 2024-04-30
python train.py --quick
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

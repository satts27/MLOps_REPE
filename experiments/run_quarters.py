"""Run isolated quarterly baseline experiments with MLflow tracking."""
from pathlib import Path
import json
import os
import subprocess
import sys

import pandas as pd
from experiments import Baseline_yfinance as baseline


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'outputs' / 'quarterly'
    output.mkdir(parents=True, exist_ok=True)
    results = []
    for quarter in pd.period_range('2024Q2', '2026Q2', freq='Q'):
        label = str(quarter)
        folder = output / label
        folder.mkdir(exist_ok=True)
        marker = folder / 'completed.json'
        if marker.exists():
            results.append(json.loads(marker.read_text()))
            continue
        start = quarter.start_time
        end = (quarter + 1).start_time
        dataset = folder / 'processed.csv'
        print(f'{label}: preparing {start.date()} through {(end - pd.Timedelta(days=1)).date()}', flush=True)
        data = baseline.download_data(baseline.TICKERS, (start - pd.Timedelta(days=90)).strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d'))
        if set(data.tic.unique()) != set(baseline.TICKERS):
            raise RuntimeError(f'{label}: missing ticker data')
        data = baseline.add_technical_indicators(data)
        data = baseline.add_turbulence(data)
        data = data[(data.datadate >= start) & (data.datadate < end)].copy()
        counts = data.groupby('datadate').tic.nunique()
        if not counts.eq(len(baseline.TICKERS)).all() or len(counts) < 56:
            raise RuntimeError(f'{label}: incomplete data or insufficient trading days: {len(counts)}')
        # Leave room for at least one rolling point (the first is day 45).
        test_window = min(15, len(counts) - 46)
        data.to_csv(dataset, index=False)
        command = [sys.executable, '-m', 'experiments.train', '--data', str(dataset), '--output-dir', str(folder), '--model-dir', str(root / 'models' / 'quarterly' / label), '--experiment', f'baseline-{label}', '--test-window', str(test_window)]
        print(f'{label}: training; log: {folder / "training.log"}', flush=True)
        with (folder / 'training.log').open('w') as log:
            env = {**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
            subprocess.run(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        summary = pd.read_csv(folder / 'ensemble_summary.csv')
        if summary.empty:
            raise RuntimeError(f'{label}: no training windows completed')
        result = {'quarter': label, 'start': str(start.date()), 'end_inclusive': str((end - pd.Timedelta(days=1)).date()), 'trading_days': len(counts), 'test_window': test_window, 'windows_completed': len(summary), 'best_model': summary.best_model.mode().iat[0], **{f'mean_{c}': float(summary[c].mean()) for c in ('a2c_sharpe', 'ppo_sharpe', 'ddpg_sharpe')}}
        marker.write_text(json.dumps(result, indent=2))
        results.append(result)
        pd.DataFrame(results).to_csv(output / 'comparison.csv', index=False)
        print(f'{label}: completed', flush=True)
    pd.DataFrame(results).to_csv(output / 'comparison.csv', index=False)


if __name__ == '__main__':
    main()

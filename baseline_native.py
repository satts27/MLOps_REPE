#!/usr/bin/env python3
"""Native baseline runner for the ICAIF 2020 ensemble strategy.

This script keeps the original
`Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020/`
folder untouched and ports the baseline workflow to a modern, non-Docker
execution path based on `gymnasium` and `stable-baselines3`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_DIR = (
    REPO_ROOT
    / "Deep-Reinforcement-Learning-for-Automated-Stock-Trading-Ensemble-Strategy-ICAIF-2020"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "baseline_native_outputs"
DEFAULT_DATA_RELATIVE_PATH = Path("data/dow_30_2009_2020.csv")


@dataclass
class BaselinePaths:
    source_dir: Path
    data_file: Path
    output_dir: Path
    cache_dir: Path
    processed_cache: Path
    run_dir: Path
    results_dir: Path
    trained_models_dir: Path


@dataclass
class BaselineConfig:
    paths: BaselinePaths
    rebalance_window: int
    validation_window: int
    a2c_timesteps: int
    ppo_timesteps: int
    ddpg_timesteps: int
    seed: int
    refresh_cache: bool
    quick: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the RL trading baseline natively without Docker while leaving "
            "the upstream baseline folder unchanged."
        )
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Path to the untouched upstream baseline folder.",
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        default=DEFAULT_DATA_RELATIVE_PATH,
        help=(
            "CSV to train on. Relative paths are resolved inside --source-dir. "
            "Defaults to data/dow_30_2009_2020.csv."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Root directory for native outputs and cached preprocessed data.",
    )
    parser.add_argument(
        "--run-name",
        default="",
        help="Optional name for the current run. Defaults to a timestamp.",
    )
    parser.add_argument(
        "--rebalance-window",
        type=int,
        default=63,
        help="Number of trading days before retraining the ensemble.",
    )
    parser.add_argument(
        "--validation-window",
        type=int,
        default=63,
        help="Number of trading days used for rolling validation.",
    )
    parser.add_argument(
        "--a2c-timesteps",
        type=int,
        default=30_000,
        help="Training timesteps for A2C.",
    )
    parser.add_argument(
        "--ppo-timesteps",
        type=int,
        default=100_000,
        help="Training timesteps for PPO.",
    )
    parser.add_argument(
        "--ddpg-timesteps",
        type=int,
        default=10_000,
        help="Training timesteps for DDPG.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed passed to the native agents.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Rebuild the preprocessed cache instead of reusing an existing file.",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a much smaller smoke configuration for faster local checks.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only verify the baseline source layout and print what was found.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> BaselineConfig:
    source_dir = args.source_dir.resolve()
    output_dir = args.output_dir.resolve()
    data_file = (
        args.data_file.resolve()
        if args.data_file.is_absolute()
        else (source_dir / args.data_file).resolve()
    )
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    cache_dir = output_dir / "cache"
    run_dir = output_dir / "runs" / run_name
    results_dir = run_dir / "results"
    trained_models_dir = run_dir / "trained_models"

    a2c_timesteps = args.a2c_timesteps
    ppo_timesteps = args.ppo_timesteps
    ddpg_timesteps = args.ddpg_timesteps

    if args.quick:
        a2c_timesteps = min(a2c_timesteps, 2_000)
        ppo_timesteps = min(ppo_timesteps, 4_000)
        ddpg_timesteps = min(ddpg_timesteps, 1_500)

    paths = BaselinePaths(
        source_dir=source_dir,
        data_file=data_file,
        output_dir=output_dir,
        cache_dir=cache_dir,
        processed_cache=cache_dir / "done_data.csv",
        run_dir=run_dir,
        results_dir=results_dir,
        trained_models_dir=trained_models_dir,
    )
    return BaselineConfig(
        paths=paths,
        rebalance_window=args.rebalance_window,
        validation_window=args.validation_window,
        a2c_timesteps=a2c_timesteps,
        ppo_timesteps=ppo_timesteps,
        ddpg_timesteps=ddpg_timesteps,
        seed=args.seed,
        refresh_cache=args.refresh_cache,
        quick=args.quick,
    )


def validate_source_layout(config: BaselineConfig) -> list[str]:
    source_dir = config.paths.source_dir
    expected = [
        source_dir,
        source_dir / "run_DRL.py",
        source_dir / "requirements.txt",
        source_dir / "config" / "config.py",
        source_dir / "preprocessing" / "preprocessors.py",
        source_dir / "model" / "models.py",
        config.paths.data_file,
    ]
    missing = [str(path) for path in expected if not path.exists()]
    return missing


def count_files(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(1 for path in directory.rglob("*") if path.is_file())


def print_check_report(config: BaselineConfig) -> int:
    missing = validate_source_layout(config)
    if missing:
        print("Baseline source layout check failed.")
        for item in missing:
            print(f"- Missing: {item}")
        return 1

    source_dir = config.paths.source_dir
    upstream_results = source_dir / "results"
    upstream_models = source_dir / "trained_models"

    print("Baseline source layout looks good.")
    print(f"Source baseline: {source_dir}")
    print(f"Training data: {config.paths.data_file}")
    print(f"Upstream results files found: {count_files(upstream_results)}")
    print(f"Upstream model files found: {count_files(upstream_models)}")
    print(f"Native output root: {config.paths.output_dir}")
    print("Preferred native install command:")
    print("pip install -r requirements-baseline-native.txt")
    return 0


def missing_dependencies() -> list[str]:
    required = [
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
        ("gymnasium", "gymnasium"),
        ("stable-baselines3", "stable_baselines3"),
        ("stockstats", "stockstats"),
    ]
    return [package for package, module_name in required if importlib.util.find_spec(module_name) is None]


def load_runtime_modules() -> SimpleNamespace:
    missing = missing_dependencies()
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(
            "Missing native baseline dependencies: "
            f"{joined}\n"
            "Create a virtual environment and install them with:\n"
            "python3 -m venv .venv\n"
            "source .venv/bin/activate\n"
            "pip install -r requirements-baseline-native.txt"
        )

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import gymnasium as gym
    from gymnasium import spaces
    from stable_baselines3 import A2C, DDPG, PPO
    from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stockstats import StockDataFrame as StockDataFrame

    return SimpleNamespace(
        np=np,
        pd=pd,
        gym=gym,
        spaces=spaces,
        plt=plt,
        A2C=A2C,
        DDPG=DDPG,
        PPO=PPO,
        DummyVecEnv=DummyVecEnv,
        OrnsteinUhlenbeckActionNoise=OrnsteinUhlenbeckActionNoise,
        StockDataFrame=StockDataFrame,
    )


def create_runtime(config: BaselineConfig, libs: SimpleNamespace) -> SimpleNamespace:
    np = libs.np
    pd = libs.pd
    gym = libs.gym
    spaces = libs.spaces
    plt = libs.plt

    hmax_normalize = 100
    initial_account_balance = 1_000_000
    stock_dim = 30
    transaction_fee_percent = 0.001
    reward_scaling = 1e-4
    observation_dim = 181

    def safe_slug(label: str) -> str:
        return label.replace(" ", "_").replace("/", "_").lower()

    def compose_state(cash: float, share_counts: list[float], daily_frame) -> "np.ndarray":
        state = [cash]
        state.extend(daily_frame.adjcp.values.tolist())
        state.extend(share_counts)
        state.extend(daily_frame.macd.values.tolist())
        state.extend(daily_frame.rsi.values.tolist())
        state.extend(daily_frame.cci.values.tolist())
        state.extend(daily_frame.adx.values.tolist())
        return np.asarray(state, dtype=np.float32)

    def total_asset(state: "np.ndarray") -> float:
        prices = state[1 : stock_dim + 1]
        shares = state[stock_dim + 1 : stock_dim * 2 + 1]
        return float(state[0] + np.dot(prices, shares))

    class BaseStockEnv(gym.Env):
        metadata = {"render_modes": ["human"]}

        def __init__(
            self,
            df,
            *,
            mode: str,
            results_dir: Path,
            label: str,
            iteration: int | str = "",
            turbulence_threshold: float | None = None,
            initial: bool = True,
            previous_state: list[float] | None = None,
        ) -> None:
            super().__init__()
            self.df = df
            self.mode = mode
            self.results_dir = results_dir
            self.label = label
            self.iteration = iteration
            self.turbulence_threshold = turbulence_threshold
            self.initial = initial
            self.previous_state = previous_state or []

            self.action_space = spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(stock_dim,),
                dtype=np.float32,
            )
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(observation_dim,),
                dtype=np.float32,
            )

            self.day = 0
            self.data = None
            self.state = None
            self.terminal = False
            self.turbulence = 0.0
            self.reward = 0.0
            self.cost = 0.0
            self.trades = 0
            self.asset_memory = []
            self.rewards_memory = []

        def _sell_stock(self, index: int, action: float) -> None:
            shares_index = index + stock_dim + 1
            price_index = index + 1

            if self.mode != "train" and self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    shares_to_sell = self.state[shares_index]
                    if shares_to_sell > 0:
                        sale_value = self.state[price_index] * shares_to_sell
                        self.state[0] += sale_value * (1 - transaction_fee_percent)
                        self.cost += sale_value * transaction_fee_percent
                        self.state[shares_index] = 0
                        self.trades += 1
                    return

            if self.state[shares_index] <= 0:
                return

            shares_to_sell = min(abs(action), self.state[shares_index])
            sale_value = self.state[price_index] * shares_to_sell
            self.state[0] += sale_value * (1 - transaction_fee_percent)
            self.state[shares_index] -= shares_to_sell
            self.cost += sale_value * transaction_fee_percent
            self.trades += 1

        def _buy_stock(self, index: int, action: float) -> None:
            if self.mode != "train" and self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    return

            price_index = index + 1
            shares_index = index + stock_dim + 1
            available_amount = self.state[0] // self.state[price_index]
            shares_to_buy = min(available_amount, action)
            if shares_to_buy <= 0:
                return

            purchase_value = self.state[price_index] * shares_to_buy
            self.state[0] -= purchase_value * (1 + transaction_fee_percent)
            self.state[shares_index] += shares_to_buy
            self.cost += purchase_value * transaction_fee_percent
            self.trades += 1

        def _account_value_frame(self):
            return pd.DataFrame({"account_value": self.asset_memory})

        def _save_plot(self, path: Path) -> None:
            plt.plot(self.asset_memory, "r")
            plt.savefig(path)
            plt.close()

        def _save_terminal_outputs(self) -> None:
            label_slug = safe_slug(self.label)
            if self.mode == "train":
                prefix = f"account_value_train_{label_slug}_{self.iteration}"
            elif self.mode == "validation":
                prefix = f"account_value_validation_{label_slug}_{self.iteration}"
            else:
                prefix = f"account_value_trade_{label_slug}_{self.iteration}"

            value_path = self.results_dir / f"{prefix}.csv"
            plot_path = self.results_dir / f"{prefix}.png"
            self._account_value_frame().to_csv(value_path, index=False)
            self._save_plot(plot_path)

            if self.mode == "trade":
                rewards_path = self.results_dir / (
                    f"account_rewards_trade_{label_slug}_{self.iteration}.csv"
                )
                pd.DataFrame({"reward": self.rewards_memory}).to_csv(
                    rewards_path,
                    index=False,
                )

        def reset(self, *, seed: int | None = None, options: dict | None = None):
            super().reset(seed=seed)
            self.day = 0
            self.data = self.df.loc[self.day, :]
            self.terminal = False
            self.turbulence = 0.0
            self.reward = 0.0
            self.cost = 0.0
            self.trades = 0
            self.rewards_memory = []

            if self.mode == "trade" and not self.initial and self.previous_state:
                previous_state = np.asarray(self.previous_state, dtype=np.float32)
                previous_total_asset = total_asset(previous_state)
                self.asset_memory = [previous_total_asset]
                cash = float(previous_state[0])
                shares = previous_state[stock_dim + 1 : stock_dim * 2 + 1].tolist()
            else:
                self.asset_memory = [initial_account_balance]
                cash = float(initial_account_balance)
                shares = [0.0] * stock_dim

            self.state = compose_state(cash, shares, self.data)
            return self.state.copy(), {}

        def render(self):
            return self.state.copy().tolist()

        def step(self, actions):
            actions = np.asarray(actions, dtype=np.float32).flatten()
            self.terminal = self.day >= len(self.df.index.unique()) - 1

            if self.terminal:
                self._save_terminal_outputs()
                return self.state.copy(), float(self.reward), True, False, {}

            scaled_actions = actions * hmax_normalize
            if self.mode != "train" and self.turbulence_threshold is not None:
                if self.turbulence >= self.turbulence_threshold:
                    scaled_actions = np.full(stock_dim, -hmax_normalize, dtype=np.float32)

            begin_total_asset = total_asset(self.state)
            argsort_actions = np.argsort(scaled_actions)
            sell_index = argsort_actions[: np.where(scaled_actions < 0)[0].shape[0]]
            buy_index = argsort_actions[::-1][: np.where(scaled_actions > 0)[0].shape[0]]

            for index in sell_index:
                self._sell_stock(index, scaled_actions[index])

            for index in buy_index:
                self._buy_stock(index, scaled_actions[index])

            self.day += 1
            self.data = self.df.loc[self.day, :]
            if self.mode != "train" and "turbulence" in self.data:
                self.turbulence = float(self.data["turbulence"].values[0])

            share_counts = self.state[stock_dim + 1 : stock_dim * 2 + 1].tolist()
            self.state = compose_state(float(self.state[0]), share_counts, self.data)

            end_total_asset = total_asset(self.state)
            self.asset_memory.append(end_total_asset)
            self.reward = (end_total_asset - begin_total_asset) * reward_scaling
            self.rewards_memory.append(end_total_asset - begin_total_asset)
            return self.state.copy(), float(self.reward), False, False, {}

    def data_split(df, start: int, end: int):
        data = df[(df.datadate >= start) & (df.datadate < end)]
        data = data.sort_values(["datadate", "tic"], ignore_index=True)
        data.index = data.datadate.factorize()[0]
        return data

    def load_or_create_preprocessed_data():
        upstream_cache = config.paths.source_dir / "done_data.csv"

        if not config.refresh_cache:
            if config.paths.processed_cache.exists():
                print(f"Using cached preprocessed data: {config.paths.processed_cache}")
                return pd.read_csv(config.paths.processed_cache, index_col=0)

            if upstream_cache.exists():
                print(f"Using upstream preprocessed data: {upstream_cache}")
                data = pd.read_csv(upstream_cache, index_col=0)
                config.paths.cache_dir.mkdir(parents=True, exist_ok=True)
                data.to_csv(config.paths.processed_cache)
                return data

        print("Building preprocessed data cache from raw dataset.")
        raw_df = pd.read_csv(config.paths.data_file)
        raw_df = raw_df[raw_df.datadate >= 20090000]
        price_df = calculate_price(raw_df)
        enriched_df = add_technical_indicator(price_df)
        enriched_df = enriched_df.bfill()
        full_df = add_turbulence(enriched_df)
        config.paths.cache_dir.mkdir(parents=True, exist_ok=True)
        full_df.to_csv(config.paths.processed_cache)
        return full_df

    def calculate_price(df):
        data = df.copy()
        data = data[
            ["datadate", "tic", "prccd", "ajexdi", "prcod", "prchd", "prcld", "cshtrd"]
        ]
        data["ajexdi"] = data["ajexdi"].replace(0, 1)
        data["adjcp"] = data["prccd"] / data["ajexdi"]
        data["open"] = data["prcod"] / data["ajexdi"]
        data["high"] = data["prchd"] / data["ajexdi"]
        data["low"] = data["prcld"] / data["ajexdi"]
        data["volume"] = data["cshtrd"]
        data = data[["datadate", "tic", "adjcp", "open", "high", "low", "volume"]]
        return data.sort_values(["tic", "datadate"], ignore_index=True)

    def add_technical_indicator(df):
        stock = libs.StockDataFrame.retype(df.copy())
        stock["close"] = stock["adjcp"]
        unique_ticker = stock.tic.unique()

        macd_frames = []
        rsi_frames = []
        cci_frames = []
        adx_frames = []

        for ticker in unique_ticker:
            ticker_data = stock[stock.tic == ticker]
            macd_frames.append(pd.DataFrame(ticker_data["macd"]).reset_index(drop=True))
            rsi_frames.append(pd.DataFrame(ticker_data["rsi_30"]).reset_index(drop=True))
            cci_frames.append(pd.DataFrame(ticker_data["cci_30"]).reset_index(drop=True))
            adx_frames.append(pd.DataFrame(ticker_data["dx_30"]).reset_index(drop=True))

        df = df.copy()
        df["macd"] = pd.concat(macd_frames, ignore_index=True).iloc[:, 0]
        df["rsi"] = pd.concat(rsi_frames, ignore_index=True).iloc[:, 0]
        df["cci"] = pd.concat(cci_frames, ignore_index=True).iloc[:, 0]
        df["adx"] = pd.concat(adx_frames, ignore_index=True).iloc[:, 0]
        return df

    def calculate_turbulence(df):
        df_price_pivot = df.pivot(index="datadate", columns="tic", values="adjcp")
        unique_date = df.datadate.unique()
        start = 252
        turbulence_index = [0.0] * start
        positive_count = 0

        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            hist_price = df_price_pivot[df_price_pivot.index.isin(unique_date[0:i])]
            cov_temp = hist_price.cov()
            current_delta = current_price - hist_price.mean(axis=0)
            temp = current_delta.values.dot(np.linalg.pinv(cov_temp.values)).dot(
                current_delta.values.T
            )

            turbulence_value = float(temp[0][0]) if temp.size else 0.0
            if turbulence_value > 0:
                positive_count += 1
                turbulence_index.append(turbulence_value if positive_count > 2 else 0.0)
            else:
                turbulence_index.append(0.0)

        return pd.DataFrame(
            {"datadate": df_price_pivot.index, "turbulence": turbulence_index}
        )

    def add_turbulence(df):
        turbulence_index = calculate_turbulence(df)
        merged = df.merge(turbulence_index, on="datadate")
        return merged.sort_values(["datadate", "tic"]).reset_index(drop=True)

    def calculate_sharpe(asset_memory, factor: int) -> float:
        account_values = pd.Series(asset_memory, dtype=float)
        returns = account_values.pct_change().dropna()
        if returns.empty:
            return float("-inf")
        std = returns.std()
        if std == 0 or pd.isna(std):
            return float("-inf")
        return float((factor ** 0.5) * returns.mean() / std)

    def create_train_env(train_df, label: str, iteration: int):
        return libs.DummyVecEnv(
            [
                lambda: BaseStockEnv(
                    train_df,
                    mode="train",
                    results_dir=config.paths.results_dir,
                    label=label,
                    iteration=iteration,
                )
            ]
        )

    def create_validation_env(validation_df, label: str, iteration: int, threshold: float):
        return libs.DummyVecEnv(
            [
                lambda: BaseStockEnv(
                    validation_df,
                    mode="validation",
                    results_dir=config.paths.results_dir,
                    label=label,
                    iteration=iteration,
                    turbulence_threshold=threshold,
                )
            ]
        )

    def create_trade_env(
        trade_df,
        label: str,
        iteration: int,
        threshold: float,
        initial: bool,
        previous_state: list[float],
    ):
        return libs.DummyVecEnv(
            [
                lambda: BaseStockEnv(
                    trade_df,
                    mode="trade",
                    results_dir=config.paths.results_dir,
                    label=label,
                    iteration=iteration,
                    turbulence_threshold=threshold,
                    initial=initial,
                    previous_state=previous_state,
                )
            ]
        )

    def train_a2c(train_df, model_name: str, iteration: int):
        env_train = create_train_env(train_df, "a2c", iteration)
        model = libs.A2C("MlpPolicy", env_train, seed=config.seed, verbose=0)
        model.learn(total_timesteps=config.a2c_timesteps)
        model.save(str(config.paths.trained_models_dir / model_name))
        env_train.close()
        return model

    def train_ppo(train_df, model_name: str, iteration: int):
        env_train = create_train_env(train_df, "ppo", iteration)
        model = libs.PPO(
            "MlpPolicy",
            env_train,
            ent_coef=0.005,
            seed=config.seed,
            verbose=0,
        )
        model.learn(total_timesteps=config.ppo_timesteps)
        model.save(str(config.paths.trained_models_dir / model_name))
        env_train.close()
        return model

    def train_ddpg(train_df, model_name: str, iteration: int):
        env_train = create_train_env(train_df, "ddpg", iteration)
        action_noise = libs.OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(stock_dim),
            sigma=0.5 * np.ones(stock_dim),
        )
        model = libs.DDPG(
            "MlpPolicy",
            env_train,
            action_noise=action_noise,
            seed=config.seed,
            verbose=0,
        )
        model.learn(total_timesteps=config.ddpg_timesteps)
        model.save(str(config.paths.trained_models_dir / model_name))
        env_train.close()
        return model

    def run_validation(model, validation_df, label: str, iteration: int, threshold: float):
        env_val = create_validation_env(validation_df, label, iteration, threshold)
        obs = env_val.reset()
        for _ in range(len(validation_df.index.unique())):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env_val.step(action)
            if bool(dones[0]):
                break
        sharpe = calculate_sharpe(env_val.envs[0].asset_memory, factor=4)
        env_val.close()
        return sharpe

    def run_trade(
        df,
        model,
        label: str,
        last_state: list[float],
        iteration: int,
        unique_trade_date,
        threshold: float,
        initial: bool,
    ):
        trade_df = data_split(
            df,
            start=unique_trade_date[iteration - config.rebalance_window],
            end=unique_trade_date[iteration],
        )
        env_trade = create_trade_env(
            trade_df,
            label,
            iteration,
            threshold,
            initial,
            last_state,
        )
        obs = env_trade.reset()
        for _ in range(len(trade_df.index.unique())):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = env_trade.step(action)
            if bool(dones[0]):
                break

        final_state = env_trade.envs[0].render()
        pd.DataFrame({"last_state": final_state}).to_csv(
            config.paths.results_dir / f"last_state_{safe_slug(label)}_{iteration}.csv",
            index=False,
        )
        env_trade.close()
        return final_state

    def run_ensemble_strategy(df, unique_trade_date):
        print("============ Start Native Ensemble Strategy ============")
        last_state_ensemble: list[float] = []
        summary_rows = []

        insample_turbulence = df[(df.datadate < 20151000) & (df.datadate >= 20090000)]
        insample_turbulence = insample_turbulence.drop_duplicates(subset=["datadate"])
        insample_threshold = np.quantile(insample_turbulence.turbulence.values, 0.90)

        start_time = datetime.now()
        for i in range(
            config.rebalance_window + config.validation_window,
            len(unique_trade_date),
            config.rebalance_window,
        ):
            print("============================================")
            initial = i - config.rebalance_window - config.validation_window == 0

            validation_start = unique_trade_date[
                i - config.rebalance_window - config.validation_window
            ]
            validation_end = unique_trade_date[i - config.rebalance_window]
            trade_end = unique_trade_date[i]

            end_date_index = df.index[df["datadate"] == validation_start].to_list()[-1]
            start_date_index = end_date_index - config.validation_window * 30 + 1
            historical_turbulence = df.iloc[start_date_index : (end_date_index + 1), :]
            historical_turbulence = historical_turbulence.drop_duplicates(subset=["datadate"])
            historical_mean = np.mean(historical_turbulence.turbulence.values)

            if historical_mean > insample_threshold:
                turbulence_threshold = float(insample_threshold)
            else:
                turbulence_threshold = float(
                    np.quantile(insample_turbulence.turbulence.values, 1)
                )

            print(f"Validation window: {validation_start} -> {validation_end}")
            print(f"Trade window: {validation_end} -> {trade_end}")
            print(f"Turbulence threshold: {turbulence_threshold:.4f}")

            train_df = data_split(df, start=20090000, end=validation_start)
            validation_df = data_split(df, start=validation_start, end=validation_end)

            print("Training A2C...")
            model_a2c = train_a2c(train_df, f"A2C_30k_dow_{i}", i)
            sharpe_a2c = run_validation(
                model_a2c,
                validation_df,
                "a2c",
                i,
                turbulence_threshold,
            )
            print(f"A2C Sharpe Ratio: {sharpe_a2c:.4f}")

            print("Training PPO...")
            model_ppo = train_ppo(train_df, f"PPO_100k_dow_{i}", i)
            sharpe_ppo = run_validation(
                model_ppo,
                validation_df,
                "ppo",
                i,
                turbulence_threshold,
            )
            print(f"PPO Sharpe Ratio: {sharpe_ppo:.4f}")

            print("Training DDPG...")
            model_ddpg = train_ddpg(train_df, f"DDPG_10k_dow_{i}", i)
            sharpe_ddpg = run_validation(
                model_ddpg,
                validation_df,
                "ddpg",
                i,
                turbulence_threshold,
            )
            print(f"DDPG Sharpe Ratio: {sharpe_ddpg:.4f}")

            sharpe_map = {
                "A2C": sharpe_a2c,
                "PPO": sharpe_ppo,
                "DDPG": sharpe_ddpg,
            }
            model_map = {
                "A2C": model_a2c,
                "PPO": model_ppo,
                "DDPG": model_ddpg,
            }
            selected_model_name = max(sharpe_map, key=sharpe_map.get)
            selected_model = model_map[selected_model_name]
            print(f"Selected model: {selected_model_name}")

            last_state_ensemble = run_trade(
                df=df,
                model=selected_model,
                label="ensemble",
                last_state=last_state_ensemble,
                iteration=i,
                unique_trade_date=unique_trade_date,
                threshold=turbulence_threshold,
                initial=initial,
            )

            summary_rows.append(
                {
                    "iteration": i,
                    "validation_start": int(validation_start),
                    "validation_end": int(validation_end),
                    "trade_end": int(trade_end),
                    "turbulence_threshold": turbulence_threshold,
                    "a2c_sharpe": sharpe_a2c,
                    "ppo_sharpe": sharpe_ppo,
                    "ddpg_sharpe": sharpe_ddpg,
                    "selected_model": selected_model_name,
                }
            )

        elapsed = datetime.now() - start_time
        print(f"Native ensemble strategy finished in {elapsed}.")
        summary_df = pd.DataFrame(summary_rows)
        summary_path = config.paths.run_dir / "ensemble_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"Saved run summary: {summary_path}")

    return SimpleNamespace(
        load_or_create_preprocessed_data=load_or_create_preprocessed_data,
        run_ensemble_strategy=run_ensemble_strategy,
    )


def save_run_metadata(config: BaselineConfig) -> None:
    config.paths.run_dir.mkdir(parents=True, exist_ok=True)
    config.paths.results_dir.mkdir(parents=True, exist_ok=True)
    config.paths.trained_models_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "source_dir": str(config.paths.source_dir),
        "data_file": str(config.paths.data_file),
        "output_dir": str(config.paths.output_dir),
        "run_dir": str(config.paths.run_dir),
        "results_dir": str(config.paths.results_dir),
        "trained_models_dir": str(config.paths.trained_models_dir),
        "rebalance_window": config.rebalance_window,
        "validation_window": config.validation_window,
        "a2c_timesteps": config.a2c_timesteps,
        "ppo_timesteps": config.ppo_timesteps,
        "ddpg_timesteps": config.ddpg_timesteps,
        "seed": config.seed,
        "refresh_cache": config.refresh_cache,
        "quick": config.quick,
        "created_at": datetime.now().isoformat(),
    }
    metadata_path = config.paths.run_dir / "run_config.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved run metadata: {metadata_path}")


def main() -> int:
    args = parse_args()
    config = build_config(args)

    if args.check:
        return print_check_report(config)

    missing = validate_source_layout(config)
    if missing:
        print("Baseline source layout check failed.")
        for item in missing:
            print(f"- Missing: {item}")
        print("Run `python baseline_native.py --check` after fixing the paths.")
        return 1

    try:
        libs = load_runtime_modules()
    except RuntimeError as exc:
        print(exc)
        return 1

    save_run_metadata(config)
    runtime = create_runtime(config, libs)
    data = runtime.load_or_create_preprocessed_data()

    print(data.head())
    print(f"Loaded rows x columns: {data.shape[0]} x {data.shape[1]}")

    unique_trade_date = data[
        (data.datadate > 20151001) & (data.datadate <= 20200707)
    ].datadate.unique()
    print(f"Unique trade windows available: {len(unique_trade_date)}")

    runtime.run_ensemble_strategy(data, unique_trade_date)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

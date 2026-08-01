"""Walk-forward training/evaluation pipeline.

Protocol (docs/design_plan.md §7):
  - Anchored folds from data/processed/splits.json with a 21-day embargo
    between train/val and val/test.
  - Normalizer fit on the train window only.
  - Per (config, fold, seed): train on train window, record VALIDATION metrics
    for model selection, and TEST metrics used only for final reporting.
  - Every scored run is appended to the trial registry (results/registry.jsonl)
    for later Deflated-Sharpe accounting.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .data import MarketData, Normalizer, load_market, PROCESSED, ROOT
from .env import ExposureTradingEnv, run_policy
from .metrics import perf, turnover_stats
from .stats import sharpe

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
EMBARGO_DAYS = 21


@dataclass
class FoldSpec:
    name: str
    train_start: str
    train_end: str
    val_start: str
    val_end: str
    test_start: str
    test_end: str


def load_folds(symbol_start: str = "1994-01-01") -> list[FoldSpec]:
    splits = json.loads((PROCESSED / "splits.json").read_text())
    folds = []
    for f in splits["folds"]:
        train_end = pd.Timestamp(f["train_end"])
        val_end = pd.Timestamp(f["val_end"])
        test_end = pd.Timestamp(f["test_end"])
        emb = pd.Timedelta(days=EMBARGO_DAYS * 7 // 5 + 7)  # calendar approx of 21 trading days
        folds.append(FoldSpec(
            name=f["name"],
            train_start=symbol_start,
            train_end=str(train_end.date()),
            val_start=str((train_end + emb).date()),
            val_end=str(val_end.date()),
            test_start=str((val_end + emb).date()),
            test_end=str(test_end.date()),
        ))
    return folds


@dataclass
class RunRecord:
    config: str
    fold: str
    seed: int
    symbol: str
    cost_bps: float
    timesteps: int
    val_sharpe: float
    val_cagr: float
    test_sharpe: float
    test_cagr: float
    test_max_dd: float
    test_avg_exposure: float
    test_ann_turnover: float
    wall_seconds: float
    extra: dict = field(default_factory=dict)


def append_registry(rec: RunRecord, path: Path | None = None) -> None:
    p = path or (RESULTS / "registry.jsonl")
    with p.open("a") as fh:
        fh.write(json.dumps(asdict(rec)) + "\n")


def make_ppo(env, seed: int, hp: dict):
    from stable_baselines3 import PPO
    return PPO(
        "MlpPolicy", env, seed=seed, verbose=0,
        learning_rate=hp.get("learning_rate", 3e-4),
        n_steps=hp.get("n_steps", 512),
        batch_size=hp.get("batch_size", 256),
        gamma=hp.get("gamma", 0.99),
        gae_lambda=hp.get("gae_lambda", 0.95),
        ent_coef=hp.get("ent_coef", 0.01),
        clip_range=hp.get("clip_range", 0.2),
        n_epochs=hp.get("n_epochs", 10),
        policy_kwargs=dict(net_arch=hp.get("net_arch", [64, 64])),
        device="cpu",
    )


def train_and_eval_one(
    market: MarketData,
    fold: FoldSpec,
    seed: int,
    config_name: str = "ppo_default",
    hp: dict | None = None,
    cost_bps: float = 2.0,
    timesteps: int = 150_000,
    episode_len: int = 252,
    n_envs: int = 4,
    save_series: bool = True,
) -> RunRecord:
    """Train one PPO agent on the fold's train window; score on val and test."""
    import torch
    torch.set_num_threads(1)
    from stable_baselines3.common.vec_env import DummyVecEnv

    hp = hp or {}
    t0 = time.time()

    train = market.slice(fold.train_start, fold.train_end)
    val = market.slice(fold.val_start, fold.val_end)
    test = market.slice(fold.test_start, fold.test_end)
    norm = Normalizer.fit(train.feat)
    reward_lambda = hp.get("reward_lambda", 0.0)

    # WP-D: mix in stationary-bootstrap synthetic paths (generated from the
    # train window only). n_boot_paths > 0 dedicates that many parallel envs
    # to synthetic paths; the rest run the real path.
    n_boot = hp.get("n_boot_paths", 0)
    train_sets = [train] * n_envs
    if n_boot > 0:
        from .synth import bootstrap_path
        for i in range(min(n_boot, n_envs - 1)):
            train_sets[i + 1] = bootstrap_path(train, seed=seed * 100 + i)

    discrete = hp.get("discrete", True)
    residual = hp.get("residual", False)
    switch_penalty = hp.get("switch_penalty_bps", 0.0)

    def mk(i):
        return lambda: ExposureTradingEnv(
            train_sets[i], norm, cost_bps=cost_bps, discrete=discrete,
            episode_len=episode_len, seed=seed * 1000 + i,
            reward_lambda=reward_lambda, residual=residual,
            switch_penalty_bps=switch_penalty)

    venv = DummyVecEnv([mk(i) for i in range(n_envs)])
    model = make_ppo(venv, seed, hp)
    model.learn(total_timesteps=timesteps, progress_bar=False)

    val_run = run_policy(val, norm, model, cost_bps=cost_bps, discrete=discrete,
                         residual=residual)
    test_run = run_policy(test, norm, model, cost_bps=cost_bps, discrete=discrete,
                          residual=residual)
    val_p = perf(val_run["net"], val.cash)
    test_p = perf(test_run["net"], test.cash)
    test_to = turnover_stats(test_run["exposure"])

    rec = RunRecord(
        config=config_name, fold=fold.name, seed=seed, symbol=market.symbol,
        cost_bps=cost_bps, timesteps=timesteps,
        val_sharpe=val_p.get("sharpe", np.nan), val_cagr=val_p.get("cagr", np.nan),
        test_sharpe=test_p.get("sharpe", np.nan), test_cagr=test_p.get("cagr", np.nan),
        test_max_dd=test_p.get("max_dd", np.nan),
        test_avg_exposure=test_to["avg_exposure"],
        test_ann_turnover=test_to["ann_turnover"],
        wall_seconds=round(time.time() - t0, 1),
    )
    append_registry(rec)

    if save_series:
        out = RESULTS / "series" / f"{config_name}_{market.symbol}_{fold.name}_s{seed}.npz"
        out.parent.mkdir(exist_ok=True)
        np.savez_compressed(
            out,
            test_net=test_run["net"], test_exposure=test_run["exposure"],
            test_dates=test.index.to_numpy(dtype="datetime64[ns]").astype("int64"),
            val_net=val_run["net"],
            val_dates=val.index.to_numpy(dtype="datetime64[ns]").astype("int64"),
        )
    return rec

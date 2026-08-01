"""Known-optimum sanity check (Lu 2023 requirement).

Build a synthetic market whose returns are partially predictable from one
feature, plus pure-noise distractor features. A competent training stack must:
  1. learn a policy that beats buy-and-hold on held-out synthetic data
     when a strong signal exists;
  2. NOT hallucinate an edge when the signal feature is shuffled (placebo).

If (1) fails, null results on real data are uninformative (the stack can't
learn). If (2) fails, the stack manufactures false positives.

Marked slow: ~2-3 min. Run with `pytest tests/test_planted_signal.py -m ''`.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rlqqq.data import MarketData, Normalizer
from rlqqq.env import ExposureTradingEnv, portfolio_returns, run_policy


def make_synthetic(n=6000, signal_strength=0.6, seed=0, shuffle_signal=False):
    """Returns are mu_t + noise where mu_t is visible in feature 0 one day
    ahead. signal_strength scales predictability. 4 distractor features."""
    rng = np.random.default_rng(seed)
    daily_vol = 0.01
    mu = signal_strength * daily_vol * rng.standard_normal(n)   # predictable part
    noise = daily_vol * rng.standard_normal(n)
    ret = mu + noise
    sig_feat = mu / daily_vol                                    # known at t for ret[t]
    if shuffle_signal:
        sig_feat = rng.permutation(sig_feat)
    feat = np.column_stack([
        sig_feat,
        rng.standard_normal((n, 4)),
    ])
    idx = pd.bdate_range("2000-01-03", periods=n)
    cash = np.full(n, 0.02 / 252)
    return MarketData("SYNTH", pd.DatetimeIndex(idx), ret, cash, feat,
                      ["signal", "n1", "n2", "n3", "n4"])


def train_ppo_on(data, timesteps=60_000, seed=0):
    import torch
    torch.set_num_threads(2)
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    train = MarketData(data.symbol, data.index[:4000], data.ret[:4000],
                       data.cash[:4000], data.feat[:4000], data.feat_names)
    test = MarketData(data.symbol, data.index[4000:], data.ret[4000:],
                      data.cash[4000:], data.feat[4000:], data.feat_names)
    norm = Normalizer.fit(train.feat)
    venv = DummyVecEnv([
        (lambda i=i: ExposureTradingEnv(train, norm, cost_bps=1.0,
                                        episode_len=252, seed=seed * 100 + i))
        for i in range(4)
    ])
    model = PPO("MlpPolicy", venv, seed=seed, verbose=0, n_steps=512,
                batch_size=256, ent_coef=0.01, learning_rate=3e-4,
                policy_kwargs=dict(net_arch=[64, 64]), device="cpu")
    model.learn(total_timesteps=timesteps)
    run = run_policy(test, norm, model, cost_bps=1.0)
    bh = portfolio_returns(np.ones(len(test)), test.ret, test.cash, 1.0)
    agent_sh = run["net"].mean() / run["net"].std(ddof=1) * np.sqrt(252)
    bh_sh = bh.mean() / bh.std(ddof=1) * np.sqrt(252)
    return agent_sh, bh_sh


@pytest.mark.slow
def test_ppo_learns_planted_signal():
    data = make_synthetic(signal_strength=0.6, seed=42)
    agent_sh, bh_sh = train_ppo_on(data, seed=0)
    # with 0.6-strength signal the oracle Sharpe is huge; the agent must at
    # least clearly beat buy-and-hold on held-out data
    assert agent_sh > bh_sh + 1.0, (
        f"stack failed to learn planted signal: agent {agent_sh:.2f} "
        f"vs B&H {bh_sh:.2f}")


@pytest.mark.slow
def test_ppo_no_edge_on_placebo():
    data = make_synthetic(signal_strength=0.6, seed=42, shuffle_signal=True)
    agent_sh, bh_sh = train_ppo_on(data, seed=0)
    # shuffled signal -> no information; agent must not beat B&H materially
    assert agent_sh < bh_sh + 0.5, (
        f"stack hallucinated an edge on placebo: agent {agent_sh:.2f} "
        f"vs B&H {bh_sh:.2f}")

"""Single-asset daily exposure-trading environment.

Design (see docs/design_plan.md §5):
  * State at step t: normalized features known at close of day t, plus the
    current exposure (so the policy can reason about switching costs).
  * Action: target exposure w in [0, 1]. Discrete mode maps action index into
    the EXPOSURE_LEVELS grid; continuous mode is Box(0, 1).
  * The chosen exposure is applied from the close of day t: the portfolio
    earns w * ret[t] + (1 - w) * cash[t] over t -> t+1, minus one-way costs
    cost_bps * |w_t - w_{t-1}|.
  * Reward: log(1 + net portfolio return). Sum of rewards == log total wealth
    multiple, so the return-maximizing policy maximizes terminal wealth.
  * No leverage, no shorting: this keeps the comparison with buy-and-hold
    honest (an always-1.0 policy IS buy-and-hold, and a test asserts that).

Accounting identity used everywhere (env, baselines, backtests):
    net_t = w_t * ret_t + (1 - w_t) * cash_t - (cost_bps/1e4) * |w_t - w_{t-1}|
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from .data import MarketData, Normalizer

EXPOSURE_LEVELS = np.array([0.0, 0.5, 1.0])
RESIDUAL_MULTIPLIERS = np.array([0.5, 1.0, 1.5])


def causal_vol_target(ret: np.ndarray, target: float = 0.10,
                      lookback: int = 21) -> np.ndarray:
    """Vol-target exposure computable on ANY return series (incl. synthetic
    bootstrap paths). ret[t] is the t->t+1 return; the exposure decided at
    close t uses only returns realized by close t, i.e. ret[t-lookback..t-1].
    """
    r = pd.Series(np.asarray(ret, dtype=np.float64))
    rv = (r.rolling(lookback).std() * np.sqrt(252)).shift(1)
    w = (target / rv).clip(upper=1.0)
    return w.fillna(0.5).to_numpy()  # neutral default during warmup


def portfolio_returns(
    exposure: np.ndarray,
    ret: np.ndarray,
    cash: np.ndarray,
    cost_bps: float,
    w_init: float = 0.0,
) -> np.ndarray:
    """Vectorized accounting identity. exposure[t] is the target held over
    t -> t+1. Used by baselines and the independent backtest cross-check."""
    w = np.asarray(exposure, dtype=np.float64)
    prev = np.concatenate([[w_init], w[:-1]])
    turnover = np.abs(w - prev)
    return w * ret + (1.0 - w) * cash - (cost_bps / 1e4) * turnover


class ExposureTradingEnv(gym.Env):
    """Gymnasium env over a MarketData slice.

    Episodes run the full slice by default; for training, `episode_len`
    samples random contiguous sub-windows to decorrelate rollouts.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        data: MarketData,
        normalizer: Normalizer,
        cost_bps: float = 2.0,
        discrete: bool = True,
        episode_len: int | None = None,
        w_init: float = 0.0,
        seed: int | None = None,
        reward_lambda: float = 0.0,
        residual: bool = False,
        switch_penalty_bps: float = 0.0,
    ):
        super().__init__()
        self.data = data
        self.normalizer = normalizer
        self.cost_bps = float(cost_bps)
        self.discrete = discrete
        self.episode_len = episode_len
        self.w_init = float(w_init)
        # risk penalty: reward -= lambda * net^2 (a mean-variance shaping term;
        # lambda=0 recovers pure log-wealth). Training-only knob - evaluation
        # always uses raw net returns.
        self.reward_lambda = float(reward_lambda)
        # residual mode: actions are multipliers on a causal vol-target
        # baseline; action index 1 (x1.0) IS the baseline policy.
        self.residual = residual
        # training-only extra cost per unit turnover (bps) - shapes the policy
        # toward fewer switches; realized accounting still uses cost_bps.
        self.switch_penalty_bps = float(switch_penalty_bps)
        self._rng = np.random.default_rng(seed)
        self._baseline = causal_vol_target(data.ret) if residual else None

        n_feat = data.feat.shape[1]
        # observation = normalized features + current exposure (+ baseline w)
        obs_dim = n_feat + 1 + (1 if residual else 0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        if residual:
            self.action_space = spaces.Discrete(len(RESIDUAL_MULTIPLIERS))
        elif discrete:
            self.action_space = spaces.Discrete(len(EXPOSURE_LEVELS))
        else:
            self.action_space = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)

        self._feat_norm = self.normalizer(self.data.feat).astype(np.float32)
        self._t = 0
        self._end = len(data)
        self._w = self.w_init

    # -- helpers ---------------------------------------------------------

    def _obs(self) -> np.ndarray:
        parts = [self._feat_norm[self._t], np.float32([self._w])]
        if self.residual:
            parts.append(np.float32([self._baseline[self._t]]))
        return np.concatenate(parts).astype(np.float32)

    def _action_to_exposure(self, action: Any) -> float:
        if self.residual:
            mult = RESIDUAL_MULTIPLIERS[int(action)]
            return float(np.clip(mult * self._baseline[self._t], 0.0, 1.0))
        if self.discrete:
            return float(EXPOSURE_LEVELS[int(action)])
        return float(np.clip(np.asarray(action).reshape(-1)[0], 0.0, 1.0))

    # -- gym API ---------------------------------------------------------

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        n = len(self.data)
        if self.episode_len is not None and self.episode_len < n:
            start = int(self._rng.integers(0, n - self.episode_len))
            self._t = start
            self._end = start + self.episode_len
        else:
            self._t = 0
            self._end = n
        self._w = self.w_init
        return self._obs(), {}

    def step(self, action):
        w = self._action_to_exposure(action)
        t = self._t
        turnover = abs(w - self._w)
        net = (
            w * self.data.ret[t]
            + (1.0 - w) * self.data.cash[t]
            - (self.cost_bps / 1e4) * turnover
        )
        reward = (float(np.log1p(net))
                  - self.reward_lambda * float(net) ** 2
                  - (self.switch_penalty_bps / 1e4) * turnover)
        self._w = w
        self._t += 1
        terminated = self._t >= self._end
        info = {"exposure": w, "net_return": net, "turnover": turnover,
                "date": self.data.index[t]}
        obs = self._obs() if not terminated else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        return obs, reward, terminated, False, info


def run_policy(
    env_data: MarketData,
    normalizer: Normalizer,
    policy,
    cost_bps: float = 2.0,
    discrete: bool = True,
    deterministic: bool = True,
    residual: bool = False,
) -> dict:
    """Roll a trained SB3 policy over a full data slice once and return the
    exposure series + net daily returns (via the shared accounting identity)."""
    env = ExposureTradingEnv(env_data, normalizer, cost_bps=cost_bps,
                             discrete=discrete, episode_len=None,
                             residual=residual)
    obs, _ = env.reset()
    exposures = np.empty(len(env_data))
    for i in range(len(env_data)):
        action, _ = policy.predict(obs, deterministic=deterministic)
        obs, _, term, _, info = env.step(action)
        exposures[i] = info["exposure"]
        if term:
            break
    net = portfolio_returns(exposures, env_data.ret, env_data.cash, cost_bps)
    return {"exposure": exposures, "net": net, "index": env_data.index}

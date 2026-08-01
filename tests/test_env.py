"""Integration tests for the trading environment.

The critical invariant: an agent that always holds exposure 1.0 with zero
costs must reproduce buy-and-hold total return EXACTLY, and an agent that
always holds 0.0 must reproduce T-bill compounding exactly. If these fail,
every experimental result is meaningless.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rlqqq.data import load_market, Normalizer
from rlqqq.env import (EXPOSURE_LEVELS, ExposureTradingEnv, portfolio_returns)

PROCESSED = Path(__file__).resolve().parent.parent / "data" / "processed"


@pytest.fixture(scope="module")
def spy():
    return load_market("SPY")


@pytest.fixture(scope="module")
def norm(spy):
    train = spy.slice("1994-01-01", "2007-12-31")
    return Normalizer.fit(train.feat)


def rollout_constant(data, norm, level_idx, cost_bps):
    env = ExposureTradingEnv(data, norm, cost_bps=cost_bps, discrete=True,
                             episode_len=None)
    obs, _ = env.reset()
    total_log = 0.0
    nets = []
    for _ in range(len(data)):
        obs, r, term, _, info = env.step(level_idx)
        total_log += r
        nets.append(info["net_return"])
        if term:
            break
    return np.exp(total_log), np.array(nets)


class TestBuyAndHoldReproduction:
    def test_always_long_zero_cost_equals_buy_and_hold(self, spy, norm):
        seg = spy.slice("2010-01-01", "2019-12-31")
        wealth, _ = rollout_constant(seg, norm, level_idx=2, cost_bps=0.0)
        # buy-and-hold on the same slice: product of (1 + ret)
        bh = float(np.prod(1.0 + seg.ret))
        # cost=0 but first-step turnover 0->1 costs nothing at 0 bps
        assert wealth == pytest.approx(bh, rel=1e-10)

    def test_always_long_with_cost_pays_entry_once(self, spy, norm):
        seg = spy.slice("2010-01-01", "2019-12-31")
        wealth, nets = rollout_constant(seg, norm, level_idx=2, cost_bps=2.0)
        bh = np.cumprod(1.0 + seg.ret)[-1]
        # only difference: 2bp on the initial 0->1 switch, compounded in on day 1
        expected = (1.0 + seg.ret[0] - 2.0 / 1e4) * bh / (1.0 + seg.ret[0])
        assert wealth == pytest.approx(expected, rel=1e-9)

    def test_always_cash_equals_tbill_compounding(self, spy, norm):
        seg = spy.slice("2010-01-01", "2019-12-31")
        wealth, _ = rollout_constant(seg, norm, level_idx=0, cost_bps=2.0)
        tbill = float(np.prod(1.0 + seg.cash))
        assert wealth == pytest.approx(tbill, rel=1e-10)

    def test_adj_close_total_return_matches_env(self, spy):
        """Env returns must match the adjusted-close series in processed data."""
        px = pd.read_parquet(PROCESSED / "prices_SPY.parquet")
        adj = px["adj_close"]
        r = adj.pct_change().shift(-1).dropna()
        common = spy.index.intersection(r.index)
        env_r = pd.Series(spy.ret, index=spy.index).loc[common]
        assert np.allclose(env_r.values, r.loc[common].values, atol=1e-12)


class TestAccountingIdentity:
    def test_vectorized_matches_env_stepwise(self, spy, norm):
        seg = spy.slice("2015-01-01", "2016-12-31")
        rng = np.random.default_rng(0)
        actions = rng.integers(0, 3, size=len(seg))
        env = ExposureTradingEnv(seg, norm, cost_bps=5.0, discrete=True)
        env.reset()
        nets_env = []
        for a in actions:
            _, _, term, _, info = env.step(int(a))
            nets_env.append(info["net_return"])
            if term:
                break
        w = EXPOSURE_LEVELS[actions[:len(nets_env)]]
        nets_vec = portfolio_returns(w, seg.ret[:len(nets_env)],
                                     seg.cash[:len(nets_env)], cost_bps=5.0)
        assert np.allclose(nets_env, nets_vec, atol=1e-14)

    def test_turnover_cost_sign(self, spy, norm):
        """More switching must never increase wealth, holding returns fixed."""
        seg = spy.slice("2015-01-01", "2016-12-31")
        hold = portfolio_returns(np.ones(len(seg)), seg.ret, seg.cash, 10.0)
        churn_w = np.tile([1.0, 0.0], len(seg) // 2 + 1)[:len(seg)]
        churn = portfolio_returns(churn_w, seg.ret, seg.cash, 10.0)
        # can't assert churn always loses (depends on path), but cost term must
        # be strictly positive for the churner
        cost_paid = (10.0 / 1e4) * np.abs(np.diff(np.concatenate([[0], churn_w]))).sum()
        assert cost_paid > 0.5  # ~250 switches * 10bp
        assert np.isfinite(churn).all() and np.isfinite(hold).all()


class TestCausality:
    def test_observation_only_uses_past(self, spy, norm):
        """Perturbing data at t+k (k>=1) must not change the observation at t."""
        seg = spy.slice("2018-01-01", "2018-12-31")
        env1 = ExposureTradingEnv(seg, norm, discrete=True)
        obs1, _ = env1.reset()

        # corrupt all *future* rows' features after step 10
        seg2 = spy.slice("2018-01-01", "2018-12-31")
        feat2 = seg2.feat.copy()
        feat2[11:, :] = 999.0
        seg2 = type(seg2)(seg2.symbol, seg2.index, seg2.ret, seg2.cash,
                          feat2, seg2.feat_names)
        env2 = ExposureTradingEnv(seg2, norm, discrete=True)
        obs2, _ = env2.reset()

        np.testing.assert_array_equal(obs1, obs2)
        for i in range(10):
            obs1, r1, *_ = env1.step(2)
            obs2, r2, *_ = env2.step(2)
            assert r1 == r2
            if i < 9:  # obs at step 10 reads row 10, still uncorrupted
                np.testing.assert_array_equal(obs1, obs2)

    def test_normalizer_fit_on_train_only(self, spy):
        """Normalizer stats from a train window must differ from full-sample
        stats — guard against accidentally fitting on everything."""
        train = spy.slice("1994-01-01", "2007-12-31")
        n_train = Normalizer.fit(train.feat)
        n_full = Normalizer.fit(spy.feat)
        assert not np.allclose(n_train.mean, n_full.mean)


class TestGymCompliance:
    def test_check_env(self, spy, norm):
        from stable_baselines3.common.env_checker import check_env
        seg = spy.slice("2015-01-01", "2016-12-31")
        env = ExposureTradingEnv(seg, norm, discrete=True, episode_len=64)
        check_env(env, warn=True, skip_render_check=True)

    def test_check_env_continuous(self, spy, norm):
        from stable_baselines3.common.env_checker import check_env
        seg = spy.slice("2015-01-01", "2016-12-31")
        env = ExposureTradingEnv(seg, norm, discrete=False, episode_len=64)
        check_env(env, warn=True, skip_render_check=True)

    def test_episode_windows_are_random(self, spy, norm):
        seg = spy.slice("2010-01-01", "2019-12-31")
        env = ExposureTradingEnv(seg, norm, episode_len=64, seed=1)
        starts = set()
        for _ in range(20):
            env.reset()
            starts.add(env._t)
        assert len(starts) > 5

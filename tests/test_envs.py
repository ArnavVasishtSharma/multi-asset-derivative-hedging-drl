"""
tests/test_envs.py
-------------------
Smoke tests for MultiAssetHedgingEnv and DeFiLPEnv.
All tests use synthetic parquet fixtures from conftest.py.
"""

import numpy as np
import pytest

from envs.multi_asset_env import MultiAssetHedgingEnv
from envs.defi_env import DeFiLPEnv, DeFiHedgingEnv


# ── MultiAssetHedgingEnv ──────────────────────────────────────────────────────

class TestMultiAssetHedgingEnv:
    """Tests for the core multi-asset hedging environment."""

    def test_reset_obs_shape(self, master_parquet_path):
        env = MultiAssetHedgingEnv(data_path=master_parquet_path, train=True, seed=0)
        obs, info = env.reset()
        assert obs.shape == (49,), f"Expected obs shape (49,), got {obs.shape}"
        assert obs.dtype == np.float32

    def test_step_returns_correct_tuple(self, master_parquet_path):
        env = MultiAssetHedgingEnv(data_path=master_parquet_path, train=True, seed=0)
        obs, _ = env.reset()
        action = np.array([0.1, -0.2, 0.3], dtype=np.float32)
        result = env.step(action)
        assert len(result) == 5, "step() should return (obs, reward, terminated, truncated, info)"
        next_obs, reward, terminated, truncated, info = result
        assert next_obs.shape == (49,)
        assert np.isfinite(reward), f"Reward is not finite: {reward}"
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "hedging_error" in info
        assert "tx_cost" in info
        assert "cvar" in info
        assert "position" in info

    def test_episode_terminates_at_episode_len(self, master_parquet_path):
        ep_len = 30
        env = MultiAssetHedgingEnv(
            data_path=master_parquet_path, train=True,
            episode_len=ep_len, seed=0,
        )
        obs, _ = env.reset()
        for step in range(ep_len):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if step < ep_len - 1:
                assert not terminated, f"Terminated early at step {step}"
        assert terminated, "Episode should terminate at episode_len"

    def test_obs_values_clipped(self, master_parquet_path):
        env = MultiAssetHedgingEnv(data_path=master_parquet_path, train=True, seed=0)
        obs, _ = env.reset()
        assert np.all(obs >= -10.0) and np.all(obs <= 10.0), "Obs should be clipped to [-10, 10]"

    def test_action_space_shape(self, master_parquet_path):
        env = MultiAssetHedgingEnv(data_path=master_parquet_path, train=True, seed=0)
        assert env.action_space.shape == (3,)
        assert env.observation_space.shape == (49,)

    def test_test_split(self, master_parquet_path):
        env_train = MultiAssetHedgingEnv(data_path=master_parquet_path, train=True, seed=0)
        env_test  = MultiAssetHedgingEnv(data_path=master_parquet_path, train=False, seed=0)
        # Test split should have fewer data points (20%)
        assert env_test.n_steps < env_train.n_steps


# ── DeFiLPEnv ─────────────────────────────────────────────────────────────────

class TestDeFiLPEnv:
    """Tests for the DeFi concentrated-liquidity hedging environment."""

    def test_reset_obs_shape(self, defi_parquet_path):
        env = DeFiLPEnv(data_path=defi_parquet_path, train=True, seed=0)
        obs, info = env.reset()
        assert obs.shape == (14,), f"Expected obs shape (14,), got {obs.shape}"
        assert obs.dtype == np.float32

    def test_step_returns_correct_keys(self, defi_parquet_path):
        env = DeFiLPEnv(data_path=defi_parquet_path, train=True, seed=0)
        obs, _ = env.reset()
        action = np.array([0.1, 0.0, 0.5], dtype=np.float32)
        _, _, _, _, info = env.step(action)
        assert "fee_income" in info
        assert "il" in info
        assert "in_range" in info
        assert "drawdown" in info
        assert "cumulative_pnl" in info

    def test_rebalance_updates_tick_range(self, defi_parquet_path):
        env = DeFiLPEnv(data_path=defi_parquet_path, train=True, seed=0)
        env.reset()
        # Force rebalance with rebalance_flag > 0
        action = np.array([0.2, 0.0, 1.0], dtype=np.float32)
        env.step(action)
        assert env.tick_lower > 0, "tick_lower should be set after rebalance"
        assert env.tick_upper > env.tick_lower, "tick_upper > tick_lower after rebalance"

    def test_alias_exists(self):
        """DeFiHedgingEnv should be an alias for DeFiLPEnv."""
        assert DeFiHedgingEnv is DeFiLPEnv

    def test_episode_terminates(self, defi_parquet_path):
        ep_len = 20
        env = DeFiLPEnv(data_path=defi_parquet_path, train=True, episode_len=ep_len, seed=0)
        env.reset()
        done = False
        steps = 0
        while not done:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1
        assert steps == ep_len

"""
envs/multi_asset_env.py
------------------------
Core multi-asset hedging Gym environment.

Observation space:
  [spx, spy, eurusd, usdjpy, rate_1y, rate_5y, rate_10y,   (7)
   iv_grid (25),                                             (25)
   corr_matrix_flat (9),                                     (9)
   greeks: delta, gamma, theta, vega (4),                    (4)
   time_to_expiry (1),                                       (1)
   position_equity, position_fx, position_rate (3)]          (3)
  Total: 49 dims

Action space:
  [delta_equity, delta_fx, delta_rate]  ∈ [-1, 1]^3
  (target hedge ratio per asset class)

Reward:
  r_t = -|HE_t|² - λ_cost × Σ|Δaction_t| - λ_cvar × CVaR_95
"""

import os
import logging
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

log = logging.getLogger(__name__)

# ── reward hyperparameters ────────────────────────────────────────────────────
LAMBDA_COST  = 0.001   # transaction cost penalty weight
LAMBDA_CVAR  = 0.1     # CVaR penalty weight
TX_COST_PROP = 0.0005  # proportional transaction cost (5 bps per trade)
TX_COST_FIX  = 0.0001  # fixed cost per rebalance


class MultiAssetHedgingEnv(gym.Env):
    """
    Multi-asset hedging environment supporting:
      - Equity (SPX/SPY)
      - FX (EURUSD/USDJPY)
      - Rates (US Treasuries 1Y/5Y/10Y)

    Designed for Novelty 1 (Multi-Asset DDPG) and Novelty 2 (BC-RPPO).

    Parameters
    ----------
    data_path  : Path to processed master_raw.parquet
    option_type: 'put' or 'call' — hedge instrument
    window_size: number of past timesteps to give as state context
    train      : If True, use training split (80%); else use test split (20%)
    episode_len: number of steps per episode (default: 60 trading days)
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path: str,
        option_type: str = "put",
        window_size: int = 1,
        train: bool = True,
        episode_len: int = 60,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.option_type  = option_type
        self.window_size  = window_size
        self.episode_len  = episode_len
        self.train        = train

        # Load data
        self.df = pd.read_parquet(data_path)
        n = len(self.df)
        split = int(n * 0.8)
        self.data = self.df.iloc[:split].values if train else self.df.iloc[split:].values
        self.n_steps = len(self.data)

        self._build_spaces()

        self.rng = np.random.default_rng(seed)
        self._reset_state()

    # ── Gym interface ─────────────────────────────────────────────────────────
    def _build_spaces(self):
        obs_dim = 49   # see module docstring
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

    def _reset_state(self):
        self.step_idx       = 0
        self.position       = np.zeros(3, dtype=np.float32)   # [eq, fx, rate]
        self.pnl_history: List[float] = []
        self.t_start        = 0

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None,
    ) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._reset_state()
        # Randomly sample episode start (ensuring enough steps)
        max_start = self.n_steps - self.episode_len - 1
        self.t_start = int(self.rng.integers(0, max(max_start, 1)))
        self.step_idx = 0
        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        t = self.t_start + self.step_idx
        row      = self.data[t]
        row_next = self.data[min(t + 1, self.n_steps - 1)]

        # Portfolio PnL attribution
        spot_now  = self._get_spot(row)
        spot_next = self._get_spot(row_next)
        asset_return = (spot_next - spot_now) / (spot_now + 1e-8)

        # Hedge P&L (position dot asset returns)
        hedge_pnl = float(np.dot(self.position, asset_return[:3]))

        # Unhedged option value change (simplified: ATM put delta ~ -0.5 for reference)
        option_delta = self._get_feature(row, "delta", default=-0.5)
        option_pnl   = option_delta * asset_return[0]   # equity-only option

        # Hedging error
        hedging_error = option_pnl + hedge_pnl   # close to 0 = perfect hedge

        # Transaction costs
        delta_action  = action - self.position
        tx_cost       = TX_COST_PROP * np.sum(np.abs(delta_action)) + TX_COST_FIX * np.any(delta_action != 0)

        # CVaR (running 95th percentile on negative PnL)
        self.pnl_history.append(hedging_error)
        cvar = self._compute_cvar(self.pnl_history)

        # Total reward
        reward = -(hedging_error**2) - LAMBDA_COST * tx_cost - LAMBDA_CVAR * cvar

        # Update position
        self.position = action.copy()
        self.step_idx += 1

        terminated = self.step_idx >= self.episode_len
        truncated  = False
        obs        = self._get_obs()
        info: Dict[str, Any] = {
            "hedging_error": hedging_error,
            "tx_cost":       tx_cost,
            "cvar":          cvar,
            "position":      self.position.copy(),
        }
        return obs, float(reward), terminated, truncated, info

    def render(self):
        t = self.t_start + self.step_idx
        row = self.data[min(t, self.n_steps - 1)]
        pos = self.position
        log.info(f"Step {self.step_idx:3d} | pos=[{pos[0]:.3f},{pos[1]:.3f},{pos[2]:.3f}]")

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _get_obs(self) -> np.ndarray:
        t = min(self.t_start + self.step_idx, self.n_steps - 1)
        row = self.data[t]
        # Build 49-dim observation
        # We index columns by name mapping; fall back to first columns if data
        # shape differs.
        market_features = row[:46].astype(np.float32)   # 7+25+9+4+1 = 46
        padding = np.zeros(max(0, 46 - len(market_features)), dtype=np.float32)
        market_features = np.concatenate([market_features, padding])[:46]
        obs = np.concatenate([market_features, self.position])
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def _get_spot(self, row: np.ndarray) -> np.ndarray:
        """Return [spx, eurusd, rate_10y] as spot prices (first 3 relevant cols)."""
        return row[:3]

    def _get_feature(self, row: np.ndarray, name: str, default: float = 0.0) -> float:
        col_map = {"delta": 41, "gamma": 42, "theta": 43, "vega": 44}
        idx = col_map.get(name, -1)
        if idx < len(row):
            return float(row[idx])
        return default

    @staticmethod
    def _compute_cvar(pnl_history: List[float], quantile: float = 0.05) -> float:
        """95% CVaR = expected loss in worst 5% of days."""
        if len(pnl_history) < 20:
            return 0.0
        arr = np.array(pnl_history)
        cutoff = np.quantile(arr, quantile)
        tail   = arr[arr <= cutoff]
        return float(-np.mean(tail)) if len(tail) > 0 else 0.0

"""
envs/defi_env.py
-----------------
Uniswap v3 LP hedging environment (Novelty 6 / Paper 6 basis).

Models a concentrated liquidity position as an approximated straddle payoff.
The agent manages tick ranges and position size to minimise impermanent loss
while collecting fees — analogous to gamma hedging in TradFi.

Observation space (14 dims):
  [pool_price, tick_lower, tick_upper, position_size,
   funding_rate, basis_spread, liquidity_depth,
   fee_accumulated, il_exposure, price_vol_30d,
   price_vol_7d, days_since_rebalance, fee_tier,
   pnl_since_open]

Action space (3 dims):
  [tick_range_pct, position_size_chg, rebalance_flag]
   tick_range_pct ∈ [0.01, 0.50]  (half-range as % of current price)
   position_size_chg ∈ [-1, 1]    (fractional change in LP capital)
   rebalance_flag ∈ [-1, 1]       (>0 triggers full rebalance)
"""

import logging
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
FEE_RATE      = 0.003   # 0.3% Uniswap pool fee tier
GAS_COST_ETH  = 0.01    # approximate gas cost per rebalance (normalised)


class DeFiLPEnv(gym.Env):
    """
    DeFi Concentrated Liquidity Position Hedging Environment.

    The LP faces impermanent loss (IL) when pool price drifts outside
    [tick_lower, tick_upper]. This env teaches the agent to:
      1. Widen tick range to reduce IL at the cost of less fees.
      2. Rebalance when price exits range.
      3. Size position to maximise fee income while controlling drawdown.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        data_path: str,
        train: bool = True,
        episode_len: int = 90,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.episode_len = episode_len
        self.train       = train

        df  = pd.read_parquet(data_path)
        n   = len(df)
        split = int(n * 0.8)
        self.data = df.iloc[:split].values if train else df.iloc[split:].values
        self.n_steps = len(self.data)

        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(14,), dtype=np.float32)
        self.action_space      = spaces.Box(
            low=np.array([0.01, -1.0, -1.0]),
            high=np.array([0.50, 1.0,  1.0]),
            dtype=np.float32
        )

        self.rng = np.random.default_rng(seed)
        self._reset_state()

    # ── Gym interface ─────────────────────────────────────────────────────────
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        self._reset_state()
        max_start = self.n_steps - self.episode_len - 1
        self.t_start   = int(self.rng.integers(0, max(max_start, 1)))
        self.step_idx  = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        action      = np.clip(action, self.action_space.low, self.action_space.high)
        tick_pct    = float(action[0])
        size_chg    = float(action[1])
        rebalance   = float(action[2]) > 0.0

        t   = self.t_start + self.step_idx
        row = self.data[min(t, self.n_steps - 1)]
        pool_price = row[0] if len(row) > 0 else 1800.0

        # Update position size
        self.position_size = float(np.clip(self.position_size * (1 + size_chg * 0.1), 0.01, 10.0))

        # Rebalance: re-centre tick range around current price
        if rebalance or not self._in_range(pool_price):
            self.tick_lower = pool_price * (1 - tick_pct)
            self.tick_upper = pool_price * (1 + tick_pct)
            self.rebalance_count += 1
            self.days_since_rebalance = 0
            gas_penalty = GAS_COST_ETH
        else:
            gas_penalty = 0.0
            self.days_since_rebalance += 1

        # Fee income: proportional to position size × fee_rate, reduced if out of range
        in_range   = self._in_range(pool_price)
        fee_income = FEE_RATE * self.position_size * (1.0 if in_range else 0.0) * 10.0

        # Impermanent loss (analytical approximation for v3 position)
        il = self._compute_il(pool_price)

        # Net PnL
        step_pnl = fee_income - il - gas_penalty
        self.cumulative_pnl += step_pnl

        # Reward: maximise net PnL, penalise drawdown
        drawdown = self._compute_drawdown()
        reward   = step_pnl - 0.1 * drawdown

        self.price_history.append(pool_price)
        self.pnl_history.append(step_pnl)

        self.step_idx += 1
        terminated  = self.step_idx >= self.episode_len
        obs = self._get_obs()
        info: Dict[str, Any] = {
            "fee_income":    fee_income,
            "il":            il,
            "in_range":      in_range,
            "drawdown":      drawdown,
            "cumulative_pnl": self.cumulative_pnl,
        }
        return obs, float(reward), terminated, False, info

    def render(self):
        t = self.t_start + self.step_idx
        log.info(f"Step {self.step_idx:3d} | PnL={self.cumulative_pnl:.4f} | "
                 f"Range=[{self.tick_lower:.1f}, {self.tick_upper:.1f}]")

    # ── Internals ─────────────────────────────────────────────────────────────
    def _reset_state(self):
        self.step_idx            = 0
        self.t_start             = 0
        self.position_size       = 1.0
        self.tick_lower          = 0.0
        self.tick_upper          = 0.0
        self.days_since_rebalance = 0
        self.rebalance_count     = 0
        self.cumulative_pnl      = 0.0
        self.price_history: List[float] = []
        self.pnl_history:   List[float] = []

    def _in_range(self, price: float) -> bool:
        return self.tick_lower <= price <= self.tick_upper

    def _compute_il(self, price_now: float) -> float:
        """
        Simplified Uniswap v3 impermanent loss for concentrated range.
        IL = 0 when price is within [tick_lower, tick_upper].
        IL grows rapidly outside range (mimics LP delta exposure).
        """
        if self.tick_upper <= self.tick_lower:
            return 0.0
        mid   = (self.tick_lower + self.tick_upper) / 2.0
        range_half = (self.tick_upper - self.tick_lower) / 2.0
        drift = abs(price_now - mid) / (range_half + 1e-6)
        if drift <= 1.0:
            return 0.0   # within range — no IL
        excess = drift - 1.0
        return float(0.5 * excess**2 * self.position_size)

    def _compute_drawdown(self) -> float:
        """Maximum drawdown from peak in PnL history."""
        if len(self.pnl_history) < 2:
            return 0.0
        cumulative = np.cumsum(self.pnl_history)
        peak   = np.maximum.accumulate(cumulative)
        dd     = peak - cumulative
        return float(np.max(dd))

    def _get_obs(self) -> np.ndarray:
        t   = min(self.t_start + self.step_idx, self.n_steps - 1)
        row = self.data[t]

        pool_price      = row[0] if len(row) > 0 else 1800.0
        funding_rate    = row[1] if len(row) > 1 else 0.0
        basis_spread    = row[2] if len(row) > 2 else 0.0
        liquidity       = row[3] if len(row) > 3 else 1e8
        in_range        = float(self._in_range(pool_price))
        il_exposure     = self._compute_il(pool_price)
        price_vol_30d   = (np.std(self.price_history[-30:]) if len(self.price_history) >= 30
                           else 0.05)
        price_vol_7d    = (np.std(self.price_history[-7:]) if len(self.price_history) >= 7
                           else 0.05)
        fee_accum       = float(sum(self.pnl_history))

        obs = np.array([
            pool_price / 2000.0,                       # normalised
            (self.tick_lower - pool_price) / pool_price,
            (self.tick_upper - pool_price) / pool_price,
            self.position_size,
            funding_rate * 1e4,
            basis_spread * 100,
            np.log(liquidity + 1) / 20.0,
            fee_accum,
            il_exposure,
            price_vol_30d / 100.0,
            price_vol_7d / 100.0,
            self.days_since_rebalance / 90.0,
            FEE_RATE,
            in_range,
        ], dtype=np.float32)
        return np.clip(obs, -10.0, 10.0)


# Alias used by training/eval scripts
DeFiHedgingEnv = DeFiLPEnv

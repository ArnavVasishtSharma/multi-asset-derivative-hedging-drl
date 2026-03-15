"""
models/novelty3_meta/defi_policy.py
--------------------------------------
DeFi Variable Policy (Novelty 3 — DeFi Arm).

Implements the DeFi sub-policy that manages Uniswap v3 concentrated liquidity
positions. This is the DeFi analogue of the TradFi DDPG arm.

Architecture (from Paper 6 — Variable Policy DRL):
  Three learnable sub-policies activated by a gating network:
    1. Concentrated  — tight tick range, high fees, high IL risk
    2. Wide          — broad tick range, lower fees, lower IL risk
    3. Single-sided  — one-sided LP provision (bullish/bearish on base asset)

State space (18-dim):
  [pool_price (1), tick_lower (1), tick_upper (1), fee_tier (1),
   il_exposure (1), funding_rate (1), eth_price (1), pool_vol (1),
   liquidity_depth (1), position_size (1), time_in_position (1),
   gas_price_gwei (1), pool_utilisation (1), basis_spread (1),
   on_chain_vol (1), defi_funding_rate (1), eth_dominance (1),
   regime_defi_prob (1)]

Action space (3-dim):
  [tick_lower_delta (normalised change), tick_upper_delta, position_size_frac]
  ∈ [-1, 1]^3  (environment maps these to Uniswap v3 ticks and amounts)

Reward:
  r_t = fee_income_t - gas_cost_t - impermanent_loss_t - λ_risk × CVaR_t
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


OBS_DIM    = 14   # matches DeFiLPEnv observation_space shape
ACTION_DIM = 3
HIDDEN_DIM = 256


class SubPolicy(nn.Module):
    """Single learnable sub-policy MLP for one liquidity concentration style."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden, action_dim)
        self.log_std_head = nn.Linear(hidden, action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, log_std)."""
        h       = self.net(x)
        mean    = torch.tanh(self.mean_head(h))
        log_std = self.log_std_head(h).clamp(-4.0, 2.0)
        return mean, log_std


class GatingNetwork(nn.Module):
    """
    Soft-gating network: weights the three sub-policies based on state.
    Output: softmax weights over [concentrated, wide, single_sided].
    """

    def __init__(self, obs_dim: int, n_policies: int = 3, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_policies),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns (B, n_policies) gate weights summing to 1."""
        return F.softmax(self.net(x), dim=-1)


class DeFiVariablePolicy(nn.Module):
    """
    DeFi Arm for Novelty 3: Variable Policy for Uniswap v3 LP management.

    Three sub-policies (concentrated, wide, single-sided) gated by a
    soft-gating network conditioned on the current state.

    Parameters
    ----------
    obs_dim    : state dimension (18)
    action_dim : action dimension (3)
    hidden_dim : sub-policy MLP hidden size
    """

    def __init__(
        self,
        obs_dim:    int = OBS_DIM,
        action_dim: int = ACTION_DIM,
        hidden_dim: int = HIDDEN_DIM,
    ):
        super().__init__()
        self.n_policies = 3
        self.sub_policies = nn.ModuleList([
            SubPolicy(obs_dim, action_dim, hidden_dim // 2)
            for _ in range(self.n_policies)
        ])
        self.gating = GatingNetwork(obs_dim, self.n_policies, hidden=64)

    def forward(
        self,
        obs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        obs : (B, obs_dim)

        Returns
        -------
        action      : (B, action_dim) — gated mixture of sub-policy means
        gate_weights: (B, 3)          — gating weights per sub-policy
        log_std     : (B, action_dim) — gated log-std
        """
        gates = self.gating(obs)        # (B, 3)

        means   = []
        log_stds = []
        for subpol in self.sub_policies:
            m, ls = subpol(obs)
            means.append(m)
            log_stds.append(ls)

        # Weighted mixture of means and log-stds
        means_stack    = torch.stack(means,    dim=1)    # (B, 3, action_dim)
        log_stds_stack = torch.stack(log_stds, dim=1)   # (B, 3, action_dim)

        g = gates.unsqueeze(-1)                          # (B, 3, 1)
        action  = (g * means_stack).sum(dim=1)           # (B, action_dim)
        log_std = (g * log_stds_stack).sum(dim=1)        # (B, action_dim)

        return action, gates, log_std

    @torch.no_grad()
    def select_action(self, obs_np) -> dict:
        """
        Select action from current DeFi state.

        Parameters
        ----------
        obs_np : (obs_dim,) numpy array

        Returns
        -------
        dict with:
          action        : (3,) numpy array [tick_lower_delta, tick_upper_delta, size_frac]
          gate_weights  : (3,) numpy array [p_concentrated, p_wide, p_single]
          active_style  : str — dominant LP style
        """
        import numpy as np
        obs_t = torch.FloatTensor(obs_np).unsqueeze(0)
        action, gates, _ = self.forward(obs_t)
        action_np = action.cpu().numpy().flatten()
        gates_np  = gates.cpu().numpy().flatten()
        styles    = ["concentrated", "wide", "single_sided"]
        return {
            "action":       action_np.astype(np.float32),
            "gate_weights": gates_np,
            "active_style": styles[int(gates_np.argmax())],
        }

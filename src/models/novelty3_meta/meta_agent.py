"""
models/novelty3_meta/meta_agent.py
-------------------------------------
Hybrid TradFi-DeFi Meta-Policy Agent (Novelty 3).

This is the top-level coordinator that:
  1. Runs the LSTM RegimeDetector to get [p_TradFi, p_DeFi, p_Neutral]
  2. Queries the TradFi Arm (MultiAssetDDPG from Novelty 1) for [δ_equity, δ_FX, δ_rate]
  3. Queries the DeFi Arm (DeFiVariablePolicy) for [tick_lower_delta, tick_upper_delta, size_frac]
  4. Blends outputs via regex probabilities into a delta-neutral aggregate position
  5. Trains the Executive Meta-Critic that learns joint Q_meta across both arms
  6. Backpropagates through the regime detector (end-to-end gradient flow)

Executive Meta-Critic (Q_meta):
  Input : concat(global_state, regime_probs, tradfi_action, defi_action)
  Output: scalar Q_meta (joint value)

Training:
  - Shared experience replay with combined TradFi+DeFi tuples
  - Soft-update target Meta-Critic
  - Regime detector gradients through Meta-Critic advantage

Portfolio Aggregator:
  final_delta_equity = p_TradFi × tradfi_delta_equity
  final_defi_size    = p_DeFi   × defi_size
  Subject to: |Σdeltas| ≈ 0  (delta-neutral constraint softly enforced)
"""

import os
import copy
import logging
from typing import Optional, Dict, Tuple, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.novelty3_meta.regime_detector import RegimeDetector
from models.novelty3_meta.defi_policy import DeFiVariablePolicy
from models.novelty1_ddpg.ddpg_agent import MultiAssetDDPG
from utils.replay_buffer import ReplayBuffer
from utils.metrics import compute_sharpe, compute_cvar

log = logging.getLogger(__name__)


# ── Executive Meta-Critic ───────────────────────────────────────────────────

class MetaCritic(nn.Module):
    """
    Executive Critic that estimates joint value across both arms.

    Input: concat(global_state, regime_probs, tradfi_action, defi_action)
    Output: Q_meta scalar
    """

    def __init__(
        self,
        global_state_dim: int = 49,   # same as multi-asset env obs
        regime_dim:       int = 3,
        tradfi_action_dim: int = 3,
        defi_action_dim:   int = 3,    # DeFiLPEnv action: (tick_pct, size_chg, rebalance)
        hidden_dim:        int = 256,
    ):
        super().__init__()
        in_dim = global_state_dim + regime_dim + tradfi_action_dim + defi_action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        global_state:   torch.Tensor,
        regime_probs:   torch.Tensor,
        tradfi_action:  torch.Tensor,
        defi_action:    torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([global_state, regime_probs, tradfi_action, defi_action], dim=-1)
        return self.net(x)


# ── Meta Replay Buffer entry ─────────────────────────────────────────────────

class MetaTransition:
    """Holds one combined TradFi+DeFi transition for the meta replay buffer."""
    __slots__ = ("global_obs", "regime_seq", "tradfi_obs", "defi_obs",
                 "tradfi_action", "defi_action", "reward_meta",
                 "next_global_obs", "next_regime_seq",
                 "next_tradfi_obs", "next_defi_obs", "done")


# ── Meta Replay Buffer ────────────────────────────────────────────────────────

class MetaReplayBuffer:
    """
    Combined replay buffer that stores multi-modal tuples for Novelty 3.
    """

    def __init__(
        self,
        global_obs_dim:   int = 49,
        regime_seq_shape: tuple = (20, 4),
        tradfi_obs_dim:   int = 49,
        defi_obs_dim:     int = 14,   # DeFiLPEnv actual obs_dim
        tradfi_action_dim: int = 3,
        defi_action_dim:   int = 3,   # DeFiLPEnv action_dim
        max_size:          int = 100_000,
    ):
        self.max_size = max_size
        self.ptr      = 0
        self.size     = 0

        self.global_obs       = np.zeros((max_size, global_obs_dim),     dtype=np.float32)
        self.regime_seqs      = np.zeros((max_size, *regime_seq_shape),  dtype=np.float32)
        self.tradfi_obs       = np.zeros((max_size, tradfi_obs_dim),     dtype=np.float32)
        self.defi_obs         = np.zeros((max_size, defi_obs_dim),       dtype=np.float32)
        self.tradfi_actions   = np.zeros((max_size, tradfi_action_dim),  dtype=np.float32)
        self.defi_actions     = np.zeros((max_size, defi_action_dim),    dtype=np.float32)
        self.rewards          = np.zeros((max_size, 1),                  dtype=np.float32)
        self.next_global_obs  = np.zeros((max_size, global_obs_dim),     dtype=np.float32)
        self.next_regime_seqs = np.zeros((max_size, *regime_seq_shape),  dtype=np.float32)
        self.next_tradfi_obs  = np.zeros((max_size, tradfi_obs_dim),     dtype=np.float32)
        self.next_defi_obs    = np.zeros((max_size, defi_obs_dim),       dtype=np.float32)
        self.dones            = np.zeros((max_size, 1),                  dtype=np.float32)

    def add(
        self,
        global_obs, regime_seq, tradfi_obs, defi_obs,
        tradfi_action, defi_action, reward_meta,
        next_global_obs, next_regime_seq, next_tradfi_obs, next_defi_obs,
        done: bool,
    ):
        p = self.ptr
        self.global_obs[p]       = global_obs
        self.regime_seqs[p]      = regime_seq
        self.tradfi_obs[p]       = tradfi_obs
        self.defi_obs[p]         = defi_obs
        self.tradfi_actions[p]   = tradfi_action
        self.defi_actions[p]     = defi_action
        self.rewards[p]          = reward_meta
        self.next_global_obs[p]  = next_global_obs
        self.next_regime_seqs[p] = next_regime_seq
        self.next_tradfi_obs[p]  = next_tradfi_obs
        self.next_defi_obs[p]    = next_defi_obs
        self.dones[p]            = float(done)
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int, device: torch.device) -> dict:
        idxs = np.random.randint(0, self.size, size=batch_size)
        f    = lambda x: torch.FloatTensor(x[idxs]).to(device)
        return {
            "global_obs":       f(self.global_obs),
            "regime_seqs":      f(self.regime_seqs),
            "tradfi_obs":       f(self.tradfi_obs),
            "defi_obs":         f(self.defi_obs),
            "tradfi_actions":   f(self.tradfi_actions),
            "defi_actions":     f(self.defi_actions),
            "rewards":          f(self.rewards),
            "next_global_obs":  f(self.next_global_obs),
            "next_regime_seqs": f(self.next_regime_seqs),
            "next_tradfi_obs":  f(self.next_tradfi_obs),
            "next_defi_obs":    f(self.next_defi_obs),
            "dones":            f(self.dones),
        }

    def ready(self, batch_size: int) -> bool:
        return self.size >= batch_size


# ── Hybrid TradFi-DeFi Meta-Policy Agent ─────────────────────────────────────

class HybridMetaPolicy:
    """
    Novelty 3: Hybrid TradFi-DeFi Meta-Policy.

    Coordinates:
      - RegimeDetector   (LSTM)          : market regime classification
      - MultiAssetDDPG   (from Nov. 1)   : TradFi hedging arm
      - DeFiVariablePolicy               : DeFi LP management arm
      - MetaCritic       (Executive)     : joint Q-value estimator

    Parameters
    ----------
    device      : 'cuda' or 'cpu'
    lr_meta     : learning rate for meta-critic and regime detector
    tau_meta    : soft update coefficient for target meta-critic
    gamma       : discount factor
    batch_size  : mini-batch for meta-critic training
    delta_neutral_penalty : regularisation strength for ΣΔ≈0 constraint
    """

    def __init__(
        self,
        device:                  str   = "cpu",
        lr_meta:                 float = 3e-4,
        tau_meta:                float = 0.005,
        gamma:                   float = 0.99,
        batch_size:              int   = 256,
        delta_neutral_penalty:   float = 0.1,
    ):
        self.device               = torch.device(device)
        self.tau                  = tau_meta
        self.gamma                = gamma
        self.batch_size           = batch_size
        self.delta_neutral_penalty = delta_neutral_penalty

        # Build component models
        self.regime_detector  = RegimeDetector().to(self.device)
        self.tradfi_agent     = MultiAssetDDPG(obs_dim=49, action_dim=3, device=device)
        self.defi_policy      = DeFiVariablePolicy().to(self.device)

        self.meta_critic        = MetaCritic().to(self.device)
        self.meta_critic_target = copy.deepcopy(self.meta_critic).to(self.device)
        for p in self.meta_critic_target.parameters():
            p.requires_grad = False

        # Only meta-critic and regime detector are updated by meta-gradient
        meta_params = (
            list(self.meta_critic.parameters())
            + list(self.regime_detector.parameters())
        )
        self.meta_optimizer = optim.Adam(meta_params, lr=lr_meta)

        # DeFi policy has its own optimizer
        self.defi_optimizer = optim.Adam(self.defi_policy.parameters(), lr=lr_meta)

        self.replay      = MetaReplayBuffer()
        self.pnl_history = []
        self.total_steps = 0

    # ── Action selection ──────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self,
        global_obs:  np.ndarray,      # (49,)
        regime_seq:  np.ndarray,      # (20, 4)
        tradfi_obs:  np.ndarray,      # (49,)
        defi_obs:    np.ndarray,      # (18,)
        explore:     bool = True,
    ) -> Dict[str, Any]:
        """
        Run the full meta-policy forward pass.

        Returns
        -------
        dict with:
          tradfi_action  : (3,) [δ_equity, δ_FX, δ_rate]
          defi_action    : (3,) [tick_lower_delta, tick_upper_delta, size_frac]
          regime_probs   : (3,) [p_TradFi, p_DeFi, p_Neutral]
          final_position : (6,) blended aggregate position
          dominant_regime: str
        """
        # 1. Regime detection
        regime_seq_t = torch.FloatTensor(regime_seq).unsqueeze(0).to(self.device)
        regime_probs, _ = self.regime_detector(regime_seq_t)
        regime_probs_np = regime_probs.cpu().numpy().flatten()

        # 2. TradFi arm
        tradfi_action = self.tradfi_agent.select_action(tradfi_obs, explore=explore)

        # 3. DeFi arm
        defi_obs_t = torch.FloatTensor(defi_obs).unsqueeze(0).to(self.device)
        defi_action_t, _, _ = self.defi_policy(defi_obs_t)
        defi_action = defi_action_t.cpu().numpy().flatten()

        # 4. Blended portfolio aggregation (soft regime weighting)
        p_tradfi = regime_probs_np[0]
        p_defi   = regime_probs_np[1]

        final_tradfi = p_tradfi * tradfi_action       # weighted TradFi deltas
        final_defi   = p_defi   * defi_action         # weighted DeFi sizing

        # Delta-neutral normalisation: ensure Σholder_deltas ≈ 0
        delta_sum = np.sum(final_tradfi)
        if abs(delta_sum) > 0.05:
            final_tradfi = final_tradfi - delta_sum / len(final_tradfi)

        labels = ["TradFi", "DeFi", "Neutral"]
        return {
            "tradfi_action":   tradfi_action.astype(np.float32),
            "defi_action":     defi_action.astype(np.float32),
            "regime_probs":    regime_probs_np.astype(np.float32),
            "final_tradfi":    final_tradfi.astype(np.float32),
            "final_defi":      final_defi.astype(np.float32),
            "final_position":  np.concatenate([final_tradfi, final_defi]),
            "dominant_regime": labels[int(regime_probs_np.argmax())],
        }

    # ── Meta-Critic Update ────────────────────────────────────────────────────

    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Single gradient step on Meta-Critic and Regime Detector.

        Updates:
          1. Meta-Critic via TD target from target Meta-Critic
          2. DeFi policy via meta-gradient
          3. Regime detector via backprop through meta-critic loss

        Returns None if buffer not ready.
        """
        if not self.replay.ready(self.batch_size):
            return None

        batch = self.replay.sample(self.batch_size, self.device)

        g_obs       = batch["global_obs"]
        r_seqs      = batch["regime_seqs"]
        ta_actions  = batch["tradfi_actions"]
        da_actions  = batch["defi_actions"]
        rewards     = batch["rewards"]
        ng_obs      = batch["next_global_obs"]
        nr_seqs     = batch["next_regime_seqs"]
        nt_obs      = batch["next_tradfi_obs"]
        nd_obs      = batch["next_defi_obs"]
        dones       = batch["dones"]

        # ── Compute TD target using target networks ────────────────────────
        with torch.no_grad():
            next_regime_probs, _ = self.regime_detector(nr_seqs)

            # Next TradFi action (deterministic, from actor target)
            nt_obs_cpu = nt_obs.cpu().numpy()
            next_ta = np.stack([
                self.tradfi_agent.select_action(nt_obs_cpu[i], explore=False)
                for i in range(len(nt_obs_cpu))
            ])
            next_ta_t = torch.FloatTensor(next_ta).to(self.device)

            # Next DeFi action (deterministic)
            next_da_t, _, _ = self.defi_policy(nd_obs)

            q_next = self.meta_critic_target(ng_obs, next_regime_probs, next_ta_t, next_da_t)
            q_target = rewards + (1 - dones) * self.gamma * q_next

        # ── Current regime probs + Q_meta ─────────────────────────────────
        regime_probs_curr, _ = self.regime_detector(r_seqs)
        q_meta     = self.meta_critic(g_obs, regime_probs_curr, ta_actions, da_actions)
        meta_loss  = nn.MSELoss()(q_meta, q_target)

        # ── Delta-neutral penalty on TradFi actions ─────────────────────
        delta_sum         = ta_actions.sum(dim=-1, keepdim=True).abs().mean()
        delta_neutral_reg = self.delta_neutral_penalty * delta_sum
        total_loss        = meta_loss + delta_neutral_reg

        self.meta_optimizer.zero_grad()
        self.defi_optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.meta_critic.parameters(),      0.5)
        nn.utils.clip_grad_norm_(self.regime_detector.parameters(),  0.5)
        nn.utils.clip_grad_norm_(self.defi_policy.parameters(),       0.5)
        self.meta_optimizer.step()
        self.defi_optimizer.step()

        # Soft update target meta-critic
        self._soft_update(self.meta_critic, self.meta_critic_target)

        self.total_steps += 1
        return {
            "meta_critic_loss": meta_loss.item(),
            "delta_neutral_reg": delta_neutral_reg.item(),
        }

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _soft_update(self, src: nn.Module, tgt: nn.Module):
        for s, t in zip(src.parameters(), tgt.parameters()):
            t.data.copy_(self.tau * s.data + (1 - self.tau) * t.data)

    def store_transition(
        self,
        global_obs, regime_seq, tradfi_obs, defi_obs,
        tradfi_action, defi_action, reward_meta,
        next_global_obs, next_regime_seq, next_tradfi_obs, next_defi_obs,
        done: bool,
    ):
        self.replay.add(
            global_obs, regime_seq, tradfi_obs, defi_obs,
            tradfi_action, defi_action, reward_meta,
            next_global_obs, next_regime_seq, next_tradfi_obs, next_defi_obs,
            done,
        )

    def save(self, path: str) -> None:
        os.makedirs(path, exist_ok=True)
        torch.save(self.regime_detector.state_dict(), f"{path}/regime_detector.pt")
        torch.save(self.defi_policy.state_dict(),     f"{path}/defi_policy.pt")
        torch.save(self.meta_critic.state_dict(),     f"{path}/meta_critic.pt")
        self.tradfi_agent.save(f"{path}/tradfi_ddpg")
        log.info(f"Meta-policy saved → {path}")

    def load(self, path: str) -> None:
        self.regime_detector.load_state_dict(
            torch.load(f"{path}/regime_detector.pt", map_location=self.device))
        self.defi_policy.load_state_dict(
            torch.load(f"{path}/defi_policy.pt", map_location=self.device))
        self.meta_critic.load_state_dict(
            torch.load(f"{path}/meta_critic.pt", map_location=self.device))
        self.meta_critic_target = copy.deepcopy(self.meta_critic)
        self.tradfi_agent.load(f"{path}/tradfi_ddpg")
        log.info(f"Meta-policy loaded ← {path}")

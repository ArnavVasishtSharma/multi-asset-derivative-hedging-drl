"""
models/novelty2_bcrppo/rppo_policy.py
---------------------------------------
Regularised PPO (RPPO) Policy — IV-Surface Aware Hedging Agent (Novelty 2).

Combines:
  - IVSurfaceTransformer (iv_transformer.py) as feature extractor for IV surface sequences
  - Gaussian Actor (policy) with no-trade zone masking (Paper 2 insight)
  - Value network (critic baseline)
  - PPO update with entropy regularisation + KL penalty (RPPO)

Architecture:
  Input  → [obs_raw (49-dim), iv_embedding (128-dim)] = 177-dim total
  Policy → FC(256) → ReLU → FC(256) → ReLU → (μ, log_σ)  → Gaussian action
  Value  → FC(256) → ReLU → FC(256) → ReLU → scalar V(s)

No-Trade Zone (action masking):
  If |action| < ε_min → clip to 0 (no rebalance)
  ε_min is a soft learnable threshold penalised in the PPO objective.

RPPO Objective (per step):
  L = -E[min(r·Adv, clip(r, 1±ε)·Adv)]
      - c_ent × H(π)      (entropy bonus)
      + c_kl  × KL(π, π_old)   (KL penalty from reference policy π_BC)
      + c_vf  × (V - V_target)² (value function loss)

Training loop: collect rollouts → compute GAE advantages → mini-batch PPO update.
"""

import copy
import logging
from typing import Optional, List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from models.novelty2_bcrppo.iv_transformer import IVSurfaceTransformer

log = logging.getLogger(__name__)


# ── Gaussian Policy Network ──────────────────────────────────────────────────

class GaussianPolicyNet(nn.Module):
    """
    Gaussian policy MLP.

    Input : combined feature vector [obs(?), iv_emb(128)] → (B, input_dim)
    Output: mean and log_std of action distribution
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX =  2.0

    def __init__(self, input_dim: int = 177, action_dim: int = 1, hidden_dim: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head    = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (mean, log_std) for each action dimension."""
        h       = self.trunk(x)
        mean    = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def get_action(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample action and compute log-probability.

        Returns
        -------
        action   : (B, action_dim) — sampled action, tanh-squashed to [-1,1]
        log_prob : (B, 1)
        mean     : (B, action_dim) — deterministic action
        """
        mean, log_std = self(x)
        std           = log_std.exp()
        dist          = Normal(mean, std)
        raw_action    = dist.rsample()

        # Tanh squash to bound action in [-1, 1]
        action   = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, torch.tanh(mean)


# ── Value Network ────────────────────────────────────────────────────────────

class ValueNet(nn.Module):
    """State value function V(s)."""

    def __init__(self, input_dim: int = 177, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Rollout Buffer ──────────────────────────────────────────────────────────

class RolloutBuffer:
    """Stores on-policy rollout data for PPO updates."""

    def __init__(self):
        self.clear()

    def clear(self):
        self.obs:       List[np.ndarray] = []
        self.iv_seqs:   List[np.ndarray] = []
        self.actions:   List[np.ndarray] = []
        self.log_probs: List[float]      = []
        self.rewards:   List[float]      = []
        self.values:    List[float]      = []
        self.dones:     List[bool]       = []

    def add(
        self,
        obs:      np.ndarray,
        iv_seq:   np.ndarray,
        action:   np.ndarray,
        log_prob: float,
        reward:   float,
        value:    float,
        done:     bool,
    ):
        self.obs.append(obs)
        self.iv_seqs.append(iv_seq)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def compute_gae(
        self,
        last_value: float,
        gamma:  float = 0.99,
        lam:    float = 0.95,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Generalised Advantage Estimates (GAE-λ).

        Returns
        -------
        advantages : (T,) array
        returns    : (T,)  = advantages + values
        """
        T         = len(self.rewards)
        advantages = np.zeros(T, dtype=np.float32)
        last_gae   = 0.0
        values_ext = self.values + [last_value]

        for t in reversed(range(T)):
            delta        = self.rewards[t] + gamma * values_ext[t + 1] * (1 - self.dones[t]) - values_ext[t]
            last_gae     = delta + gamma * lam * (1 - self.dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(self.values, dtype=np.float32)
        return advantages, returns

    def to_tensors(self, device: torch.device) -> dict:
        return {
            "obs":       torch.FloatTensor(np.stack(self.obs)).to(device),
            "iv_seqs":   torch.FloatTensor(np.stack(self.iv_seqs)).to(device),
            "actions":   torch.FloatTensor(np.stack(self.actions)).to(device),
            "log_probs": torch.FloatTensor(self.log_probs).to(device),
        }


# ── RPPO Agent ───────────────────────────────────────────────────────────────

class IVSurfaceBCRPPO:
    """
    IV-Surface Aware BC-RPPO Agent (Novelty 2).

    Wraps:
      - IVSurfaceTransformer : IV feature extractor
      - GaussianPolicyNet    : actor
      - ValueNet             : critic
      - PPO update loop with KL penalty (RPPO)
      - No-trade zone action masking

    Parameters
    ----------
    obs_dim       : raw observation dim (49)
    iv_dim        : flattened IV surface dim (25)
    iv_seq_len    : number of historical IV steps (30)
    action_dim    : action dimensionality (1 = single delta, 3 for multi-asset)
    hidden_dim    : MLP hidden size
    lr            : learning rate
    gamma, lam    : GAE discount + lambda
    clip_eps      : PPO clip parameter ε
    c_ent         : entropy bonus coefficient
    c_kl          : KL penalty coefficient (RPPO)
    c_vf          : value loss coefficient
    n_epochs      : number of PPO epochs per rollout
    batch_size    : mini-batch size for PPO
    no_trade_eps  : minimum delta change to consider trading
    device        : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        obs_dim:      int   = 49,
        iv_dim:       int   = 25,
        iv_seq_len:   int   = 30,
        action_dim:   int   = 1,
        hidden_dim:   int   = 256,
        lr:           float = 3e-4,
        gamma:        float = 0.99,
        lam:          float = 0.95,
        clip_eps:     float = 0.2,
        c_ent:        float = 0.01,
        c_kl:         float = 0.01,
        c_vf:         float = 0.5,
        n_epochs:     int   = 10,
        batch_size:   int   = 256,
        no_trade_eps: float = 0.02,
        device:       str   = "cpu",
    ):
        self.device       = torch.device(device)
        self.gamma        = gamma
        self.lam          = lam
        self.clip_eps     = clip_eps
        self.c_ent        = c_ent
        self.c_kl         = c_kl
        self.c_vf         = c_vf
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.no_trade_eps = no_trade_eps
        self.action_dim   = action_dim

        # Networks
        embed_dim  = 128
        policy_in  = obs_dim + embed_dim
        self.transformer = IVSurfaceTransformer(
            iv_dim=iv_dim, seq_len=iv_seq_len, embed_dim=embed_dim
        ).to(self.device)
        self.policy  = GaussianPolicyNet(policy_in, action_dim, hidden_dim).to(self.device)
        self.value   = ValueNet(policy_in, hidden_dim).to(self.device)

        # Frozen reference policy for KL regularisation (loaded after BC pretraining)
        self.ref_policy: Optional[GaussianPolicyNet] = None

        all_params = (
            list(self.transformer.parameters())
            + list(self.policy.parameters())
            + list(self.value.parameters())
        )
        self.optimizer = optim.Adam(all_params, lr=lr, eps=1e-5)
        self.rollout   = RolloutBuffer()
        self.total_updates = 0

    # ── Feature extraction ────────────────────────────────────────────────────

    @torch.no_grad()
    def _get_features(self, obs: np.ndarray, iv_seq: np.ndarray) -> torch.Tensor:
        """
        Concatenate raw observation with IV embedding.

        Returns
        -------
        features : (1, obs_dim + embed_dim) FloatTensor on device
        """
        obs_t   = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        iv_t    = torch.FloatTensor(iv_seq).unsqueeze(0).to(self.device)
        iv_emb  = self.transformer(iv_t)
        return torch.cat([obs_t, iv_emb], dim=-1)

    # ── Action selection ──────────────────────────────────────────────────────

    @torch.no_grad()
    def select_action(
        self,
        obs:    np.ndarray,
        iv_seq: np.ndarray,
        explore: bool = True,
    ) -> Tuple[np.ndarray, float, float]:
        """
        Select an action with optional stochastic exploration.

        Applies no-trade zone: if predicted delta change < ε_min, force 0.

        Returns
        -------
        action   : (action_dim,) numpy array
        log_prob : scalar
        value    : scalar V(s)
        """
        feats = self._get_features(obs, iv_seq)

        if explore:
            action_t, lp_t, _ = self.policy.get_action(feats)
        else:
            _, _, action_t = self.policy.get_action(feats)
            lp_t = torch.zeros(1, 1, device=self.device)

        value_t = self.value(feats)

        action   = action_t.cpu().numpy().flatten()
        log_prob = lp_t.cpu().item()
        value    = value_t.cpu().item()

        # No-trade zone mask
        action = np.where(np.abs(action) < self.no_trade_eps, 0.0, action)
        return action.astype(np.float32), log_prob, value

    def store_transition(
        self,
        obs: np.ndarray, iv_seq: np.ndarray, action: np.ndarray,
        log_prob: float, reward: float, value: float, done: bool
    ):
        self.rollout.add(obs, iv_seq, action, log_prob, reward, value, done)

    # ── PPO Update ────────────────────────────────────────────────────────────

    def update(self, last_value: float = 0.0) -> Dict[str, float]:
        """
        Perform PPO update on collected rollout data.

        Parameters
        ----------
        last_value : bootstrap value for end of rollout (0 if terminal)

        Returns
        -------
        stats : dict with policy_loss, value_loss, entropy, kl
        """
        advantages, returns = self.rollout.compute_gae(last_value, self.gamma, self.lam)
        data = self.rollout.to_tensors(self.device)

        obs_t      = data["obs"]
        iv_seq_t   = data["iv_seqs"]
        actions_t  = data["actions"]
        old_lp_t   = data["log_probs"].unsqueeze(-1)
        adv_t      = torch.FloatTensor(advantages).unsqueeze(-1).to(self.device)
        returns_t  = torch.FloatTensor(returns).unsqueeze(-1).to(self.device)

        # Normalise advantages
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

        N = len(obs_t)
        stats_accum = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "kl": 0.0}
        n_updates = 0

        for _ in range(self.n_epochs):
            perm = torch.randperm(N)
            for start in range(0, N - self.batch_size, self.batch_size):
                idx = perm[start:start + self.batch_size]

                b_obs     = obs_t[idx]
                b_iv      = iv_seq_t[idx]
                b_actions = actions_t[idx]
                b_old_lp  = old_lp_t[idx]
                b_adv     = adv_t[idx]
                b_ret     = returns_t[idx]

                # Compute current features
                iv_emb  = self.transformer(b_iv)
                feats   = torch.cat([b_obs, iv_emb], dim=-1)

                # Policy evaluation
                mean, log_std = self.policy(feats)
                std   = log_std.exp()
                dist  = Normal(mean, std)
                new_lp = dist.log_prob(b_actions).sum(-1, keepdim=True)
                entropy = dist.entropy().sum(-1).mean()

                # PPO clipped objective
                ratio      = (new_lp - b_old_lp).exp()
                surr1      = ratio * b_adv
                surr2      = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                v_pred    = self.value(feats)
                value_loss = nn.MSELoss()(v_pred, b_ret)

                # KL penalty from reference policy (RPPO)
                kl_loss = torch.tensor(0.0, device=self.device)
                if self.ref_policy is not None:
                    with torch.no_grad():
                        ref_mean, ref_log_std = self.ref_policy(feats)
                    ref_std  = ref_log_std.exp()
                    kl_loss  = torch.distributions.kl_divergence(
                        Normal(mean, std), Normal(ref_mean, ref_std)
                    ).sum(-1).mean()

                total_loss = (
                    policy_loss
                    + self.c_vf * value_loss
                    - self.c_ent * entropy
                    + self.c_kl * kl_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.transformer.parameters())
                    + list(self.policy.parameters())
                    + list(self.value.parameters()),
                    max_norm=0.5,
                )
                self.optimizer.step()

                stats_accum["policy_loss"] += policy_loss.item()
                stats_accum["value_loss"]  += value_loss.item()
                stats_accum["entropy"]     += entropy.item()
                stats_accum["kl"]          += kl_loss.item() if self.ref_policy else 0.0
                n_updates += 1

        self.rollout.clear()
        self.total_updates += 1

        denom = max(n_updates, 1)
        return {k: v / denom for k, v in stats_accum.items()}

    # ── BC Warm-start integration ─────────────────────────────────────────────

    def load_bc_pretrained(
        self,
        transformer_state: dict,
        policy_state: dict,
    ) -> None:
        """
        Load BC-pretrained weights into transformer and policy.
        Sets the loaded policy as the frozen KL reference (π_BC).
        """
        self.transformer.load_state_dict(transformer_state)
        self.policy.load_state_dict(policy_state, strict=False)

        # Freeze reference policy
        self.ref_policy = copy.deepcopy(self.policy).to(self.device)
        for p in self.ref_policy.parameters():
            p.requires_grad = False
        log.info("BC-pretrained weights loaded. Reference policy frozen for KL regularisation.")

    # ── Save / Load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        import os
        os.makedirs(path, exist_ok=True)
        torch.save(self.transformer.state_dict(), f"{path}/transformer.pt")
        torch.save(self.policy.state_dict(),      f"{path}/policy.pt")
        torch.save(self.value.state_dict(),       f"{path}/value.pt")
        log.info(f"BC-RPPO model saved → {path}")

    def load(self, path: str) -> None:
        self.transformer.load_state_dict(torch.load(f"{path}/transformer.pt", map_location=self.device))
        self.policy.load_state_dict(torch.load(f"{path}/policy.pt", map_location=self.device))
        self.value.load_state_dict(torch.load(f"{path}/value.pt", map_location=self.device))
        log.info(f"BC-RPPO model loaded ← {path}")

"""
models/novelty1_ddpg/ddpg_agent.py
-------------------------------------
Full Correlation-Aware Multi-Asset DDPG Agent (Novelty 1).

Combines:
  - MultiAssetActor (actor.py)
  - TwinCritic (critic.py)
  - Ornstein-Uhlenbeck exploration noise (utils/noise.py)
  - ReplayBuffer (utils/replay_buffer.py)
  - Soft target updates (τ = 0.005)
  - Symbol-specific calibration: fine-tune actor per underlying (Paper 1)

Key differences from vanilla DDPG:
  1. 3D action space [δ_equity, δ_FX, δ_rate]
  2. CorrelationEncoder inside actor
  3. Twin-critic (clipped double-Q, TD3 trick)
  4. CVaR term in reward (from environment)
  5. Leak-free: no future data in replay buffer (Paper 4)
"""

import os
import copy
import logging
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.novelty1_ddpg.actor import MultiAssetActor
from models.novelty1_ddpg.critic import TwinCritic
from utils.replay_buffer import ReplayBuffer
from utils.noise import OrnsteinUhlenbeck
from utils.metrics import compute_sharpe, compute_cvar

log = logging.getLogger(__name__)


class MultiAssetDDPG:
    """
    Correlation-Aware Multi-Asset DDPG Agent.

    Parameters
    ----------
    obs_dim, action_dim: environment dimensions
    hidden_dim         : hidden layer size (default 512)
    lr_actor, lr_critic: learning rates
    gamma              : discount factor
    tau                : soft update coefficient
    buffer_size        : replay buffer capacity
    batch_size         : mini-batch size
    noise_sigma        : OU noise standard deviation
    device             : 'cuda' or 'cpu'
    """

    def __init__(
        self,
        obs_dim:    int   = 49,
        action_dim: int   = 3,
        hidden_dim: int   = 512,
        lr_actor:   float = 1e-4,
        lr_critic:  float = 3e-4,
        gamma:      float = 0.99,
        tau:        float = 0.005,
        buffer_size: int  = 200_000,
        batch_size:  int  = 256,
        noise_sigma: float= 0.1,
        device:     str   = "cpu",
    ):
        self.device     = torch.device(device)
        self.gamma      = gamma
        self.tau        = tau
        self.batch_size = batch_size
        self.action_dim = action_dim

        # Networks
        self.actor   = MultiAssetActor(obs_dim, action_dim, hidden_dim=hidden_dim).to(self.device)
        self.critic  = TwinCritic(obs_dim, action_dim, hidden_dim=hidden_dim).to(self.device)

        # Target networks (Polyak average)
        self.actor_target  = copy.deepcopy(self.actor).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.actor_target.parameters():  p.requires_grad = False
        for p in self.critic_target.parameters(): p.requires_grad = False

        # Optimisers
        self.actor_optim  = optim.Adam(self.actor.parameters(),  lr=lr_actor)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Replay buffer & exploration noise
        self.replay = ReplayBuffer(obs_dim, action_dim, max_size=buffer_size)
        self.noise  = OrnsteinUhlenbeck(action_dim, sigma=noise_sigma)

        self.total_steps = 0

    # ── Action selection ──────────────────────────────────────────────────────
    @torch.no_grad()
    def select_action(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        """
        Select action with optional OU noise for exploration.

        Parameters
        ----------
        obs     : (obs_dim,) numpy array
        explore : if True, add OU noise

        Returns
        -------
        action : (3,) numpy array clipped to [-1, 1]
        """
        obs_t  = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        action = self.actor(obs_t).cpu().numpy().flatten()
        if explore:
            action = action + self.noise.sample()
        return np.clip(action, -1.0, 1.0)

    # ── Training step ─────────────────────────────────────────────────────────
    def train_step(self) -> Optional[Dict[str, float]]:
        """
        Sample a batch from replay buffer and perform one gradient update.

        Returns dict with losses, or None if buffer not ready.
        """
        if not self.replay.ready(self.batch_size):
            return None

        batch = self.replay.sample(self.batch_size, device=self.device)
        obs, actions, rewards, next_obs, dones = batch

        # ── Critic update ──────────────────────────────────────────────────
        with torch.no_grad():
            next_actions = self.actor_target(next_obs)
            target_q     = self.critic_target.q_min(next_obs, next_actions)
            target_q     = rewards + (1 - dones) * self.gamma * target_q

        q1, q2 = self.critic(obs, actions)
        critic_loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optim.step()

        # ── Actor update ───────────────────────────────────────────────────
        pred_actions = self.actor(obs)
        actor_loss   = -self.critic.q_min(obs, pred_actions).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optim.step()

        # ── Soft target updates ────────────────────────────────────────────
        self._soft_update(self.actor,  self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        self.total_steps += 1
        return {
            "critic_loss": critic_loss.item(),
            "actor_loss":  actor_loss.item(),
        }

    # ── Symbol-specific calibration (Paper 1 insight) ─────────────────────────
    def calibrate_to_symbol(
        self,
        symbol_data: list,            # list of (obs, action) supervised pairs
        epochs: int = 100,
        lr: float = 5e-5,
    ) -> float:
        """
        Fine-tune only the per-asset heads of the actor on symbol-specific data.
        This implements Paper 1's symbol-specific calibration for American options.

        Parameters
        ----------
        symbol_data : list of (obs_array, target_action_array) tuples
        epochs      : number of supervised epochs
        lr          : learning rate for fine-tuning

        Returns
        -------
        final_mse : float
        """
        # Freeze all trunk layers; only train output heads
        for name, param in self.actor.named_parameters():
            param.requires_grad = "head_" in name

        calib_optim = optim.Adam(
            filter(lambda p: p.requires_grad, self.actor.parameters()), lr=lr
        )
        criterion = nn.MSELoss()

        obs_arr = torch.FloatTensor(np.stack([x[0] for x in symbol_data])).to(self.device)
        act_arr = torch.FloatTensor(np.stack([x[1] for x in symbol_data])).to(self.device)

        for _ in range(epochs):
            pred = self.actor(obs_arr)
            loss = criterion(pred, act_arr)
            calib_optim.zero_grad()
            loss.backward()
            calib_optim.step()

        # Unfreeze all parameters
        for param in self.actor.parameters():
            param.requires_grad = True
        log.info(f"Symbol calibration complete | Final MSE: {loss.item():.6f}")
        return loss.item()

    # ── Utilities ─────────────────────────────────────────────────────────────
    def _soft_update(self, source: nn.Module, target: nn.Module):
        for s_param, t_param in zip(source.parameters(), target.parameters()):
            t_param.data.copy_(self.tau * s_param.data + (1 - self.tau) * t_param.data)

    def store_transition(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ):
        self.replay.add(obs, action, reward, next_obs, done)

    def save(self, path: str):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(),  os.path.join(path, "actor.pt"))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pt"))
        log.info(f"Model saved → {path}")

    def load(self, path: str):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pt"),
                                               map_location=self.device))
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pt"),
                                                map_location=self.device))
        self.actor_target  = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        log.info(f"Model loaded ← {path}")

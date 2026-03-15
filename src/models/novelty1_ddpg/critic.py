"""
models/novelty1_ddpg/critic.py
--------------------------------
Multi-Asset DDPG Twin-Critic (Clipped Double Q, from TD3 improvement).

Processes:
  - State (49-dim)
  - Action (3-dim delta vector)

Returns two Q-values; training uses min(Q1, Q2) to prevent overestimation.
Cost-aware critic: transaction cost is embedded directly in the state to
allow the critic to model rebalancing friction (Paper 4's key insight).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CriticNetwork(nn.Module):
    """Single Q-network for DDPG/TD3 critic."""

    def __init__(self, obs_dim: int = 49, action_dim: int = 3, hidden_dim: int = 512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.zeros_(m.bias)
        # Last layer small init for stable Q-values
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs    : (B, 49)
        action : (B, 3)

        Returns
        -------
        q_value : (B, 1)
        """
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)


class TwinCritic(nn.Module):
    """
    Clipped Double Q-Learning (TD3-style).
    Uses two independent critics to reduce Q overestimation.
    Cost-awareness: transaction cost signal appended to obs before Q estimation.
    """

    def __init__(self, obs_dim: int = 49, action_dim: int = 3, hidden_dim: int = 512):
        super().__init__()
        self.q1 = CriticNetwork(obs_dim, action_dim, hidden_dim)
        self.q2 = CriticNetwork(obs_dim, action_dim, hidden_dim)

    def forward(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (Q1, Q2) for training."""
        return self.q1(obs, action), self.q2(obs, action)

    def q_min(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Clipped Q-value: min(Q1, Q2) for target computation."""
        q1, q2 = self.forward(obs, action)
        return torch.min(q1, q2)

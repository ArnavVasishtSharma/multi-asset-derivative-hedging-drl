"""
utils/replay_buffer.py
-----------------------
Circular replay buffer for off-policy RL (DDPG / TD3 style).

Stores (obs, action, reward, next_obs, done) tuples and supports
random mini-batch sampling onto a specified torch device.
"""

from typing import Tuple
import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-capacity circular replay buffer.

    Parameters
    ----------
    obs_dim    : dimension of observation vector
    action_dim : dimension of action vector
    max_size   : maximum number of transitions to store
    """

    def __init__(self, obs_dim: int, action_dim: int, max_size: int = 200_000):
        self.max_size   = max_size
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.ptr        = 0
        self.size       = 0

        self.obs      = np.zeros((max_size, obs_dim),    dtype=np.float32)
        self.actions  = np.zeros((max_size, action_dim), dtype=np.float32)
        self.rewards  = np.zeros((max_size, 1),          dtype=np.float32)
        self.next_obs = np.zeros((max_size, obs_dim),    dtype=np.float32)
        self.dones    = np.zeros((max_size, 1),          dtype=np.float32)

    # ------------------------------------------------------------------
    def add(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> None:
        """Store a single transition."""
        self.obs[self.ptr]      = obs
        self.actions[self.ptr]  = action
        self.rewards[self.ptr]  = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr]    = float(done)
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def add_batch(
        self,
        obs:      np.ndarray,
        actions:  np.ndarray,
        rewards:  np.ndarray,
        next_obs: np.ndarray,
        dones:    np.ndarray,
    ) -> None:
        """Batch insert (useful for vectorised envs)."""
        n = len(obs)
        idxs = np.arange(self.ptr, self.ptr + n) % self.max_size
        self.obs[idxs]      = obs
        self.actions[idxs]  = actions
        self.rewards[idxs]  = rewards.reshape(-1, 1)
        self.next_obs[idxs] = next_obs
        self.dones[idxs]    = dones.reshape(-1, 1)
        self.ptr  = (self.ptr + n) % self.max_size
        self.size = min(self.size + n, self.max_size)

    # ------------------------------------------------------------------
    def sample(
        self,
        batch_size: int,
        device: torch.device,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Sample a random mini-batch.

        Returns
        -------
        (obs, actions, rewards, next_obs, dones) — all FloatTensors on `device`
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[idxs]).to(device),
            torch.FloatTensor(self.actions[idxs]).to(device),
            torch.FloatTensor(self.rewards[idxs]).to(device),
            torch.FloatTensor(self.next_obs[idxs]).to(device),
            torch.FloatTensor(self.dones[idxs]).to(device),
        )

    def ready(self, batch_size: int) -> bool:
        """Return True only when buffer has enough samples to train."""
        return self.size >= batch_size

    def __len__(self) -> int:
        return self.size

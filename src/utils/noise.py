"""
utils/noise.py
--------------
Exploration noise processes for off-policy RL.

Provides:
  - OrnsteinUhlenbeck : mean-reverting temporally correlated noise (DDPG standard)
  - GaussianNoise     : i.i.d. Gaussian noise (simpler alternative, TD3)
"""

import numpy as np


class OrnsteinUhlenbeck:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration noise.

        dX_t = θ(μ - X_t)dt + σ dW_t

    Parameters
    ----------
    dim   : dimensionality of the action space
    mu    : mean reversion level (default 0)
    theta : mean reversion rate
    sigma : noise standard deviation
    dt    : time step
    """

    def __init__(
        self,
        dim:   int,
        mu:    float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.1,
        dt:    float = 1e-2,
    ):
        self.dim   = dim
        self.mu    = mu * np.ones(dim, dtype=np.float32)
        self.theta = theta
        self.sigma = sigma
        self.dt    = dt
        self._state = np.copy(self.mu)

    def reset(self) -> None:
        """Reset state to the mean."""
        self._state = np.copy(self.mu)

    def sample(self) -> np.ndarray:
        """Draw next noise sample."""
        dx = (
            self.theta * (self.mu - self._state) * self.dt
            + self.sigma * np.sqrt(self.dt) * np.random.randn(self.dim)
        )
        self._state += dx
        return self._state.astype(np.float32)

    def __call__(self) -> np.ndarray:
        return self.sample()


class GaussianNoise:
    """
    i.i.d. Gaussian exploration noise.

    Parameters
    ----------
    dim   : dimensionality of the action space
    sigma : standard deviation of the noise
    """

    def __init__(self, dim: int, sigma: float = 0.1):
        self.dim   = dim
        self.sigma = sigma

    def reset(self) -> None:
        """No state to reset."""
        pass

    def sample(self) -> np.ndarray:
        return (self.sigma * np.random.randn(self.dim)).astype(np.float32)

    def __call__(self) -> np.ndarray:
        return self.sample()

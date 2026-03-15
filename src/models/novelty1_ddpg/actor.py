"""
models/novelty1_ddpg/actor.py
-------------------------------
Multi-Asset DDPG Actor Network (Novelty 1).

Outputs a 3-dimensional delta vector:
  [delta_equity, delta_fx, delta_rate] ∈ [-1, 1]^3

Architecture:
  Shared trunk → LayerNorm → 2× residual blocks → 3-head Tanh output
  Correlation matrix is processed by a dedicated sub-network and fused.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class ResidualBlock(nn.Module):
    """Simple 2-layer residual block with LayerNorm."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1  = nn.Linear(hidden_dim, hidden_dim)
        self.fc2  = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.norm(x + residual)


class CorrelationEncoder(nn.Module):
    """
    Encodes the flattened 3×3 = 9-dim correlation matrix.
    Uses a small MLP → 32-dim embedding.
    """
    def __init__(self, corr_input_dim: int = 9, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(corr_input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.Tanh(),
        )

    def forward(self, corr_flat: torch.Tensor) -> torch.Tensor:
        return self.net(corr_flat)


class MultiAssetActor(nn.Module):
    """
    DDPG Actor for multi-asset hedging.

    Parameters
    ----------
    obs_dim      : Total observation dimension (49)
    action_dim   : Output dimension (3)
    corr_start   : Index in obs where the 9-dim correlation features begin
    corr_len     : Length of correlation features (9)
    hidden_dim   : Hidden layer size (512)
    """

    def __init__(
        self,
        obs_dim:    int = 49,
        action_dim: int = 3,
        corr_start: int = 32,   # after: 7 prices + 25 IV + 0-indexed start of corr
        corr_len:   int = 9,
        hidden_dim: int = 512,
    ):
        super().__init__()
        self.corr_start = corr_start
        self.corr_len   = corr_len
        market_dim      = obs_dim - corr_len   # remaining features

        # Correlation sub-network
        self.corr_encoder = CorrelationEncoder(corr_len, embed_dim=32)

        # Market feature trunk
        self.market_trunk = nn.Sequential(
            nn.Linear(market_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Fusion + residual blocks
        fusion_dim = hidden_dim + 32
        self.fusion = nn.Linear(fusion_dim, hidden_dim)
        self.res1   = ResidualBlock(hidden_dim)
        self.res2   = ResidualBlock(hidden_dim)
        self.res3   = ResidualBlock(hidden_dim)

        # Per-asset heads (symbol-specific calibration)
        self.head_equity = nn.Linear(hidden_dim // 3, 1)
        self.head_fx     = nn.Linear(hidden_dim // 3, 1)
        self.head_rate   = nn.Linear(hidden_dim // 3, 1)
        self.split_proj  = nn.Linear(hidden_dim, hidden_dim)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        obs : (batch, obs_dim)

        Returns
        -------
        actions : (batch, 3)  — delta_equity, delta_fx, delta_rate ∈ [-1, 1]
        """
        # Split correlation features
        corr_feat   = obs[:, self.corr_start:self.corr_start + self.corr_len]
        market_feat = torch.cat([
            obs[:, :self.corr_start],
            obs[:, self.corr_start + self.corr_len:]
        ], dim=-1)

        # Encode
        corr_emb  = self.corr_encoder(corr_feat)          # (B, 32)
        market_emb = self.market_trunk(market_feat)        # (B, 512)

        # Fuse
        x = F.relu(self.fusion(torch.cat([market_emb, corr_emb], dim=-1)))  # (B, 512)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)

        # Split for symbol-specific heads (Paper 1 insight)
        x_split = self.split_proj(x)     # (B, 512)
        chunk    = x_split.shape[-1] // 3
        eq_feat  = x_split[:, :chunk]
        fx_feat  = x_split[:, chunk:2*chunk]
        r_feat   = x_split[:, 2*chunk:3*chunk]

        delta_equity = torch.tanh(self.head_equity(eq_feat))   # (B,1)
        delta_fx     = torch.tanh(self.head_fx(fx_feat))       # (B,1)
        delta_rate   = torch.tanh(self.head_rate(r_feat))       # (B,1)

        return torch.cat([delta_equity, delta_fx, delta_rate], dim=-1)   # (B,3)

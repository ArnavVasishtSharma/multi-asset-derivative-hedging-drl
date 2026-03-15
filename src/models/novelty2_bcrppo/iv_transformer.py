"""
models/novelty2_bcrppo/iv_transformer.py
-----------------------------------------
IV Surface Transformer Forecaster (Novelty 2).

Ingests a sequence of T historical IV surfaces (each 5×5=25 dims) and
produces a dense 128-dim embedding capturing volatility regime features.

Architecture:
  PatchEmbed  → Positional Encoding → TransformerEncoder (4 heads, 4 layers)
  → Pool (mean) → FC(128)

This embedding is fed as additional state to the BC-RPPO policy.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 50, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)   # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class IVSurfaceTransformer(nn.Module):
    """
    Transformer-based IV surface forecaster.

    Parameters
    ----------
    iv_dim      : dimension of one flattened IV surface (25)
    seq_len     : number of historical timesteps (30 days)
    d_model     : Transformer model dimension (128)
    nhead       : number of attention heads (4)
    num_layers  : number of Transformer encoder layers (4)
    embed_dim   : final embedding dimension (128)
    dropout     : dropout rate
    """

    def __init__(
        self,
        iv_dim:     int   = 25,
        seq_len:    int   = 30,
        d_model:    int   = 128,
        nhead:      int   = 4,
        num_layers: int   = 4,
        embed_dim:  int   = 128,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.seq_len  = seq_len
        self.d_model  = d_model
        self.embed_dim = embed_dim

        # Project IV surface → d_model
        self.patch_embed = nn.Sequential(
            nn.Linear(iv_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
        )

        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Projection head → embedding
        self.head = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(
        self,
        iv_seq: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        iv_seq  : (B, T, iv_dim) — sequence of flattened IV surfaces
        src_key_padding_mask : optional padding mask (B, T)

        Returns
        -------
        embedding : (B, embed_dim=128)
        """
        # Embed each timestep
        x = self.patch_embed(iv_seq)          # (B, T, d_model)
        x = self.pos_encoding(x)              # (B, T, d_model)

        # Transformer encoding
        x = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)  # (B, T, d_model)

        # Pool over time dimension → (B, d_model)
        x = x.mean(dim=1)

        return self.head(x)   # (B, embed_dim)


class IVSurfacePredictor(nn.Module):
    """
    Auxiliary next-surface prediction head for self-supervised pretraining.
    Predicts next step's IV surface from the TransformerEncoder output.
    Pretraining objective: dense prediction to bootstrap the transformer
    before RL training (addresses the sparse-reward cold start from Paper 5).
    """

    def __init__(self, d_model: int = 128, iv_dim: int = 25):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, iv_dim),
            nn.Softplus(),   # IV surfaces are positive
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """embedding: (B, d_model) → predicted_iv: (B, iv_dim)"""
        return self.predictor(embedding)

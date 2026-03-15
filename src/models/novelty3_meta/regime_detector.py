"""
models/novelty3_meta/regime_detector.py
-----------------------------------------
LSTM-based Regime Detector for the Hybrid TradFi-DeFi Meta-Policy (Novelty 3).

Purpose:
  Classifies the current market regime into one of three classes:
    0 = TradFi  (exchange-traded derivatives dominate)
    1 = DeFi    (on-chain liquidity pool environment dominates)
    2 = Neutral (balanced / transition period)

Input features (rolling 20-period window):
  [VIX, funding_rate, on_chain_vol, basis_spread]  → 4 features × 20 timesteps

Architecture:
  LSTM (hidden=64, layers=2) → last hidden state → FC(32) → ReLU → FC(3) (logits)

Output: probability distribution over regimes [p_TradFi, p_DeFi, p_Neutral]
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class RegimeDetector(nn.Module):
    """
    LSTM-based market regime classifier.

    Parameters
    ----------
    input_dim   : number of regime indicator features (default 4)
    seq_len     : rolling window length (default 20)
    hidden_dim  : LSTM hidden dimension (default 64)
    num_layers  : LSTM depth (default 2)
    n_regimes   : number of regimes to classify (default 3)
    dropout     : LSTM dropout (only applied when num_layers > 1)
    """

    def __init__(
        self,
        input_dim:  int = 4,
        seq_len:    int = 20,
        hidden_dim: int = 64,
        num_layers: int = 2,
        n_regimes:  int = 3,
        dropout:    float = 0.1,
    ):
        super().__init__()
        self.input_dim  = input_dim
        self.seq_len    = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.n_regimes  = n_regimes

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, n_regimes),
        )

    def forward(
        self,
        regime_seq: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
        c0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Parameters
        ----------
        regime_seq : (B, T, input_dim) — rolling window of regime indicators
        h0, c0     : optional initial LSTM hidden states

        Returns
        -------
        regime_probs : (B, n_regimes) — softmax probabilities
        (h_n, c_n)   : final hidden states for autoregressive streaming
        """
        if h0 is None or c0 is None:
            out, (h_n, c_n) = self.lstm(regime_seq)
        else:
            out, (h_n, c_n) = self.lstm(regime_seq, (h0, c0))

        # Use the last timestep's hidden state
        last_hidden   = h_n[-1]              # (B, hidden_dim)
        logits        = self.classifier(last_hidden)  # (B, n_regimes)
        regime_probs  = torch.softmax(logits, dim=-1)
        return regime_probs, (h_n, c_n)

    @torch.no_grad()
    def classify(self, regime_seq_np) -> dict:
        """
        Convenience method: numpy array → regime dict.

        Parameters
        ----------
        regime_seq_np : (T, input_dim) numpy array

        Returns
        -------
        dict with keys: tradfi_prob, defi_prob, neutral_prob, regime_label
        """
        import numpy as np
        x = torch.FloatTensor(regime_seq_np).unsqueeze(0)  # (1, T, 4)
        probs, _ = self.forward(x)
        probs_np = probs.cpu().numpy().flatten()
        label    = int(probs_np.argmax())
        return {
            "tradfi_prob":  float(probs_np[0]),
            "defi_prob":    float(probs_np[1]),
            "neutral_prob": float(probs_np[2]),
            "regime_label": label,           # 0=TradFi, 1=DeFi, 2=Neutral
        }

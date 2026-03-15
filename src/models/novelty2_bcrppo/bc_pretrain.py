"""
models/novelty2_bcrppo/bc_pretrain.py
---------------------------------------
Behavior Cloning Pretraining Module (Novelty 2).

Two-stage pretraining before BC-RPPO RL training:

Stage 1 — IV Transformer self-supervised pretraining:
  Predict next IV surface from past 30-day sequence.
  Objective: MSE of next_iv prediction.

Stage 2 — Behavior Cloning on Black-Scholes deltas:
  Jointly trains the Transformer + policy network to mimic BS delta
  across different IV regimes. This injects prior knowledge and addresses
  sparse reward cold start (Paper 5's core challenge).
  Objective: MSE(π(s), δ_BS) + λ_KL × KL(π, π_BC_frozen)
"""

import logging
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from models.novelty2_bcrppo.iv_transformer import IVSurfaceTransformer, IVSurfacePredictor
from data.preprocessor import bs_greeks

log = logging.getLogger(__name__)


# ── Stage 1: Transformer Self-Supervised Pretraining ─────────────────────────
def pretrain_iv_transformer(
    transformer: IVSurfaceTransformer,
    iv_sequences: np.ndarray,     # (N, T+1, 25)  — T past + 1 next
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    device: str = "cpu",
) -> List[float]:
    """
    Stage 1: self-supervised pretraining on next-IV-surface prediction.

    Parameters
    ----------
    iv_sequences : (N, T+1, 25) — sequences of IV surfaces
    Returns list of epoch losses.
    """
    log.info(f"Stage 1 — IV Transformer self-supervised pretraining ({epochs} epochs)")
    predictor = IVSurfacePredictor(d_model=transformer.d_model, iv_dim=25).to(device)
    transformer = transformer.to(device)

    params = list(transformer.parameters()) + list(predictor.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Split input/target
    iv_input  = torch.FloatTensor(iv_sequences[:, :-1, :])  # (N, T, 25)
    iv_target = torch.FloatTensor(iv_sequences[:, -1,  :])  # (N, 25)

    dataset    = TensorDataset(iv_input, iv_target)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    losses = []
    for epoch in range(epochs):
        epoch_losses = []
        for batch_iv, batch_target in dataloader:
            batch_iv     = batch_iv.to(device)
            batch_target = batch_target.to(device)
            embedding = transformer(batch_iv)
            pred_iv   = predictor(embedding)
            loss      = criterion(pred_iv, batch_target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        avg = float(np.mean(epoch_losses))
        losses.append(avg)
        if epoch % 10 == 0:
            log.info(f"  Epoch {epoch:3d}/{epochs} | IV-pred MSE: {avg:.6f}")
    log.info("Stage 1 complete.")
    return losses


# ── Black-Scholes Delta Generator ─────────────────────────────────────────────
def generate_bs_delta_targets(
    states: np.ndarray,      # (N, obs_dim) observation vectors
    spot_idx:   int = 0,     # column index for spot price
    rate_idx:   int = 4,     # column index for rate_1y
    iv_idx:     int = 7,     # column index for first IV value (ATM 30d)
    tte_idx:    int = 45,    # column index for time to expiry
    option_type: str = "put",
) -> np.ndarray:
    """
    Generate BS delta targets for each observation.

    Returns
    -------
    deltas : (N,) array of BS delta values ∈ [-1, 0] for puts / [0,1] for calls
    """
    N = len(states)
    deltas = np.zeros(N, dtype=np.float32)
    for i, state in enumerate(states):
        S     = float(state[spot_idx]) * 400.0 + 400.0   # de-normalise proxy
        K     = S   # ATM
        T     = max(float(state[tte_idx]) * 90.0 / 365.0, 1e-3)
        r     = max(float(state[rate_idx]) * 0.05 + 0.04, 0.001)
        sigma = max(float(state[iv_idx]) * 0.1 + 0.2, 0.01)
        g     = bs_greeks(S, K, T, r, sigma, option_type)
        deltas[i] = float(g["delta"])
    return deltas


# ── Stage 2: Behavior Cloning ─────────────────────────────────────────────────
class BehaviorCloningTrainer:
    """
    Stage 2: Jointly pretrain Transformer + Policy network on BS delta targets.

    The policy is a simple Gaussian MLP that accepts:
      [obs_raw (49-dim), iv_embedding (128-dim)] → mean_action (scalar or 3D)

    KL regularisation against a frozen checkpoint prevents catastrophic
    forgetting once RL training begins.
    """

    def __init__(
        self,
        transformer: IVSurfaceTransformer,
        policy_net:  nn.Module,
        obs_dim:     int   = 49,
        iv_seq_len:  int   = 30,
        action_dim:  int   = 1,
        lr:          float = 1e-4,
        lambda_kl:   float = 0.01,
        device:      str   = "cpu",
    ):
        self.transformer  = transformer.to(device)
        self.policy_net   = policy_net.to(device)
        self.device       = device
        self.lambda_kl    = lambda_kl
        self.action_dim   = action_dim

        self.frozen_policy: Optional[nn.Module] = None

        params = list(self.transformer.parameters()) + list(self.policy_net.parameters())
        self.optimizer = optim.Adam(params, lr=lr)

    def clone(
        self,
        obs_sequences: np.ndarray,    # (N, T, 25) IV sequences
        obs_states:    np.ndarray,    # (N, obs_dim) current observations
        bs_deltas:     np.ndarray,    # (N, action_dim) BS delta targets
        epochs:        int = 100,
        batch_size:    int = 256,
    ) -> List[float]:
        """
        Train actor to mimic Black-Scholes delta across IV regimes.

        Implements the loss:
          L = MSE(π(s), δ_BS) + λ_KL × KL(π, π_frozen)
        """
        log.info(f"Stage 2 — Behavior Cloning on BS deltas ({epochs} epochs)")
        criterion = nn.MSELoss()

        iv_seq_t  = torch.FloatTensor(obs_sequences).to(self.device)
        states_t  = torch.FloatTensor(obs_states).to(self.device)
        targets_t = torch.FloatTensor(bs_deltas).to(self.device)
        if targets_t.dim() == 1:
            targets_t = targets_t.unsqueeze(-1)

        N = len(iv_seq_t)
        losses = []
        for epoch in range(epochs):
            # Shuffle
            perm  = torch.randperm(N)
            epoch_loss = 0.0
            n_batches  = 0
            for start in range(0, N - batch_size, batch_size):
                idx = perm[start:start + batch_size]
                iv_batch  = iv_seq_t[idx]
                st_batch  = states_t[idx]
                tgt_batch = targets_t[idx]

                # Forward pass
                iv_emb  = self.transformer(iv_batch)                  # (B, 128)
                policy_input = torch.cat([st_batch, iv_emb], dim=-1)  # (B, 49+128)
                pred_action  = self.policy_net(policy_input)           # (B, action_dim)

                mse_loss = criterion(pred_action, tgt_batch)

                # KL regularisation (if frozen policy available)
                kl_loss = torch.tensor(0.0, device=self.device)
                if self.frozen_policy is not None:
                    with torch.no_grad():
                        frozen_action = self.frozen_policy(policy_input)
                    # Approximation: KL ≈ MSE between distributions
                    kl_loss = criterion(pred_action, frozen_action)

                loss = mse_loss + self.lambda_kl * kl_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.transformer.parameters()) + list(self.policy_net.parameters()),
                    max_norm=1.0
                )
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches  += 1

            avg = epoch_loss / max(n_batches, 1)
            losses.append(avg)
            if epoch % 20 == 0:
                log.info(f"  Epoch {epoch:3d}/{epochs} | BC loss: {avg:.6f}")

        # Save frozen policy copy for future KL regularisation
        import copy
        self.frozen_policy = copy.deepcopy(self.policy_net)
        for p in self.frozen_policy.parameters():
            p.requires_grad = False
        log.info("Stage 2 complete. Frozen policy checkpoint saved.")
        return losses

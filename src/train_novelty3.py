"""
train_novelty3.py
------------------
Training script for Novelty 3: Hybrid TradFi-DeFi Meta-Policy.

Prerequisite: Novelty 1 and 2 checkpoints must exist (models are loaded
and their weights fine-tuned jointly with the Meta-Critic and Regime Detector).

Training strategy:
  Phase A — Regime Detector warm-up: train LSTM on synthetic labelled regime
             periods (COVID crash = TradFi, DeFi summer 2021 = DeFi, otherwise Neutral)
  Phase B — Joint meta-policy training:
             Multi-env rollouts from both TradFi and DeFi environments.
             Executive Meta-Critic updates every `meta_update_freq` steps.
  Phase C — End-to-end fine-tuning: unfreeze all modules, joint gradient updates.

Usage:
  cd src
  python train_novelty3.py \
    --tradfi_data  data/processed/master_raw.parquet \
    --defi_data    data/processed/defi_processed.parquet \
    --tradfi_ckpt  checkpoints/novelty1 \
    --save_path    checkpoints/novelty3 \
    --timesteps    300000 \
    --device       cpu
"""

import argparse
import logging
import os
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train Novelty 3 — Hybrid TradFi-DeFi Meta-Policy")
    p.add_argument("--tradfi_data",   default="data/processed/master_raw.parquet")
    p.add_argument("--defi_data",     default="data/processed/defi_processed.parquet")
    p.add_argument("--tradfi_ckpt",   default=None,   help="Path to Novelty 1 DDPG checkpoint")
    p.add_argument("--save_path",     default="checkpoints/novelty3")
    p.add_argument("--device",        default="cpu")
    p.add_argument("--timesteps",     type=int,   default=300_000)
    p.add_argument("--meta_update_freq", type=int, default=256)
    p.add_argument("--lr_meta",       type=float, default=3e-4)
    p.add_argument("--batch_size",    type=int,   default=256)
    p.add_argument("--no_wandb",      action="store_true")
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


def _make_regime_label(date_idx: int, total: int) -> int:
    """
    Assign synthetic regime labels for regime detector warm-up.
    Approximations based on known periods (scaled to relative index):
      - COVID crash (March 2020 onset): TradFi=0
      - DeFi summer (mid 2021):         DeFi=1
      - Otherwise:                       Neutral=2
    """
    frac = date_idx / max(total, 1)
    # COVID-19 crash: ~first 8% of a 2020-2024 dataset
    if 0.0 <= frac < 0.08:
        return 0   # TradFi
    # DeFi summer 2021: ~28-42% through dataset
    elif 0.28 <= frac < 0.42:
        return 1   # DeFi
    # UST crisis Oct 2023: ~82-88%
    elif 0.82 <= frac < 0.88:
        return 0   # TradFi
    else:
        return 2   # Neutral


def pretrain_regime_detector(regime_detector, env_data: np.ndarray, device: str, epochs: int = 30):
    """
    Warm-up the LSTM regime detector with synthetic regime labels.
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim

    log.info("Phase A: Regime Detector warm-up ...")
    seq_len, input_dim = 20, 4
    N = max(len(env_data) - seq_len, 0)
    if N == 0:
        log.warning("Not enough data to pretrain regime detector.")
        return

    # Build sequences: use [spx, rate_1y, rate_10y, vol_proxy] cols (indices 0,4,6,7)
    regime_cols = [0, 4, 6, 7]
    seqs   = []
    labels = []
    for i in range(N):
        seg = env_data[i: i + seq_len, :][:, regime_cols]  # (T, 4)
        seqs.append(seg)
        labels.append(_make_regime_label(i + seq_len, len(env_data)))

    seqs_t   = torch.FloatTensor(np.stack(seqs)).to(device)    # (N, 20, 4)
    labels_t = torch.LongTensor(labels).to(device)             # (N,)

    optimizer = optim.Adam(regime_detector.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    B = 128
    for epoch in range(epochs):
        perm     = torch.randperm(len(seqs_t))
        ep_loss  = 0.0
        n_batches = 0
        for start in range(0, len(seqs_t) - B, B):
            idx  = perm[start:start + B]
            prob, _ = regime_detector(seqs_t[idx])
            loss = criterion(prob, labels_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ep_loss += loss.item()
            n_batches += 1
        if epoch % 10 == 0:
            log.info(f"  Regime warm-up epoch {epoch:3d}/{epochs} | loss={ep_loss/max(n_batches,1):.4f}")
    log.info("Phase A complete.")


def main():
    args = parse_args()
    np.random.seed(args.seed)

    from envs.multi_asset_env import MultiAssetHedgingEnv
    from envs.defi_env import DeFiHedgingEnv
    from models.novelty3_meta.meta_agent import HybridMetaPolicy

    tracker = None
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(project="hedge-derivation", name="novelty3-meta", config=vars(args))
            tracker = wandb
        except ImportError:
            log.warning("wandb not installed — logging to stdout only")

    def log_metrics(step, metrics):
        log.info(f"step={step} | " + " | ".join(f"{k}={v:.5f}" for k, v in metrics.items()))
        if tracker:
            tracker.log(metrics, step=step)

    # ── Environments ───────────────────────────────────────────────────────
    log.info("Loading environments ...")
    tradfi_env = MultiAssetHedgingEnv(args.tradfi_data, train=True, seed=args.seed)
    defi_env   = DeFiHedgingEnv(args.defi_data,         train=True, seed=args.seed)

    # ── Meta-Policy ────────────────────────────────────────────────────────
    agent = HybridMetaPolicy(
        device=args.device,
        lr_meta=args.lr_meta,
        batch_size=args.batch_size,
    )

    # Load pretrained TradFi DDPG if checkpoint exists
    if args.tradfi_ckpt and os.path.isdir(args.tradfi_ckpt):
        agent.tradfi_agent.load(args.tradfi_ckpt)
        log.info(f"Loaded TradFi DDPG ← {args.tradfi_ckpt}")

    # ── Phase A: Regime Detector Warm-up ───────────────────────────────────
    import torch
    pretrain_regime_detector(
        agent.regime_detector,
        tradfi_env.data,
        device=args.device,
        epochs=30,
    )

    # ── Phase B: Joint Meta-Policy Training ────────────────────────────────
    log.info("=== Phase B: Joint meta-policy training ===")

    tradfi_obs, _ = tradfi_env.reset()
    defi_obs, _   = defi_env.reset()

    # Rolling regime sequence window
    regime_cols = [0, 4, 6, 7]
    regime_window = np.zeros((20, 4), dtype=np.float32)

    step = 0
    ep_meta_losses = []
    ep_rewards     = []

    while step < args.timesteps:
        # Update regime window
        regime_window = np.roll(regime_window, shift=-1, axis=0)
        regime_window[-1, :] = tradfi_obs[regime_cols]

        # Build DeFi obs (18-dim): use defi_env observation directly
        global_obs = tradfi_obs   # use TradFi obs as global state proxy

        action_dict = agent.select_action(
            global_obs=global_obs,
            regime_seq=regime_window,
            tradfi_obs=tradfi_obs,
            defi_obs=defi_obs,
            explore=True,
        )

        tradfi_action = action_dict["tradfi_action"]
        defi_action   = action_dict["defi_action"]

        # Step both environments
        next_tradfi_obs, r_tradfi, t_tradfi, tr_tradfi, info_tf = tradfi_env.step(tradfi_action)
        next_defi_obs,   r_defi,   t_defi,   tr_defi,   info_df = defi_env.step(defi_action)

        # Combined meta-reward: weighted by regime (TradFi heavy when p_TradFi high)
        p_tf = action_dict["regime_probs"][0]
        p_df = action_dict["regime_probs"][1]
        reward_meta = p_tf * r_tradfi + p_df * r_defi

        done = (t_tradfi or tr_tradfi) and (t_defi or tr_defi)

        # Update rolling regime window for next state
        next_regime_window = np.roll(regime_window, shift=-1, axis=0)
        next_regime_window[-1, :] = next_tradfi_obs[regime_cols]

        agent.store_transition(
            global_obs=global_obs,
            regime_seq=regime_window.copy(),
            tradfi_obs=tradfi_obs,
            defi_obs=defi_obs,
            tradfi_action=tradfi_action,
            defi_action=defi_action,
            reward_meta=reward_meta,
            next_global_obs=next_tradfi_obs,
            next_regime_seq=next_regime_window.copy(),
            next_tradfi_obs=next_tradfi_obs,
            next_defi_obs=next_defi_obs,
            done=done,
        )

        # Also keep TradFi arm warm with its own replay
        agent.tradfi_agent.store_transition(tradfi_obs, tradfi_action, r_tradfi,
                                            next_tradfi_obs, t_tradfi or tr_tradfi)

        ep_rewards.append(reward_meta)
        tradfi_obs = next_tradfi_obs
        defi_obs   = next_defi_obs

        if done:
            tradfi_obs, _ = tradfi_env.reset()
            defi_obs, _   = defi_env.reset()
            regime_window[:] = 0.0

        # Meta-critic update
        if step % args.meta_update_freq == 0 and step > 0:
            meta_stats = agent.train_step()
            # Also update TradFi DDPG arm
            agent.tradfi_agent.train_step()

            if meta_stats:
                ep_meta_losses.append(meta_stats["meta_critic_loss"])
                log_metrics(step, {
                    "meta_reward_mean": np.mean(ep_rewards[-256:]) if ep_rewards else 0.0,
                    **meta_stats,
                })

        step += 1

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(args.save_path, exist_ok=True)
    agent.save(args.save_path)
    log.info(f"Novelty 3 training complete. Model saved → {args.save_path}")
    if tracker:
        tracker.finish()


if __name__ == "__main__":
    main()

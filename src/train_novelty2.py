"""
train_novelty2.py
------------------
Training script for Novelty 2: IV-Surface Aware BC-RPPO.

Phases:
  1. Transformer self-supervised pretraining (next-IV prediction)
  2. Behavior Cloning pretraining on Black-Scholes deltas
  3. RPPO RL training with pretrained weights + KL regularisation

Usage:
  cd src
  python train_novelty2.py \
    --data_path  data/processed/master_raw.parquet \
    --save_path  checkpoints/novelty2 \
    --device     cpu \
    --timesteps  500000

All phase losses and RL metrics are logged to W&B if available,
otherwise printed to stdout.
"""

import argparse
import logging
import os
import sys
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train Novelty 2 — IV-Surface BC-RPPO")
    p.add_argument("--config",        default=None, help="Path to YAML config file")
    p.add_argument("--data_path",     default=None)
    p.add_argument("--save_path",     default=None)
    p.add_argument("--device",        default=None)
    p.add_argument("--timesteps",     type=int,   default=None)
    p.add_argument("--rollout_len",   type=int,   default=None,
                   help="Steps collected per PPO update")
    p.add_argument("--n_epochs",      type=int,   default=None)
    p.add_argument("--batch_size",    type=int,   default=None)
    p.add_argument("--lr",            type=float, default=None)
    p.add_argument("--bc_epochs",     type=int,   default=None,
                   help="Behavior cloning pretraining epochs")
    p.add_argument("--transformer_epochs", type=int, default=None)
    p.add_argument("--iv_seq_len",    type=int,   default=None)
    p.add_argument("--no_wandb",      action="store_true")
    p.add_argument("--seed",          type=int,   default=None)
    args = p.parse_args()

    # Merge YAML config (if provided) with CLI args (CLI wins)
    defaults = dict(data_path="data/processed/master_raw.parquet",
                    save_path="checkpoints/novelty2", device="cpu",
                    timesteps=500_000, rollout_len=2048, n_epochs=10,
                    batch_size=256, lr=3e-4, bc_epochs=100,
                    transformer_epochs=50, iv_seq_len=30, seed=42)
    if args.config:
        from utils.config import load_config
        cfg = load_config(args.config)
        flat = {**cfg.get("data", {}), **cfg.get("training", {}),
                **cfg.get("optimizer", {}), **cfg.get("rppo", {}),
                **cfg.get("pretraining", {}), **cfg.get("output", {})}
        defaults.update({k: v for k, v in flat.items() if v is not None})
    for k, v in vars(args).items():
        if v is not None and k != "config":
            defaults[k] = v
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return args


def make_iv_sequences(env_data: np.ndarray, seq_len: int = 30) -> np.ndarray:
    """
    Extract rolling IV surface sequences from environment data.

    Each IV surface is a 25-dim flattened 5×5 grid (cols 7:32 in the 49-dim obs).
    Returns (N, seq_len+1, 25) array where last surface is prediction target.
    """
    iv_start, iv_end = 7, 32   # IV grid slice in the 49-dim observation
    n = len(env_data)
    sequences = []
    for i in range(seq_len, n):
        seq = env_data[i - seq_len: i + 1, iv_start:iv_end]
        sequences.append(seq)
    return np.array(sequences, dtype=np.float32)   # (N-seq_len, seq_len+1, 25)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # ── Imports ────────────────────────────────────────────────────────────
    from envs.multi_asset_env import MultiAssetHedgingEnv
    from models.novelty2_bcrppo.iv_transformer import IVSurfaceTransformer
    from models.novelty2_bcrppo.bc_pretrain import (
        pretrain_iv_transformer,
        generate_bs_delta_targets,
        BehaviorCloningTrainer,
    )
    from models.novelty2_bcrppo.rppo_policy import IVSurfaceBCRPPO, GaussianPolicyNet

    # W&B setup
    tracker = None
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(project="hedge-derivation", name="novelty2-bcrppo", config=vars(args))
            tracker = wandb
        except ImportError:
            log.warning("wandb not installed — logging to stdout only")

    def log_metrics(step, metrics):
        log.info(f"step={step} | " + " | ".join(f"{k}={v:.5f}" for k, v in metrics.items()))
        if tracker:
            tracker.log(metrics, step=step)

    # ── Environment ────────────────────────────────────────────────────────
    log.info("Loading environment and data ...")
    env = MultiAssetHedgingEnv(
        data_path=args.data_path,
        option_type="put",
        train=True,
        episode_len=60,
        seed=args.seed,
    )

    # Build raw data matrix for pretraining (use underlying data array)
    raw_data = env.data   # (N, n_features) — already normalised

    # ── Phase 1: Transformer Self-Supervised Pretraining ──────────────────
    log.info("=== Phase 1: IV Transformer Self-Supervised Pretraining ===")
    iv_seqs = make_iv_sequences(raw_data, seq_len=args.iv_seq_len)

    transformer = IVSurfaceTransformer(iv_dim=25, seq_len=args.iv_seq_len)
    stage1_losses = pretrain_iv_transformer(
        transformer,
        iv_sequences=iv_seqs,
        epochs=args.transformer_epochs,
        device=args.device,
    )
    log.info(f"Stage 1 done. Final MSE: {stage1_losses[-1]:.6f}")

    # ── Phase 2: Behavior Cloning ──────────────────────────────────────────
    log.info("=== Phase 2: Behavior Cloning on BS Deltas ===")
    obs_states   = raw_data[args.iv_seq_len:, :]    # (N', 49) — aligned with iv_seqs
    iv_sequences = iv_seqs[:, :-1, :]               # (N', T, 25) — drop the prediction target
    bs_deltas    = generate_bs_delta_targets(obs_states)

    policy_dim   = 49 + 128   # obs + iv_embedding
    policy_net   = GaussianPolicyNet(input_dim=policy_dim, action_dim=1)

    bc_trainer   = BehaviorCloningTrainer(
        transformer=transformer,
        policy_net=policy_net,
        obs_dim=49,
        iv_seq_len=args.iv_seq_len,
        action_dim=1,
        lr=args.lr,
        device=args.device,
    )
    stage2_losses = bc_trainer.clone(
        obs_sequences=iv_sequences,
        obs_states=obs_states,
        bs_deltas=bs_deltas,
        epochs=args.bc_epochs,
    )
    log.info(f"Stage 2 done. Final BC loss: {stage2_losses[-1]:.6f}")

    # ── Phase 3: RPPO RL Training ──────────────────────────────────────────
    log.info("=== Phase 3: RPPO RL Training ===")
    agent = IVSurfaceBCRPPO(
        obs_dim=49,
        iv_dim=25,
        iv_seq_len=args.iv_seq_len,
        action_dim=1,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )
    agent.load_bc_pretrained(
        transformer_state=transformer.state_dict(),
        policy_state=policy_net.state_dict(),
    )

    obs, _ = env.reset()
    # Build initial IV sequence window (use zeros for the first seq_len steps)
    iv_window = np.zeros((args.iv_seq_len, 25), dtype=np.float32)

    step       = 0
    episode    = 0
    ep_rewards = []
    ep_hes     = []

    while step < args.timesteps:
        # Collect rollout
        for _ in range(args.rollout_len):
            # Update IV window with current IV surface from obs
            iv_window = np.roll(iv_window, shift=-1, axis=0)
            iv_window[-1, :] = obs[7:32]   # cols 7:32 = IV grid

            action, log_prob, value = agent.select_action(obs, iv_window, explore=True)
            # Scale action from [-1,1] to full delta; env expects (3,) for multi-asset
            # For single-delta, extend to (3,) by broadcasting the scalar
            full_action = np.array([action[0], 0.0, 0.0], dtype=np.float32)

            next_obs, reward, terminated, truncated, info = env.step(full_action)
            done = terminated or truncated

            agent.store_transition(obs, iv_window.copy(), action, log_prob, reward, value, done)
            ep_rewards.append(reward)
            ep_hes.append(info["hedging_error"])

            obs = next_obs
            step += 1

            if done:
                obs, _ = env.reset()
                iv_window[:] = 0.0
                episode += 1

        # PPO update
        last_val = 0.0
        if not (terminated or truncated):
            iv_window_t = np.roll(iv_window, shift=-1, axis=0)
            iv_window_t[-1, :] = obs[7:32]
            _, _, last_val = agent.select_action(obs, iv_window_t, explore=False)

        update_stats = agent.update(last_value=last_val)
        log_metrics(step, {
            "ep_reward_mean": np.mean(ep_rewards[-200:]),
            "he_mean":        np.mean(np.abs(ep_hes[-200:])),
            **update_stats,
        })
        ep_rewards.clear()
        ep_hes.clear()

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(args.save_path, exist_ok=True)
    agent.save(args.save_path)
    log.info(f"Novelty 2 training complete. Model saved → {args.save_path}")
    if tracker:
        tracker.finish()


if __name__ == "__main__":
    main()

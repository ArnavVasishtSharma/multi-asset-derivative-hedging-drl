"""
train_novelty1.py
------------------
Training script for Novelty 1: Correlation-Aware Multi-Asset DDPG.

Phases (per implementation plan):
  Phase 1 — BC pretraining: actor pretrained on BS delta targets per asset
  Phase 2 — Single-asset calibration (equity, FX, rate independently)
  Phase 3 — Multi-asset joint DDPG training

Usage:
  cd src
  python train_novelty1.py \
    --data_path data/processed/master_raw.parquet \
    --save_path checkpoints/novelty1 \
    --device    cpu \
    --timesteps 1000000
"""

import argparse
import logging
import os
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train Novelty 1 — Multi-Asset DDPG")
    p.add_argument("--config",     default=None, help="Path to YAML config file")
    p.add_argument("--data_path",  default=None)
    p.add_argument("--save_path",  default=None)
    p.add_argument("--device",     default=None)
    p.add_argument("--timesteps",  type=int,   default=None)
    p.add_argument("--episode_len",type=int,   default=None)
    p.add_argument("--start_steps",type=int,   default=None,
                   help="Steps of random exploration before training")
    p.add_argument("--update_freq",type=int,   default=None)
    p.add_argument("--batch_size", type=int,   default=None)
    p.add_argument("--no_wandb",   action="store_true")
    p.add_argument("--seed",       type=int,   default=None)
    args = p.parse_args()

    # Merge YAML config (if provided) with CLI args (CLI wins)
    defaults = dict(data_path="data/processed/master_raw.parquet",
                    save_path="checkpoints/novelty1", device="cpu",
                    timesteps=1_000_000, episode_len=60, start_steps=10_000,
                    update_freq=1, batch_size=256, seed=42)
    if args.config:
        from utils.config import load_config
        cfg = load_config(args.config)
        flat = {**cfg.get("data", {}), **cfg.get("training", {}),
                **cfg.get("optimizer", {}), **cfg.get("ddpg", {}),
                **cfg.get("output", {})}
        defaults.update({k: v for k, v in flat.items() if v is not None})
    # CLI overrides
    for k, v in vars(args).items():
        if v is not None and k != "config":
            defaults[k] = v
    for k, v in defaults.items():
        if getattr(args, k, None) is None:
            setattr(args, k, v)
    return args


def main():
    args = parse_args()
    np.random.seed(args.seed)

    from envs.multi_asset_env import MultiAssetHedgingEnv
    from models.novelty1_ddpg.ddpg_agent import MultiAssetDDPG
    from utils.metrics import episode_summary

    tracker = None
    if not args.no_wandb:
        try:
            import wandb
            wandb.init(project="hedge-derivation", name="novelty1-ddpg", config=vars(args))
            tracker = wandb
        except ImportError:
            log.warning("wandb not installed — logging to stdout only")

    def log_metrics(step, metrics):
        log.info(f"step={step} | " + " | ".join(f"{k}={v:.5f}" for k, v in metrics.items()))
        if tracker:
            tracker.log(metrics, step=step)

    # ── Environment ────────────────────────────────────────────────────────
    env = MultiAssetHedgingEnv(
        data_path=args.data_path,
        option_type="put",
        train=True,
        episode_len=args.episode_len,
        seed=args.seed,
    )

    # ── Agent ──────────────────────────────────────────────────────────────
    agent = MultiAssetDDPG(
        obs_dim=49,
        action_dim=3,
        batch_size=args.batch_size,
        device=args.device,
    )

    obs, _ = env.reset()
    ep_rewards   = []
    ep_pnls      = []
    ep_hes       = []
    ep_reward_sum = 0.0
    episode       = 0
    step          = 0

    log.info(f"=== Novelty 1 Training: {args.timesteps:,} steps ===")

    while step < args.timesteps:
        # Random exploration before training
        if step < args.start_steps:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, explore=True)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        agent.store_transition(obs, action, reward, next_obs, done)
        ep_reward_sum += reward
        ep_hes.append(info["hedging_error"])

        obs = next_obs
        step += 1

        # Training update
        if step >= args.start_steps and step % args.update_freq == 0:
            stats = agent.train_step()
            if stats and step % 5000 == 0:
                log_metrics(step, {
                    "critic_loss":     stats["critic_loss"],
                    "actor_loss":      stats["actor_loss"],
                    "ep_reward_mean":  np.mean(ep_rewards[-50:]) if ep_rewards else 0.0,
                    "he_mean":         np.mean(np.abs(ep_hes[-500:])) if ep_hes else 0.0,
                })

        if done:
            ep_rewards.append(ep_reward_sum)
            ep_reward_sum = 0.0
            episode      += 1
            obs, _        = env.reset()

    # ── Save ───────────────────────────────────────────────────────────────
    os.makedirs(args.save_path, exist_ok=True)
    agent.save(args.save_path)
    log.info(f"Novelty 1 training complete. Model saved → {args.save_path}")
    if tracker:
        tracker.finish()


if __name__ == "__main__":
    main()

"""
evaluate.py
-----------
Evaluation script for all three novelties.

Runs a trained model on the test split of the environment and computes
all evaluation metrics defined in the implementation plan:
  - Sharpe Ratio
  - Hedging Error Variance (HE²)
  - CVaR-95
  - Maximum Drawdown
  - Trade Count (rebalance frequency proxy)

Usage:
  cd src

  # Evaluate Novelty 1 DDPG
  python evaluate.py --model novelty1_ddpg  --ckpt checkpoints/novelty1  --data data/processed/master_raw.parquet

  # Evaluate Novelty 2 BC-RPPO
  python evaluate.py --model novelty2_bcrppo --ckpt checkpoints/novelty2 --data data/processed/master_raw.parquet

  # Evaluate Novelty 3 Meta-Policy
  python evaluate.py --model novelty3_meta   --ckpt checkpoints/novelty3 \
      --data data/processed/master_raw.parquet --defi_data data/processed/defi_processed.parquet

Outputs a JSON summary and optionally a CSV of per-step metrics.
"""

import argparse
import json
import logging
import os
import sys

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate hedging models")
    p.add_argument("--model",       required=True,
                   choices=["novelty1_ddpg", "novelty2_bcrppo", "novelty3_meta", "bs_delta"],
                   help="Model to evaluate")
    p.add_argument("--ckpt",        default=None,    help="Checkpoint directory")
    p.add_argument("--data",        required=True,   help="Path to master_raw.parquet")
    p.add_argument("--defi_data",   default=None,    help="DeFi data path (for novelty3)")
    p.add_argument("--n_episodes",  type=int, default=100,  help="Number of evaluation episodes")
    p.add_argument("--episode_len", type=int, default=60)
    p.add_argument("--device",      default="cpu")
    p.add_argument("--metric",      default=None,
                   help="If set, only print this metric (e.g. hedging_error_variance)")
    p.add_argument("--output_json", default=None,    help="Save results to JSON file")
    p.add_argument("--seed",        type=int, default=99)
    return p.parse_args()


# ── Black-Scholes baseline ────────────────────────────────────────────────────

def bs_delta_action(obs: np.ndarray) -> np.ndarray:
    """
    Compute Black-Scholes delta hedge from the observation vector.
    Returns a (3,) action: [equity_delta, 0, 0] (BS only hedges the equity leg).
    """
    from data.preprocessor import bs_greeks
    S     = float(obs[0]) * 400.0 + 400.0
    K     = S
    T     = max(float(obs[45]) * 90.0 / 365.0, 1e-3)
    r     = max(float(obs[4]) * 0.05 + 0.04, 0.001)
    sigma = max(float(obs[7]) * 0.1 + 0.2, 0.01)
    greeks = bs_greeks(S, K, T, r, sigma, "put")
    delta  = np.clip(float(greeks["delta"]), -1.0, 1.0)
    return np.array([delta, 0.0, 0.0], dtype=np.float32)


# ── Evaluate one episode ──────────────────────────────────────────────────────

def run_episode(model_name, agent_or_fn, env, iv_seq_len: int = 30) -> dict:
    """Run one episode and return per-episode aggregated metrics."""
    obs, _ = env.reset()
    iv_window = np.zeros((iv_seq_len, 25), dtype=np.float32)
    rewards, hes, actions_taken, positions = [], [], [], []
    done = False

    while not done:
        if model_name == "bs_delta":
            action = agent_or_fn(obs)

        elif model_name == "novelty1_ddpg":
            action = agent_or_fn.select_action(obs, explore=False)

        elif model_name == "novelty2_bcrppo":
            iv_window = np.roll(iv_window, shift=-1, axis=0)
            iv_window[-1, :] = obs[7:32]
            single_action, _, _ = agent_or_fn.select_action(obs, iv_window, explore=False)
            action = np.array([single_action[0], 0.0, 0.0], dtype=np.float32)

        elif model_name == "novelty3_meta":
            agent, regime_cols, regime_window, defi_env = agent_or_fn
            regime_window = np.roll(regime_window, shift=-1, axis=0)
            regime_window[-1, :] = obs[regime_cols]
            defi_obs, _ = defi_env.reset() if not hasattr(defi_env, "_ep_started") else (defi_env._current_obs, {})
            result = agent.select_action(
                global_obs=obs, regime_seq=regime_window,
                tradfi_obs=obs, defi_obs=defi_obs, explore=False
            )
            action = result["tradfi_action"]
        else:
            action = np.zeros(3, dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        rewards.append(reward)
        hes.append(info["hedging_error"])
        positions.append(info["position"].copy())
        # Count rebalances: non-zero action changes
        if len(actions_taken) > 0:
            if np.any(np.abs(action - actions_taken[-1]) > 1e-4):
                actions_taken.append(action)
        else:
            actions_taken.append(action)

    from utils.metrics import compute_sharpe, compute_cvar, compute_max_dd, compute_he_variance
    cumr = np.cumsum(rewards)
    return {
        "sharpe":          compute_sharpe(rewards),
        "cvar":            compute_cvar(rewards),
        "max_drawdown":    compute_max_dd(cumr),
        "he_variance":     compute_he_variance(hes),
        "total_return":    float(np.sum(rewards)),
        "n_rebalances":    len(actions_taken),
        "mean_abs_he":     float(np.mean(np.abs(hes))),
    }


# ── Main evaluation loop ──────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(args.seed)

    from envs.multi_asset_env import MultiAssetHedgingEnv

    env = MultiAssetHedgingEnv(
        data_path=args.data,
        option_type="put",
        train=False,   # test split
        episode_len=args.episode_len,
        seed=args.seed,
    )

    # ── Load model ─────────────────────────────────────────────────────────
    if args.model == "bs_delta":
        agent_or_fn = bs_delta_action

    elif args.model == "novelty1_ddpg":
        from models.novelty1_ddpg.ddpg_agent import MultiAssetDDPG
        agent = MultiAssetDDPG(obs_dim=49, action_dim=3, device=args.device)
        if args.ckpt:
            agent.load(args.ckpt)
        agent_or_fn = agent

    elif args.model == "novelty2_bcrppo":
        from models.novelty2_bcrppo.rppo_policy import IVSurfaceBCRPPO
        agent = IVSurfaceBCRPPO(obs_dim=49, iv_dim=25, iv_seq_len=30,
                                 action_dim=1, device=args.device)
        if args.ckpt:
            agent.load(args.ckpt)
        agent_or_fn = agent

    elif args.model == "novelty3_meta":
        from models.novelty3_meta.meta_agent import HybridMetaPolicy
        from envs.defi_env import DeFiHedgingEnv
        agent = HybridMetaPolicy(device=args.device)
        if args.ckpt:
            agent.load(args.ckpt)
        defi_env = DeFiHedgingEnv(
            args.defi_data or "data/processed/defi_processed.parquet",
            train=False, seed=args.seed
        )
        regime_cols   = [0, 4, 6, 7]
        regime_window = np.zeros((20, 4), dtype=np.float32)
        agent_or_fn   = (agent, regime_cols, regime_window, defi_env)

    # ── Run episodes ───────────────────────────────────────────────────────
    log.info(f"Evaluating {args.model} over {args.n_episodes} episodes ...")
    all_results = []
    for ep in range(args.n_episodes):
        result = run_episode(args.model, agent_or_fn, env)
        all_results.append(result)
        if ep % 20 == 0:
            log.info(f"  ep={ep:3d} | Sharpe={result['sharpe']:.3f} | "
                     f"HE_var={result['he_variance']:.5f} | CVaR={result['cvar']:.5f}")

    # ── Aggregate ──────────────────────────────────────────────────────────
    keys = list(all_results[0].keys())
    summary = {
        k: {
            "mean": float(np.mean([r[k] for r in all_results])),
            "std":  float(np.std([r[k] for r in all_results])),
        }
        for k in keys
    }

    print("\n" + "="*60)
    print(f"  Model: {args.model.upper()}")
    print("="*60)
    for k, v in summary.items():
        print(f"  {k:25s}: {v['mean']:+.5f}  ± {v['std']:.5f}")
    print("="*60)

    # If only one metric is requested (for CI pass/fail assertions)
    if args.metric:
        metric_map = {
            "sharpe_ratio":          "sharpe",
            "hedging_error_variance":"he_variance",
            "cvar":                  "cvar",
            "max_drawdown":          "max_drawdown",
        }
        key = metric_map.get(args.metric, args.metric)
        if key in summary:
            print(f"\n{args.metric}: {summary[key]['mean']:.6f}")

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump({"model": args.model, "n_episodes": args.n_episodes, "results": summary}, f, indent=2)
        log.info(f"Results saved → {args.output_json}")


if __name__ == "__main__":
    main()

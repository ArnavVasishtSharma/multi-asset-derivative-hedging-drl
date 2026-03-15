"""
backtest.py
-----------
Backtest runner: run a trained model over the full historical test dataset
to produce a trade-by-trade P&L series and comparison table vs baselines.

Usage:
  cd src
  python backtest.py \
    --model novelty1_ddpg \
    --ckpt  checkpoints/novelty1 \
    --data  data/processed/master_raw.parquet \
    --output_csv results/novelty1_backtest.csv

The script also runs the Black-Scholes baseline and prints a comparison table.
"""

import argparse
import logging
import os
import json

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Backtest hedging model")
    p.add_argument("--model",       required=True,
                   choices=["novelty1_ddpg", "novelty2_bcrppo", "novelty3_meta", "bs_delta"])
    p.add_argument("--ckpt",        default=None)
    p.add_argument("--data",        required=True)
    p.add_argument("--defi_data",   default=None)
    p.add_argument("--output_csv",  default=None)
    p.add_argument("--device",      default="cpu")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def load_env_and_agent(args):
    """Load environment and agent based on --model flag."""
    from envs.multi_asset_env import MultiAssetHedgingEnv

    env = MultiAssetHedgingEnv(
        data_path=args.data, option_type="put", train=False,
        episode_len=60, seed=args.seed
    )

    if args.model == "bs_delta":
        return env, "bs_delta"

    elif args.model == "novelty1_ddpg":
        from models.novelty1_ddpg.ddpg_agent import MultiAssetDDPG
        agent = MultiAssetDDPG(obs_dim=49, action_dim=3, device=args.device)
        if args.ckpt: agent.load(args.ckpt)
        return env, agent

    elif args.model == "novelty2_bcrppo":
        from models.novelty2_bcrppo.rppo_policy import IVSurfaceBCRPPO
        agent = IVSurfaceBCRPPO(obs_dim=49, iv_dim=25, iv_seq_len=30,
                                 action_dim=1, device=args.device)
        if args.ckpt: agent.load(args.ckpt)
        return env, agent

    elif args.model == "novelty3_meta":
        from models.novelty3_meta.meta_agent import HybridMetaPolicy
        from envs.defi_env import DeFiHedgingEnv
        agent = HybridMetaPolicy(device=args.device)
        if args.ckpt: agent.load(args.ckpt)
        defi_env = DeFiHedgingEnv(args.defi_data or "data/processed/defi_processed.parquet",
                                   train=False, seed=args.seed)
        return env, (agent, defi_env)


def run_full_backtest(model_name: str, agent, env) -> pd.DataFrame:
    """
    Step through the entire test dataset sequentially.
    Returns a DataFrame with one row per step.
    """
    from data.preprocessor import bs_greeks

    records = []
    obs, _ = env.reset()
    iv_window    = np.zeros((30, 25), dtype=np.float32)
    regime_window = np.zeros((20, 4), dtype=np.float32)
    done = False
    episode, step = 0, 0

    while True:
        # ── Select action ────────────────────────────────────────────────
        if model_name == "bs_delta":
            S     = float(obs[0]) * 400.0 + 400.0
            K     = S
            T     = max(float(obs[45]) * 90.0 / 365.0, 1e-3)
            r     = max(float(obs[4]) * 0.05 + 0.04, 0.001)
            sigma = max(float(obs[7]) * 0.1 + 0.2, 0.01)
            g     = bs_greeks(S, K, T, r, sigma, "put")
            action = np.array([np.clip(g["delta"], -1.0, 1.0), 0.0, 0.0], dtype=np.float32)
            dominant_regime = "N/A"

        elif model_name == "novelty1_ddpg":
            action = agent.select_action(obs, explore=False)
            dominant_regime = "TradFi"

        elif model_name == "novelty2_bcrppo":
            iv_window = np.roll(iv_window, shift=-1, axis=0)
            iv_window[-1, :] = obs[7:32]
            single_a, _, _ = agent.select_action(obs, iv_window, explore=False)
            action = np.array([single_a[0], 0.0, 0.0], dtype=np.float32)
            dominant_regime = "TradFi"

        elif model_name == "novelty3_meta":
            meta_agent, defi_env = agent
            regime_window = np.roll(regime_window, shift=-1, axis=0)
            regime_window[-1, :] = obs[[0, 4, 6, 7]]
            defi_obs, _ = defi_env.reset()
            result = meta_agent.select_action(
                global_obs=obs, regime_seq=regime_window,
                tradfi_obs=obs, defi_obs=defi_obs, explore=False
            )
            action = result["tradfi_action"]
            dominant_regime = result["dominant_regime"]

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        records.append({
            "step":           step,
            "episode":        episode,
            "reward":         reward,
            "hedging_error":  info["hedging_error"],
            "tx_cost":        info["tx_cost"],
            "cvar":           info["cvar"],
            "delta_equity":   action[0],
            "delta_fx":       action[1],
            "delta_rate":     action[2],
            "regime":         dominant_regime,
        })

        obs = next_obs
        step += 1

        if done:
            episode += 1
            try:
                obs, _ = env.reset()
                iv_window[:] = 0.0
                regime_window[:] = 0.0
            except Exception:
                break   # All test episodes exhausted

        # Safety cap at 50k steps
        if step >= 50_000:
            break

    return pd.DataFrame.from_records(records)


def print_comparison_table(df: pd.DataFrame, model_name: str):
    """Print a structured summary table."""
    from utils.metrics import compute_sharpe, compute_cvar, compute_max_dd, compute_he_variance

    pnl    = df["reward"].values
    hes    = df["hedging_error"].values
    cumr   = np.cumsum(pnl)
    sharpe = compute_sharpe(pnl)
    cvar   = compute_cvar(pnl)
    maxdd  = compute_max_dd(cumr)
    hevar  = compute_he_variance(hes)
    ns     = df["episode"].max() + 1
    rebal  = int((df["delta_equity"].diff().abs() > 1e-4).sum())

    print("\n" + "="*65)
    print(f"  BACKTEST RESULTS — {model_name.upper()}")
    print("="*65)
    print(f"  {'Episodes evaluated:':<35} {ns}")
    print(f"  {'Total steps:':<35} {len(df)}")
    print(f"  {'Annualised Sharpe Ratio:':<35} {sharpe:+.4f}")
    print(f"  {'CVaR-95 (expected tail loss):':<35} {cvar:.6f}")
    print(f"  {'Max Drawdown:':<35} {maxdd:.6f}")
    print(f"  {'Hedging Error Variance:':<35} {hevar:.6f}")
    print(f"  {'Rebalance events:':<35} {rebal}")
    print(f"  {'Mean |HE|:':<35} {np.mean(np.abs(hes)):.6f}")
    print("="*65)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    env, agent = load_env_and_agent(args)

    log.info(f"Running backtest: {args.model} ...")
    df = run_full_backtest(args.model, agent, env)
    print_comparison_table(df, args.model)

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        log.info(f"Backtest data saved → {args.output_csv}")


if __name__ == "__main__":
    main()

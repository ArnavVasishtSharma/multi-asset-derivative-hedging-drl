"""
visualize.py
-------------
Results visualization dashboard for the Multi-Asset DRL Hedging System.

Loads evaluation JSON results and backtest CSVs, generating:
  1. Cumulative PnL curves (all models + BS baseline overlaid)
  2. Bar chart: Sharpe / CVaR / HE-variance per model
  3. Rolling hedging error time series
  4. Regime detection heatmap (Novelty 3 backtest)

Outputs interactive Plotly HTML + static PNG files.

Usage:
  cd src
  python visualize.py --results_dir ../results --output_dir ../results/plots

  # Or specify individual files:
  python visualize.py --json ../results/novelty1.json ../results/novelty2.json \
                      --csv  ../results/backtest_nov1.csv
"""

import argparse
import json
import logging
import os
from typing import List, Optional

import numpy as np
import pandas as pd

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_eval_jsons(json_paths: List[str]) -> dict:
    """Load evaluation JSON results into a unified dict keyed by model name."""
    data = {}
    for p in json_paths:
        if not os.path.isfile(p):
            log.warning(f"JSON not found: {p}")
            continue
        with open(p) as f:
            d = json.load(f)
        data[d.get("model", os.path.basename(p))] = d.get("results", d)
    return data


def load_backtest_csvs(csv_paths: List[str]) -> dict:
    """Load backtest CSVs into a dict keyed by model name (from filename)."""
    data = {}
    for p in csv_paths:
        if not os.path.isfile(p):
            log.warning(f"CSV not found: {p}")
            continue
        name = os.path.splitext(os.path.basename(p))[0].replace("backtest_", "")
        data[name] = pd.read_csv(p)
    return data


# ── Plot: Metric Comparison Bar Chart ─────────────────────────────────────────

def plot_metric_comparison(eval_data: dict, output_dir: str):
    """Bar chart comparing Sharpe, CVaR, HE-variance, max drawdown across models."""
    metrics = ["sharpe", "cvar", "he_variance", "max_drawdown"]
    labels  = ["Sharpe Ratio", "CVaR-95", "HE Variance", "Max Drawdown"]
    models  = list(eval_data.keys())

    if HAS_PLOTLY:
        fig = make_subplots(rows=2, cols=2, subplot_titles=labels)
        colors = px.colors.qualitative.Set2

        for idx, metric in enumerate(metrics):
            row, col = idx // 2 + 1, idx % 2 + 1
            vals = [eval_data[m].get(metric, {}).get("mean", 0) for m in models]
            errs = [eval_data[m].get(metric, {}).get("std", 0) for m in models]
            fig.add_trace(
                go.Bar(
                    x=models, y=vals,
                    error_y=dict(type="data", array=errs, visible=True),
                    marker_color=colors[:len(models)],
                    name=metric,
                    showlegend=False,
                ),
                row=row, col=col,
            )

        fig.update_layout(
            title_text="Model Comparison — Evaluation Metrics",
            height=700, width=1000,
            template="plotly_dark",
        )
        path = os.path.join(output_dir, "metric_comparison.html")
        fig.write_html(path)
        log.info(f"Saved metric comparison → {path}")

    elif HAS_MPL:
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for idx, (metric, label) in enumerate(zip(metrics, labels)):
            ax = axes[idx // 2][idx % 2]
            vals = [eval_data[m].get(metric, {}).get("mean", 0) for m in models]
            errs = [eval_data[m].get(metric, {}).get("std", 0) for m in models]
            ax.bar(models, vals, yerr=errs, capsize=5, color=plt.cm.Set2.colors[:len(models)])
            ax.set_title(label)
            ax.tick_params(axis="x", rotation=30)
        plt.suptitle("Model Comparison — Evaluation Metrics", fontsize=14)
        plt.tight_layout()
        path = os.path.join(output_dir, "metric_comparison.png")
        plt.savefig(path, dpi=150)
        plt.close()
        log.info(f"Saved metric comparison → {path}")


# ── Plot: Cumulative PnL Curves ──────────────────────────────────────────────

def plot_cumulative_pnl(backtest_data: dict, output_dir: str):
    """Overlay cumulative PnL curves from backtest CSVs."""
    if not backtest_data:
        log.warning("No backtest CSVs to plot.")
        return

    if HAS_PLOTLY:
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        for i, (name, df) in enumerate(backtest_data.items()):
            cum_pnl = df["reward"].cumsum()
            fig.add_trace(go.Scatter(
                x=df["step"], y=cum_pnl,
                mode="lines", name=name,
                line=dict(color=colors[i % len(colors)], width=2),
            ))
        fig.update_layout(
            title="Cumulative P&L — Backtest Comparison",
            xaxis_title="Step", yaxis_title="Cumulative Reward (P&L)",
            template="plotly_dark", height=500, width=900,
            legend=dict(x=0.01, y=0.99),
        )
        path = os.path.join(output_dir, "cumulative_pnl.html")
        fig.write_html(path)
        log.info(f"Saved cumulative PnL → {path}")

    elif HAS_MPL:
        plt.figure(figsize=(12, 5))
        for name, df in backtest_data.items():
            cum_pnl = df["reward"].cumsum()
            plt.plot(df["step"], cum_pnl, label=name, linewidth=1.5)
        plt.title("Cumulative P&L — Backtest Comparison")
        plt.xlabel("Step"), plt.ylabel("Cumulative Reward")
        plt.legend(), plt.grid(alpha=0.3)
        path = os.path.join(output_dir, "cumulative_pnl.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Saved cumulative PnL → {path}")


# ── Plot: Rolling Hedging Error ──────────────────────────────────────────────

def plot_hedging_error(backtest_data: dict, output_dir: str, window: int = 50):
    """Rolling mean absolute hedging error time series."""
    if not backtest_data:
        return

    if HAS_PLOTLY:
        fig = go.Figure()
        colors = px.colors.qualitative.Plotly
        for i, (name, df) in enumerate(backtest_data.items()):
            rolling_he = df["hedging_error"].abs().rolling(window, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=df["step"], y=rolling_he,
                mode="lines", name=name,
                line=dict(color=colors[i % len(colors)], width=1.5),
            ))
        fig.update_layout(
            title=f"Rolling Mean |Hedging Error| (window={window})",
            xaxis_title="Step", yaxis_title="|HE| (rolling mean)",
            template="plotly_dark", height=450, width=900,
        )
        path = os.path.join(output_dir, "hedging_error.html")
        fig.write_html(path)
        log.info(f"Saved hedging error → {path}")

    elif HAS_MPL:
        plt.figure(figsize=(12, 4))
        for name, df in backtest_data.items():
            rolling_he = df["hedging_error"].abs().rolling(window, min_periods=1).mean()
            plt.plot(df["step"], rolling_he, label=name, linewidth=1.2)
        plt.title(f"Rolling Mean |Hedging Error| (window={window})")
        plt.xlabel("Step"), plt.ylabel("|HE|")
        plt.legend(), plt.grid(alpha=0.3)
        path = os.path.join(output_dir, "hedging_error.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        log.info(f"Saved hedging error → {path}")


# ── Plot: Regime Heatmap ─────────────────────────────────────────────────────

def plot_regime_heatmap(backtest_data: dict, output_dir: str):
    """Regime classification heatmap from Novelty 3 backtest (if 'regime' column exists)."""
    for name, df in backtest_data.items():
        if "regime" not in df.columns or df["regime"].str.contains("N/A").all():
            continue

        regime_map = {"TradFi": 0, "DeFi": 1, "Neutral": 2}
        numeric_regime = df["regime"].map(regime_map).fillna(2).astype(int)

        if HAS_PLOTLY:
            fig = go.Figure(go.Heatmap(
                z=[numeric_regime.values],
                x=df["step"].values,
                colorscale=[[0, "#1f77b4"], [0.5, "#ff7f0e"], [1, "#2ca02c"]],
                showscale=False,
            ))
            fig.update_layout(
                title=f"Regime Detection — {name}",
                xaxis_title="Step", yaxis_visible=False,
                template="plotly_dark", height=200, width=900,
                annotations=[
                    dict(text="TradFi=Blue  DeFi=Orange  Neutral=Green",
                         x=0.5, y=-0.3, xref="paper", yref="paper",
                         showarrow=False, font_size=11),
                ],
            )
            path = os.path.join(output_dir, f"regime_heatmap_{name}.html")
            fig.write_html(path)
            log.info(f"Saved regime heatmap → {path}")

        elif HAS_MPL:
            fig, ax = plt.subplots(figsize=(14, 1.5))
            ax.imshow([numeric_regime.values], aspect="auto", cmap="Set1",
                       extent=[0, len(df), 0, 1])
            ax.set_title(f"Regime Detection — {name}")
            ax.set_xlabel("Step")
            ax.set_yticks([])
            path = os.path.join(output_dir, f"regime_heatmap_{name}.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            log.info(f"Saved regime heatmap → {path}")


# ── Summary table ────────────────────────────────────────────────────────────

def print_summary_table(eval_data: dict):
    """Print a formatted comparison table to stdout."""
    metrics = ["sharpe", "cvar", "he_variance", "max_drawdown", "total_return"]
    header  = f"{'Model':<20}" + "".join(f"{m:<18}" for m in metrics)
    print("\n" + "=" * len(header))
    print("  EVALUATION SUMMARY")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for model, results in eval_data.items():
        row = f"{model:<20}"
        for m in metrics:
            val = results.get(m, {})
            if isinstance(val, dict):
                row += f"{val.get('mean', 0):>+.4f} ±{val.get('std', 0):.4f}  "
            else:
                row += f"{val:>+.4f}              "
        print(row)
    print("=" * len(header))


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Visualize hedging system results")
    p.add_argument("--results_dir", default=None,
                   help="Directory containing JSON + CSV results (auto-discover)")
    p.add_argument("--json", nargs="*", default=[], help="Evaluation JSON files")
    p.add_argument("--csv",  nargs="*", default=[], help="Backtest CSV files")
    p.add_argument("--output_dir", default="../results/plots")
    return p.parse_args()


def main():
    args = parse_args()

    if not HAS_PLOTLY and not HAS_MPL:
        log.error("Neither plotly nor matplotlib is installed. Cannot generate plots.")
        return

    # Auto-discover from results_dir
    json_files = list(args.json)
    csv_files  = list(args.csv)
    if args.results_dir and os.path.isdir(args.results_dir):
        for f in os.listdir(args.results_dir):
            fp = os.path.join(args.results_dir, f)
            if f.endswith(".json") and fp not in json_files:
                json_files.append(fp)
            elif f.endswith(".csv") and fp not in csv_files:
                csv_files.append(fp)

    os.makedirs(args.output_dir, exist_ok=True)

    eval_data     = load_eval_jsons(json_files)
    backtest_data = load_backtest_csvs(csv_files)

    if eval_data:
        print_summary_table(eval_data)
        plot_metric_comparison(eval_data, args.output_dir)
    else:
        log.warning("No evaluation JSONs found — skipping metric comparison.")

    if backtest_data:
        plot_cumulative_pnl(backtest_data, args.output_dir)
        plot_hedging_error(backtest_data, args.output_dir)
        plot_regime_heatmap(backtest_data, args.output_dir)
    else:
        log.warning("No backtest CSVs found — skipping PnL / HE plots.")

    log.info(f"All plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()

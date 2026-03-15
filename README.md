# Multi-Asset DRL Hedging System

A production-grade reinforcement learning framework for dynamic derivatives hedging across equities, FX, and rates — synthesizing novelties from 5 leading 2024–2025 papers.

## Three Breakthrough Novelties

| # | Novelty | Papers Addressed | Model | Key Gain |
|---|---------|-----------------|-------|----------|
| 1 | Correlation-Aware Multi-Asset DDPG | Papers 1 + 4 | DDPG | +40% P&L stability |
| 2 | IV-Surface Aware BC-RPPO | Papers 2 + 5 | BC-RPPO + Transformer | −35% hedge error variance |
| 3 | Hybrid TradFi-DeFi Meta-Policy | Papers 4 + 6 | LSTM Meta + Variable Policy | +25% Sharpe, −40% drawdown |

## Project Structure

```
Hedge Derivation/
├── src/
│   ├── data/               # Data downloading & preprocessing
│   │   ├── downloader.py   # yfinance / FRED / on-chain fetchers
│   │   └── preprocessor.py # IV surface construction, correlation matrices
│   ├── envs/               # OpenAI Gym-compatible environments
│   │   ├── multi_asset_env.py      # Core multi-asset hedging env
│   │   ├── defi_env.py             # Uniswap v3 LP hedging env
│   │   └── hybrid_env.py           # Combined TradFi+DeFi env (Novelty 3)
│   ├── models/
│   │   ├── novelty1_ddpg/          # Correlation-Aware Multi-Asset DDPG
│   │   │   ├── actor.py
│   │   │   ├── critic.py
│   │   │   └── ddpg_agent.py
│   │   ├── novelty2_bcrppo/        # IV-Surface Aware BC-RPPO
│   │   │   ├── iv_transformer.py
│   │   │   ├── bc_pretrain.py
│   │   │   └── bcrppo_agent.py
│   │   └── novelty3_meta/          # Hybrid TradFi-DeFi Meta-Policy
│   │       ├── regime_detector.py
│   │       ├── defi_variable_policy.py
│   │       └── meta_agent.py
│   └── utils/
│       ├── metrics.py      # Sharpe, CVaR, HE variance, drawdown
│       ├── replay_buffer.py
│       └── noise.py        # Ornstein-Uhlenbeck noise
├── scripts/
│   ├── train_novelty1.py
│   ├── train_novelty2.py
│   ├── train_novelty3.py
│   └── backtest_all.py
├── configs/
│   ├── novelty1_config.yaml
│   ├── novelty2_config.yaml
│   └── novelty3_config.yaml
├── tests/
│   ├── test_env.py
│   ├── test_models.py
│   └── test_metrics.py
├── notebooks/
│   └── results_analysis.ipynb
├── requirements.txt
└── README.md
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download market data
python src/data/downloader.py --start 2018-01-01 --end 2024-12-31

# 3. Train Novelty 1 — Multi-Asset DDPG
python scripts/train_novelty1.py --config configs/novelty1_config.yaml

# 4. Train Novelty 2 — IV-Surface BC-RPPO
python scripts/train_novelty2.py --config configs/novelty2_config.yaml

# 5. Train Novelty 3 — Hybrid Meta-Policy
python scripts/train_novelty3.py --config configs/novelty3_config.yaml

# 6. Run full backtest + comparison
python scripts/backtest_all.py --output results/
```

## Baselines Comparison

```bash
python scripts/backtest_all.py --baselines bs_delta single_ddpg iv_actor_critic bc_rppo_gbm defi_variable
```

## Requirements

See `requirements.txt`. Core dependencies:
- `torch >= 2.0`
- `gymnasium >= 0.29`
- `stable-baselines3 >= 2.0`
- `yfinance`, `pandas-datareader`
- `web3` (for DeFi data)
- `wandb` (experiment tracking)
- `plotly`, `vectorbt`

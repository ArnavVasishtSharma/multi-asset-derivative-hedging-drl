#!/usr/bin/env bash
# ============================================================
# run_pipeline.sh
# Full data download + preprocessing + training pipeline
# Run from project root: bash src/run_pipeline.sh
# ============================================================
set -e   # exit on first error

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$PROJECT_ROOT/src"

cd "$SRC"
echo "Working directory: $(pwd)"

# ── Step 1: Install dependencies ────────────────────────────
echo ""
echo "=== Step 1: Installing dependencies ==="
pip install -r "$PROJECT_ROOT/requirements.txt" --quiet

# ── Step 2: Download data ─────────────────────────────────
echo ""
echo "=== Step 2: Downloading market data (2018-01-01 → 2024-12-31) ==="
python -m data.downloader --start 2018-01-01 --end 2024-12-31

# ── Step 3: Preprocess ────────────────────────────────────
echo ""
echo "=== Step 3: Preprocessing → master_raw.parquet + defi_processed.parquet ==="
python -m data.preprocessor

# ── Step 4: Train Novelty 1 (DDPG) ──────────────────────
echo ""
echo "=== Step 4: Training Novelty 1 — Multi-Asset DDPG ==="
python train_novelty1.py \
    --data_path ../data/processed/master_raw.parquet \
    --save_path ../checkpoints/novelty1 \
    --device    cpu \
    --timesteps 300000 \
    --no_wandb

# ── Step 5: Train Novelty 2 (BC-RPPO) ───────────────────
echo ""
echo "=== Step 5: Training Novelty 2 — IV-Surface BC-RPPO ==="
python train_novelty2.py \
    --data_path         ../data/processed/master_raw.parquet \
    --save_path         ../checkpoints/novelty2 \
    --device            cpu \
    --timesteps         200000 \
    --bc_epochs         50 \
    --transformer_epochs 30 \
    --no_wandb

# ── Step 6: Train Novelty 3 (Meta-Policy) ────────────────
echo ""
echo "=== Step 6: Training Novelty 3 — Hybrid TradFi-DeFi Meta-Policy ==="
python train_novelty3.py \
    --tradfi_data  ../data/processed/master_raw.parquet \
    --defi_data    ../data/processed/defi_processed.parquet \
    --tradfi_ckpt  ../checkpoints/novelty1 \
    --save_path    ../checkpoints/novelty3 \
    --device       cpu \
    --timesteps    150000 \
    --no_wandb

# ── Step 7: Evaluate ──────────────────────────────────────
echo ""
echo "=== Step 7: Evaluation ==="
mkdir -p ../results

python evaluate.py \
    --model   bs_delta \
    --data    ../data/processed/master_raw.parquet \
    --n_episodes 50 \
    --output_json ../results/bs_delta.json

python evaluate.py \
    --model   novelty1_ddpg \
    --ckpt    ../checkpoints/novelty1 \
    --data    ../data/processed/master_raw.parquet \
    --n_episodes 50 \
    --output_json ../results/novelty1.json

python evaluate.py \
    --model   novelty2_bcrppo \
    --ckpt    ../checkpoints/novelty2 \
    --data    ../data/processed/master_raw.parquet \
    --n_episodes 50 \
    --output_json ../results/novelty2.json

python evaluate.py \
    --model    novelty3_meta \
    --ckpt     ../checkpoints/novelty3 \
    --data     ../data/processed/master_raw.parquet \
    --defi_data ../data/processed/defi_processed.parquet \
    --n_episodes 50 \
    --output_json ../results/novelty3.json

# ── Step 8: Backtest comparison table ─────────────────────
echo ""
echo "=== Step 8: Backtest vs Baselines ==="
python backtest.py \
    --model   novelty1_ddpg \
    --ckpt    ../checkpoints/novelty1 \
    --data    ../data/processed/master_raw.parquet \
    --output_csv ../results/backtest_nov1.csv

python backtest.py \
    --model   novelty2_bcrppo \
    --ckpt    ../checkpoints/novelty2 \
    --data    ../data/processed/master_raw.parquet \
    --output_csv ../results/backtest_nov2.csv

python backtest.py \
    --model    novelty3_meta \
    --ckpt     ../checkpoints/novelty3 \
    --data     ../data/processed/master_raw.parquet \
    --defi_data ../data/processed/defi_processed.parquet \
    --output_csv ../results/backtest_nov3.csv

# ── Step 9: Visualize results ────────────────────────────────
echo ""
echo "=== Step 9: Generating result visualizations ==="
python visualize.py --results_dir ../results --output_dir ../results/plots

echo ""
echo "============================================================"
echo " Pipeline complete! Results in: $PROJECT_ROOT/results/"
echo " Plots in: $PROJECT_ROOT/results/plots/"
echo "============================================================"

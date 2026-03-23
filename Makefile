# ============================================================
# Makefile — Multi-Asset DRL Hedging System
# Usage: make <target>
# Run from project root: /Users/arnavvasishtsharma/Desktop/Hedge Derivation/
# ============================================================

SHELL  := /bin/bash
SRC    := src
DATA   := data/processed
CKPTS  := checkpoints

.PHONY: all install data preprocess train-nov1 train-nov2 train-nov3 \
        train-all eval-all backtest visualize test quick-start clean

# ── Default: full pipeline ───────────────────────────────────
all: install data preprocess train-all eval-all backtest visualize

# ── Install dependencies ─────────────────────────────────────
install:
	pip install -r requirements.txt --quiet

# ── Data: Download raw market data ───────────────────────────
data:
	cd $(SRC) && python -m data.downloader --start 2018-01-01 --end 2024-12-31

# ── Preprocess → parquet files ───────────────────────────────
preprocess:
	cd $(SRC) && python -m data.preprocessor

# ── Train individual novelties ────────────────────────────────
train-nov1:
	cd $(SRC) && python train_novelty1.py \
		--config ../configs/novelty1.yaml \
		--data_path ../$(DATA)/master_raw.parquet \
		--save_path ../$(CKPTS)/novelty1 \
		--device cpu --timesteps 1000000 --no_wandb

train-nov2:
	cd $(SRC) && python train_novelty2.py \
		--config ../configs/novelty2.yaml \
		--data_path ../$(DATA)/master_raw.parquet \
		--save_path ../$(CKPTS)/novelty2 \
		--device cpu --timesteps 500000 --no_wandb

train-nov3:
	cd $(SRC) && python train_novelty3.py \
		--config ../configs/novelty3.yaml \
		--tradfi_data ../$(DATA)/master_raw.parquet \
		--defi_data   ../$(DATA)/defi_processed.parquet \
		--tradfi_ckpt ../$(CKPTS)/novelty1 \
		--save_path   ../$(CKPTS)/novelty3 \
		--device cpu --timesteps 300000 --no_wandb

train-all: train-nov1 train-nov2 train-nov3

# ── Quick sanity: data + imports only ────────────────────────
quick-start:
	cd $(SRC) && bash quick_start.sh

# ── Run test suite ───────────────────────────────────────────
test:
	cd $(SRC) && python -m pytest ../tests/ -v --tb=short

# ── Evaluate all models ───────────────────────────────────────
eval-all:
	mkdir -p results
	cd $(SRC) && python evaluate.py --model bs_delta \
		--data ../$(DATA)/master_raw.parquet \
		--n_episodes 100 --output_json ../results/bs_delta.json
	cd $(SRC) && python evaluate.py --model novelty1_ddpg \
		--ckpt ../$(CKPTS)/novelty1 \
		--data ../$(DATA)/master_raw.parquet \
		--n_episodes 100 --output_json ../results/novelty1.json
	cd $(SRC) && python evaluate.py --model novelty2_bcrppo \
		--ckpt ../$(CKPTS)/novelty2 \
		--data ../$(DATA)/master_raw.parquet \
		--n_episodes 100 --output_json ../results/novelty2.json
	cd $(SRC) && python evaluate.py --model novelty3_meta \
		--ckpt ../$(CKPTS)/novelty3 \
		--data ../$(DATA)/master_raw.parquet \
		--defi_data ../$(DATA)/defi_processed.parquet \
		--n_episodes 100 --output_json ../results/novelty3.json

# ── Backtest all models ───────────────────────────────────────
backtest:
	mkdir -p results
	cd $(SRC) && python backtest.py --model bs_delta \
		--data ../$(DATA)/master_raw.parquet \
		--output_csv ../results/backtest_bs.csv
	cd $(SRC) && python backtest.py --model novelty1_ddpg \
		--ckpt ../$(CKPTS)/novelty1 \
		--data ../$(DATA)/master_raw.parquet \
		--output_csv ../results/backtest_nov1.csv
	cd $(SRC) && python backtest.py --model novelty2_bcrppo \
		--ckpt ../$(CKPTS)/novelty2 \
		--data ../$(DATA)/master_raw.parquet \
		--output_csv ../results/backtest_nov2.csv
	cd $(SRC) && python backtest.py --model novelty3_meta \
		--ckpt ../$(CKPTS)/novelty3 \
		--data ../$(DATA)/master_raw.parquet \
		--defi_data ../$(DATA)/defi_processed.parquet \
		--output_csv ../results/backtest_nov3.csv

# ── Visualize results ────────────────────────────────────────
visualize:
	mkdir -p results/plots
	cd $(SRC) && python visualize.py \
		--results_dir ../results \
		--output_dir ../results/plots

# ── Clean checkpoints and results ────────────────────────────
clean:
	rm -rf $(CKPTS) results data/raw data/processed
	@echo "Cleaned checkpoints, results, and data."

#!/usr/bin/env bash
# ============================================================
# quick_start.sh
# Run ONLY the data download + preprocessing (no training).
# Useful to verify the data pipeline works before committing
# to multi-hour training runs.
# Run from project root: bash src/quick_start.sh
# ============================================================
set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$PROJECT_ROOT/src"
cd "$SRC"

echo "=== Installing dependencies ==="
pip install torch numpy pandas scipy yfinance pandas-datareader \
    gymnasium --quiet

echo "=== Downloading data ==="
python -m data.downloader --start 2018-01-01 --end 2024-12-31

echo "=== Preprocessing ==="
python -m data.preprocessor

echo "=== Sanity check - import all model modules ==="
python -c "
import sys; sys.path.insert(0,'.')
from envs.multi_asset_env import MultiAssetHedgingEnv
from envs.defi_env import DeFiHedgingEnv
from models.novelty1_ddpg.ddpg_agent import MultiAssetDDPG
from models.novelty2_bcrppo.iv_transformer import IVSurfaceTransformer
from models.novelty2_bcrppo.bc_pretrain import BehaviorCloningTrainer
from models.novelty2_bcrppo.rppo_policy import IVSurfaceBCRPPO
from models.novelty3_meta.regime_detector import RegimeDetector
from models.novelty3_meta.defi_policy import DeFiVariablePolicy
from models.novelty3_meta.meta_agent import HybridMetaPolicy
from utils.replay_buffer import ReplayBuffer
from utils.noise import OrnsteinUhlenbeck
from utils.metrics import compute_sharpe, compute_cvar
print('All imports OK!')

# Quick constructor checks
agent1 = MultiAssetDDPG(obs_dim=49, action_dim=3)
agent2 = IVSurfaceBCRPPO(obs_dim=49, iv_dim=25, iv_seq_len=30)
agent3 = HybridMetaPolicy()
print('All agents instantiated OK!')
"

echo ""
echo "=== Quick-start complete. Data ready in: \$(realpath ../data/processed/)  ==="
echo "=== Next step: run the full training with bash src/run_pipeline.sh ==="

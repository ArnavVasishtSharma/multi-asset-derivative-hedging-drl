"""
tests/test_models.py
---------------------
Tests for all three novelty model architectures.
Covers instantiation, forward pass, action selection, train step,
and save/load round-trips.
"""

import os
import numpy as np
import torch
import pytest

from models.novelty1_ddpg.actor import MultiAssetActor, CorrelationEncoder
from models.novelty1_ddpg.critic import TwinCritic, CriticNetwork
from models.novelty1_ddpg.ddpg_agent import MultiAssetDDPG
from models.novelty2_bcrppo.iv_transformer import IVSurfaceTransformer
from models.novelty2_bcrppo.rppo_policy import (
    GaussianPolicyNet, ValueNet, IVSurfaceBCRPPO,
)
from models.novelty3_meta.regime_detector import RegimeDetector
from models.novelty3_meta.defi_policy import DeFiVariablePolicy
from models.novelty3_meta.meta_agent import HybridMetaPolicy


# ── Novelty 1: Multi-Asset DDPG ──────────────────────────────────────────────

class TestNovelty1Actor:
    def test_actor_output_shape(self):
        actor = MultiAssetActor(obs_dim=49, action_dim=3)
        x = torch.randn(8, 49)
        out = actor(x)
        assert out.shape == (8, 3)

    def test_actor_output_bounded(self):
        actor = MultiAssetActor(obs_dim=49, action_dim=3)
        x = torch.randn(16, 49)
        out = actor(x)
        assert torch.all(out >= -1.0) and torch.all(out <= 1.0), \
            "Actor outputs should be in [-1, 1] (tanh)"

    def test_correlation_encoder_output(self):
        enc = CorrelationEncoder(corr_input_dim=9, embed_dim=32)
        x = torch.randn(4, 9)
        out = enc(x)
        assert out.shape == (4, 32)


class TestNovelty1Critic:
    def test_critic_output_shape(self):
        critic = CriticNetwork(obs_dim=49, action_dim=3)
        obs = torch.randn(8, 49)
        act = torch.randn(8, 3)
        q = critic(obs, act)
        assert q.shape == (8, 1)

    def test_twin_critic_q_min(self):
        tc = TwinCritic(obs_dim=49, action_dim=3)
        obs = torch.randn(4, 49)
        act = torch.randn(4, 3)
        q1, q2 = tc(obs, act)
        q_min = tc.q_min(obs, act)
        expected = torch.min(q1, q2)
        assert torch.allclose(q_min, expected)


class TestNovelty1DDPG:
    def test_select_action_shape(self):
        agent = MultiAssetDDPG(obs_dim=49, action_dim=3)
        obs = np.random.randn(49).astype(np.float32)
        action = agent.select_action(obs, explore=True)
        assert action.shape == (3,)
        assert np.all(action >= -1.0) and np.all(action <= 1.0)

    def test_train_step_returns_none_when_buffer_empty(self):
        agent = MultiAssetDDPG(obs_dim=49, action_dim=3, batch_size=32)
        result = agent.train_step()
        assert result is None

    def test_train_step_after_fill(self):
        agent = MultiAssetDDPG(obs_dim=49, action_dim=3, batch_size=8, buffer_size=100)
        for _ in range(20):
            obs = np.random.randn(49).astype(np.float32)
            act = np.random.randn(3).astype(np.float32)
            agent.store_transition(obs, act, -0.1, np.random.randn(49).astype(np.float32), False)
        result = agent.train_step()
        assert result is not None
        assert "critic_loss" in result
        assert "actor_loss" in result

    def test_save_load_roundtrip(self, tmp_checkpoint_dir):
        agent = MultiAssetDDPG(obs_dim=49, action_dim=3)
        obs = np.random.randn(49).astype(np.float32)
        action_before = agent.select_action(obs, explore=False)

        agent.save(tmp_checkpoint_dir)
        agent2 = MultiAssetDDPG(obs_dim=49, action_dim=3)
        agent2.load(tmp_checkpoint_dir)
        action_after = agent2.select_action(obs, explore=False)

        np.testing.assert_allclose(action_before, action_after, atol=1e-6)


# ── Novelty 2: IV-Surface BC-RPPO ────────────────────────────────────────────

class TestNovelty2:
    def test_iv_transformer_output(self):
        t = IVSurfaceTransformer(iv_dim=25, seq_len=30, embed_dim=128)
        x = torch.randn(4, 30, 25)
        emb = t(x)
        assert emb.shape == (4, 128)

    def test_gaussian_policy_output(self):
        policy = GaussianPolicyNet(input_dim=177, action_dim=1)
        x = torch.randn(8, 177)
        mean, log_std = policy(x)
        assert mean.shape == (8, 1)
        assert log_std.shape == (8, 1)

    def test_gaussian_policy_get_action(self):
        policy = GaussianPolicyNet(input_dim=177, action_dim=1)
        x = torch.randn(4, 177)
        action, log_prob, det_action = policy.get_action(x)
        assert action.shape == (4, 1)
        assert log_prob.shape == (4, 1)
        assert torch.all(action >= -1.0) and torch.all(action <= 1.0)

    def test_value_net_output(self):
        vn = ValueNet(input_dim=177)
        x = torch.randn(4, 177)
        v = vn(x)
        assert v.shape == (4, 1)

    def test_bcrppo_select_action(self):
        agent = IVSurfaceBCRPPO(obs_dim=49, iv_dim=25, iv_seq_len=30, action_dim=1)
        obs = np.random.randn(49).astype(np.float32)
        iv_seq = np.random.randn(30, 25).astype(np.float32)
        action, log_prob, value = agent.select_action(obs, iv_seq, explore=True)
        assert action.shape == (1,)
        assert isinstance(log_prob, float)
        assert isinstance(value, float)

    def test_bcrppo_no_trade_zone(self):
        agent = IVSurfaceBCRPPO(obs_dim=49, iv_dim=25, iv_seq_len=30,
                                 action_dim=1, no_trade_eps=10.0)  # huge threshold
        obs = np.random.randn(49).astype(np.float32)
        iv_seq = np.random.randn(30, 25).astype(np.float32)
        action, _, _ = agent.select_action(obs, iv_seq, explore=False)
        # With eps=10.0, all actions should be masked to 0
        np.testing.assert_allclose(action, 0.0, atol=1e-6)

    def test_bcrppo_save_load(self, tmp_checkpoint_dir):
        agent = IVSurfaceBCRPPO(obs_dim=49, iv_dim=25, iv_seq_len=30, action_dim=1)
        agent.save(tmp_checkpoint_dir)
        agent2 = IVSurfaceBCRPPO(obs_dim=49, iv_dim=25, iv_seq_len=30, action_dim=1)
        agent2.load(tmp_checkpoint_dir)
        # Should not raise


# ── Novelty 3: Meta-Policy ───────────────────────────────────────────────────

class TestNovelty3:
    def test_regime_detector_output_shape(self):
        rd = RegimeDetector()
        x = torch.randn(4, 20, 4)
        probs, hidden = rd(x)
        assert probs.shape == (4, 3)
        # Probabilities should sum to ~1
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_defi_policy_output(self):
        dp = DeFiVariablePolicy()
        obs = torch.randn(4, 14)
        action, gating_weights, sub_actions = dp(obs)
        assert action.shape == (4, 3)
        assert gating_weights.shape[0] == 4

    def test_meta_agent_select_action(self):
        agent = HybridMetaPolicy(device="cpu")
        global_obs = np.random.randn(49).astype(np.float32)
        regime_seq = np.random.randn(20, 4).astype(np.float32)
        tradfi_obs = np.random.randn(49).astype(np.float32)
        defi_obs   = np.random.randn(14).astype(np.float32)

        result = agent.select_action(global_obs, regime_seq, tradfi_obs, defi_obs)
        assert "tradfi_action" in result
        assert "defi_action" in result
        assert "regime_probs" in result
        assert "final_position" in result
        assert "dominant_regime" in result
        assert result["tradfi_action"].shape == (3,)
        assert result["defi_action"].shape == (3,)
        assert result["regime_probs"].shape == (3,)
        assert result["dominant_regime"] in ("TradFi", "DeFi", "Neutral")

    def test_meta_agent_save_load(self, tmp_checkpoint_dir):
        agent = HybridMetaPolicy(device="cpu")
        agent.save(tmp_checkpoint_dir)
        agent2 = HybridMetaPolicy(device="cpu")
        agent2.load(tmp_checkpoint_dir)
        # Should not raise

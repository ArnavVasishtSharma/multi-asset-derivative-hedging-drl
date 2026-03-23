"""
tests/test_metrics.py
----------------------
Unit tests for all evaluation metrics in utils/metrics.py.
"""

import numpy as np
import pytest

from utils.metrics import (
    compute_sharpe,
    compute_cvar,
    compute_max_dd,
    compute_he_variance,
    episode_summary,
)


class TestSharpe:
    def test_constant_returns_zero_sharpe(self):
        """Constant returns → zero excess std → Sharpe = 0."""
        returns = [0.001] * 100
        assert compute_sharpe(returns) == 0.0

    def test_positive_excess_returns(self):
        """Clearly positive excess returns should give positive Sharpe."""
        rng = np.random.default_rng(42)
        returns = 0.01 + rng.normal(0, 0.005, 500)  # high mean, low vol
        sharpe = compute_sharpe(returns)
        assert sharpe > 0, f"Expected positive Sharpe, got {sharpe}"

    def test_short_sequence_returns_zero(self):
        assert compute_sharpe([0.01]) == 0.0

    def test_no_annualise(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(0.001, 0.01, 100)
        s_ann = compute_sharpe(returns, annualise=True)
        s_raw = compute_sharpe(returns, annualise=False)
        # Annualised should be ~√252 times daily
        assert abs(s_ann / (s_raw * np.sqrt(252))) - 1 < 0.01


class TestCVaR:
    def test_uniform_distribution(self):
        """CVaR at 95% on uniform [-1, 1] should be close to mean of bottom 5%."""
        rng = np.random.default_rng(0)
        pnl = rng.uniform(-1, 1, 10_000)
        cvar = compute_cvar(pnl, confidence=0.95)
        # Bottom 5% of uniform[-1,1] ≈ [-1, -0.9], mean ≈ -0.95, CVaR ≈ 0.95
        assert 0.8 < cvar < 1.05, f"CVaR={cvar}, expected ~0.95"

    def test_short_sequence_returns_zero(self):
        assert compute_cvar([0.1, 0.2]) == 0.0

    def test_all_positive_pnl(self):
        """All positive P&L → CVaR should be negative (i.e., returned as negative)."""
        cvar = compute_cvar([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        # CVaR = -mean(tail), tail is smallest values which are positive → CVaR < 0
        assert cvar < 0


class TestMaxDrawdown:
    def test_known_drawdown(self):
        cumulative = [0, 1, 2, 3, 1, 0, 2, 3, 4]
        dd = compute_max_dd(cumulative)
        # Peak at 3, trough at 0 → drawdown = 3
        assert dd == 3.0

    def test_monotonic_increase_zero_drawdown(self):
        cumulative = list(range(50))
        dd = compute_max_dd(cumulative)
        assert dd == 0.0

    def test_short_sequence(self):
        assert compute_max_dd([1.0]) == 0.0


class TestHEVariance:
    def test_matches_numpy(self):
        rng = np.random.default_rng(10)
        hes = rng.normal(0, 0.05, 200)
        var = compute_he_variance(hes)
        expected = float(np.var(hes, ddof=1))
        np.testing.assert_allclose(var, expected, rtol=1e-6)

    def test_zero_errors(self):
        var = compute_he_variance([0.0] * 50)
        assert var == 0.0

    def test_short_sequence(self):
        assert compute_he_variance([0.1]) == 0.0


class TestEpisodeSummary:
    def test_all_keys_present(self):
        pnl = np.random.randn(100).tolist()
        hes = np.random.randn(100).tolist()
        summary = episode_summary(pnl, hes)
        expected_keys = {"sharpe", "cvar", "max_drawdown", "he_variance",
                         "total_return", "n_steps"}
        assert set(summary.keys()) == expected_keys

    def test_n_steps_correct(self):
        pnl = [0.1] * 42
        summary = episode_summary(pnl, pnl)
        assert summary["n_steps"] == 42

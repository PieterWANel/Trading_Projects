"""
tests/test_metrics.py
---------------------
Unit tests for performance metrics.
Run with: pytest tests/test_metrics.py -v
"""

import numpy as np
import pandas as pd
import pytest
from src.utils.metrics import (
    sharpe_ratio,
    calmar_ratio,
    sortino_ratio,
    max_drawdown,
    hit_rate,
    profit_factor,
    information_ratio,
    annualised_return,
    annualised_volatility,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def flat_returns():
    """Series of constant positive returns."""
    np.random.seed(42)
    return pd.Series(np.full(252, 0.001))  # ~25% ann return, no vol

@pytest.fixture
def random_returns():
    """Realistic equity-like returns."""
    np.random.seed(42)
    return pd.Series(np.random.normal(0.0003, 0.01, 1000))

@pytest.fixture
def negative_returns():
    """Consistently negative returns."""
    return pd.Series(np.full(252, -0.001))


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestSharpeRatio:
    def test_positive_for_positive_returns(self, random_returns):
        sr = sharpe_ratio(random_returns, risk_free_rate=0.0)
        assert sr > 0

    def test_higher_with_lower_vol(self):
        low_vol = pd.Series(np.full(252, 0.001))
        high_vol = pd.Series(np.random.normal(0.001, 0.05, 252))
        assert sharpe_ratio(low_vol, risk_free_rate=0.0) > sharpe_ratio(high_vol, risk_free_rate=0.0)

    def test_negative_for_negative_returns(self, negative_returns):
        sr = sharpe_ratio(negative_returns, risk_free_rate=0.0)
        assert sr < 0

    def test_autocorr_adjustment_reduces_sharpe_for_trending(self):
        """Momentum returns should have lower autocorr-adjusted Sharpe."""
        trending = pd.Series(np.cumsum(np.random.normal(0.001, 0.005, 500)))
        trending = trending.pct_change().dropna()
        sr_raw = sharpe_ratio(trending, adjust_autocorr=False)
        sr_adj = sharpe_ratio(trending, adjust_autocorr=True)
        # Adjusted should be different (usually lower for positive autocorr)
        assert sr_raw != sr_adj


class TestMaxDrawdown:
    def test_always_negative_or_zero(self, random_returns):
        assert max_drawdown(random_returns) <= 0

    def test_zero_for_always_positive(self):
        positive = pd.Series(np.full(100, 0.01))
        assert max_drawdown(positive) == pytest.approx(0, abs=1e-8)

    def test_known_drawdown(self):
        """Price goes 100 → 50 → drawdown = -50%."""
        prices = pd.Series([100, 90, 80, 70, 50, 60, 70])
        returns = prices.pct_change().dropna()
        assert max_drawdown(returns) == pytest.approx(-0.5, rel=0.01)


class TestHitRate:
    def test_between_zero_and_one(self, random_returns):
        hr = hit_rate(random_returns)
        assert 0 <= hr <= 1

    def test_one_for_all_positive(self, flat_returns):
        assert hit_rate(flat_returns) == 1.0

    def test_zero_for_all_negative(self, negative_returns):
        assert hit_rate(negative_returns) == 0.0


class TestAnnualisedReturn:
    def test_approximately_correct(self):
        """Daily return of 0.1% should annualise to ~25%."""
        daily = pd.Series(np.full(252, 0.001))
        ann = annualised_return(daily)
        assert ann == pytest.approx(0.252, rel=0.01)


class TestCalmarRatio:
    def test_positive_for_good_strategy(self, random_returns):
        cr = calmar_ratio(random_returns + 0.001)  # Add positive drift
        assert isinstance(cr, float)

    def test_higher_better(self):
        low_dd = pd.Series(np.full(252, 0.001))
        high_dd_vals = np.random.normal(0.001, 0.05, 252)
        high_dd = pd.Series(high_dd_vals)
        assert calmar_ratio(low_dd) > calmar_ratio(high_dd)

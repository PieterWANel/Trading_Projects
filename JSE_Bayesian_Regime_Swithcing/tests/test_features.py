"""
tests/test_features.py
----------------------
Unit tests for the feature engineering pipeline.
Run with: pytest tests/test_features.py -v

Critical property tested throughout: NO LOOK-AHEAD BIAS.
Every feature at time t must use only data available at t.
"""

import pytest
import pandas as pd
import numpy as np

from src.data.features import (
    compute_log_returns,
    realized_volatility,
    vol_of_vol,
    rolling_autocorrelation,
    rolling_skewness,
    drawdown_series,
    zscore,
    rand_stress_indicator,
    commodity_regime_signal,
    build_features,
    get_model_input,
    load_config,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def sample_prices():
    """Synthetic price DataFrame with all required columns."""
    np.random.seed(123)
    n = 500
    dates = pd.date_range("2018-01-02", periods=n, freq="B")

    return pd.DataFrame({
        "index":  np.cumprod(1 + np.random.normal(0.0003, 0.012, n)) * 50000,
        "usdzar": np.cumprod(1 + np.random.normal(0.0001, 0.008, n)) * 15.0,
        "gold":   np.cumprod(1 + np.random.normal(0.0002, 0.010, n)) * 1800,
        "brent":  np.cumprod(1 + np.random.normal(0.0001, 0.015, n)) * 70,
        "vix":    np.abs(np.random.normal(18, 6, n)),
    }, index=dates)


@pytest.fixture
def sample_returns(sample_prices):
    return compute_log_returns(sample_prices)


# ── Log Returns Tests ─────────────────────────────────────────────────────────

class TestLogReturns:

    def test_first_row_is_nan(self, sample_prices):
        """First return must be NaN (no prior price)."""
        returns = compute_log_returns(sample_prices)
        assert returns.iloc[0].isna().all()

    def test_correct_shape(self, sample_prices):
        returns = compute_log_returns(sample_prices)
        assert returns.shape == sample_prices.shape

    def test_log_returns_near_pct_for_small_moves(self, sample_prices):
        """For small daily moves, log returns ≈ simple returns."""
        log_ret = compute_log_returns(sample_prices["index"]).dropna()
        pct_ret = sample_prices["index"].pct_change().dropna()
        # Should be close (within 0.1% difference for typical daily moves)
        diff = (log_ret - pct_ret).abs()
        assert diff.mean() < 0.001

    def test_symmetry(self):
        """Log return of +10% followed by -10% should approximately cancel."""
        prices = pd.Series([100, 110, 100])
        returns = compute_log_returns(prices).dropna()
        assert abs(returns.sum()) < 0.01


# ── Realised Volatility Tests ─────────────────────────────────────────────────

class TestRealizedVolatility:

    def test_always_positive(self, sample_returns):
        rv = realized_volatility(sample_returns["index"], window=21)
        assert (rv.dropna() >= 0).all()

    def test_nan_during_warmup(self, sample_returns):
        window = 21
        rv = realized_volatility(sample_returns["index"], window=window)
        # First (window - 1) non-NaN return rows should produce NaN RV
        n_nans = rv.isna().sum()
        assert n_nans >= window - 1

    def test_higher_vol_series_produces_higher_rv(self):
        """High-vol series should produce higher realised vol than low-vol series."""
        np.random.seed(0)
        low_vol = pd.Series(np.random.normal(0, 0.005, 100))
        high_vol = pd.Series(np.random.normal(0, 0.03, 100))
        rv_low = realized_volatility(low_vol, 21).dropna().mean()
        rv_high = realized_volatility(high_vol, 21).dropna().mean()
        assert rv_high > rv_low

    def test_annualisation(self):
        """Constant daily vol of 1% should annualise to ~15.87%."""
        const_ret = pd.Series(np.full(100, 0.01))
        rv = realized_volatility(const_ret, 21).dropna()
        expected = 0.01 * np.sqrt(252)
        assert abs(rv.iloc[-1] - expected) < 0.001


# ── Rolling Autocorrelation Tests ─────────────────────────────────────────────

class TestRollingAutocorrelation:

    def test_output_in_valid_range(self, sample_returns):
        ac = rolling_autocorrelation(sample_returns["index"], window=21)
        valid = ac.dropna()
        assert (valid >= -1).all() and (valid <= 1).all()

    def test_positive_for_trending_series(self):
        """A monotonically increasing series should have positive autocorrelation."""
        trending = pd.Series(np.linspace(0, 1, 100))
        returns = trending.diff().dropna()
        ac = rolling_autocorrelation(returns, window=50).dropna()
        # Trending returns should have positive mean autocorrelation
        assert ac.mean() > 0

    def test_negative_for_mean_reverting_series(self):
        """Alternating +/- returns should have negative autocorrelation."""
        alternating = pd.Series(np.tile([0.01, -0.01], 50))
        ac = rolling_autocorrelation(alternating, window=30).dropna()
        assert ac.mean() < 0


# ── Drawdown Tests ────────────────────────────────────────────────────────────

class TestDrawdown:

    def test_always_between_minus_one_and_zero(self, sample_prices):
        dd = drawdown_series(sample_prices["index"])
        assert (dd >= -1).all() and (dd <= 0).all()

    def test_zero_when_at_all_time_high(self, sample_prices):
        dd = drawdown_series(sample_prices["index"])
        ath_dates = sample_prices["index"] == sample_prices["index"].expanding().max()
        # At ATH, drawdown should be 0
        assert (dd[ath_dates] == 0).all()

    def test_known_drawdown_value(self):
        prices = pd.Series([100.0, 120.0, 90.0, 110.0])
        dd = drawdown_series(prices)
        # After 120, drops to 90: drawdown = (90-120)/120 = -25%
        assert abs(dd.iloc[2] - (-0.25)) < 1e-6


# ── Z-Score Tests ─────────────────────────────────────────────────────────────

class TestZScore:

    def test_mean_approximately_zero(self, sample_returns):
        z = zscore(sample_returns["index"].dropna(), window=63)
        assert abs(z.dropna().mean()) < 0.1

    def test_std_approximately_one(self, sample_returns):
        z = zscore(sample_returns["index"].dropna(), window=63)
        assert abs(z.dropna().std() - 1) < 0.2


# ── Full Feature Pipeline Tests ───────────────────────────────────────────────

class TestBuildFeatures:

    def test_output_is_dataframe(self, sample_prices, config):
        features = build_features(sample_prices, config)
        assert isinstance(features, pd.DataFrame)

    def test_no_nulls_after_build(self, sample_prices, config):
        features = build_features(sample_prices, config)
        assert features.isnull().sum().sum() == 0, "Features should have no NaN after warmup drop"

    def test_expected_columns_present(self, sample_prices, config):
        features = build_features(sample_prices, config)
        required = [
            "jse_return", "jse_rv", "jse_autocorr",
            "zar_rv", "vix_level", "sa_stress", "commodity_signal",
        ]
        for col in required:
            assert col in features.columns, f"Missing feature: {col}"

    def test_z_scored_columns_present(self, sample_prices, config):
        features = build_features(sample_prices, config)
        z_cols = [c for c in features.columns if c.endswith("_z")]
        assert len(z_cols) >= 5, "Should have at least 5 z-scored features"

    def test_no_lookahead_bias(self, sample_prices, config):
        """
        Critical: features computed on full dataset vs. up to T should match at T.
        Tests that rolling functions don't accidentally use future data.
        """
        cutoff = len(sample_prices) // 2
        prices_partial = sample_prices.iloc[:cutoff]
        prices_full = sample_prices

        features_partial = build_features(prices_partial, config)
        features_full = build_features(prices_full, config)

        # Last row of partial features should match corresponding row of full features
        last_date = features_partial.index[-1]
        if last_date in features_full.index:
            for col in features_partial.columns:
                if col in features_full.columns:
                    val_partial = features_partial.loc[last_date, col]
                    val_full = features_full.loc[last_date, col]
                    assert abs(val_partial - val_full) < 1e-6, (
                        f"Look-ahead bias detected in column '{col}': "
                        f"partial={val_partial:.6f}, full={val_full:.6f}"
                    )

    def test_date_index_preserved(self, sample_prices, config):
        features = build_features(sample_prices, config)
        assert features.index.name == "date" or features.index.dtype == "datetime64[ns]"

    def test_fewer_rows_than_input(self, sample_prices, config):
        """Feature warmup period should trim some rows."""
        features = build_features(sample_prices, config)
        assert len(features) < len(sample_prices)


class TestGetModelInput:

    def test_returns_only_z_scored_cols(self, sample_prices, config):
        features = build_features(sample_prices, config)
        model_input = get_model_input(features)
        assert all(c.endswith("_z") for c in model_input.columns)

    def test_no_nulls(self, sample_prices, config):
        features = build_features(sample_prices, config)
        model_input = get_model_input(features)
        assert model_input.isnull().sum().sum() == 0

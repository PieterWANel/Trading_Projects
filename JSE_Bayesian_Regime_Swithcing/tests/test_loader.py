"""
tests/test_loader.py
--------------------
Unit tests for the data loading pipeline.
Run with: pytest tests/test_loader.py -v

Note: Tests use small synthetic data to avoid hitting yfinance in CI.
Integration tests that require network access are marked with @pytest.mark.integration.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from datetime import datetime, timedelta

from src.data.loader import YFinanceLoader, IressLoader, load_config


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    return load_config()


@pytest.fixture
def sample_prices():
    """Synthetic price DataFrame matching the expected output of download_all()."""
    np.random.seed(42)
    dates = pd.date_range("2020-01-02", periods=500, freq="B")  # Business days
    n = len(dates)

    df = pd.DataFrame({
        "index":  np.cumprod(1 + np.random.normal(0.0003, 0.012, n)) * 50000,
        "usdzar": np.cumprod(1 + np.random.normal(0.0001, 0.008, n)) * 15.0,
        "gold":   np.cumprod(1 + np.random.normal(0.0002, 0.010, n)) * 1800,
        "brent":  np.cumprod(1 + np.random.normal(0.0001, 0.015, n)) * 70,
        "vix":    np.abs(np.random.normal(18, 6, n)),
    }, index=dates)
    df.index.name = "date"
    return df


@pytest.fixture
def loader(config):
    return YFinanceLoader(config)


# ── YFinanceLoader Tests ───────────────────────────────────────────────────────

class TestYFinanceLoader:

    def test_init_sets_dates(self, config):
        loader = YFinanceLoader(config)
        assert loader.start == config["data"]["start_date"]
        assert loader.end != "today"  # Should be resolved to actual date string

    def test_raw_path_created(self, config, tmp_path):
        config["paths"]["raw_data"] = str(tmp_path / "raw")
        loader = YFinanceLoader(config)
        assert loader.raw_path.exists()

    def test_save_and_load_raw(self, loader, sample_prices, tmp_path):
        """Round-trip: save → load should return identical DataFrame."""
        loader.raw_path = tmp_path
        loader.save_raw(sample_prices, "test_prices.csv")
        loaded = loader.load_raw("test_prices.csv")
        pd.testing.assert_frame_equal(sample_prices, loaded, check_freq=False)

    def test_load_raw_raises_if_missing(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_raw("nonexistent_file.csv")

    def test_save_creates_file(self, loader, sample_prices, tmp_path):
        loader.raw_path = tmp_path
        loader.save_raw(sample_prices, "prices.csv")
        assert (tmp_path / "prices.csv").exists()

    @patch("src.data.loader.yf.download")
    def test_download_ticker_returns_dataframe(self, mock_download, loader):
        """download_ticker should rename Close column and return clean Series."""
        dates = pd.date_range("2020-01-02", periods=10, freq="B")
        mock_df = pd.DataFrame({"Close": np.random.uniform(50000, 60000, 10)}, index=dates)
        mock_download.return_value = mock_df

        result = loader.download_ticker("^J200", "index")

        assert isinstance(result, pd.DataFrame)
        assert "index" in result.columns
        assert result.index.name == "date"

    @patch("src.data.loader.yf.download")
    def test_download_ticker_handles_empty_response(self, mock_download, loader):
        """Empty yfinance response should return empty DataFrame gracefully."""
        mock_download.return_value = pd.DataFrame()
        result = loader.download_ticker("INVALID", "bad_ticker")
        assert result.empty

    @patch("src.data.loader.yf.download")
    def test_download_all_merges_tickers(self, mock_download, loader):
        """download_all should merge all configured tickers into one DataFrame."""
        dates = pd.date_range("2020-01-02", periods=100, freq="B")

        def side_effect(ticker, **kwargs):
            return pd.DataFrame(
                {"Close": np.random.uniform(1, 100, 100)},
                index=dates,
            )

        mock_download.side_effect = side_effect
        result = loader.download_all()
        assert isinstance(result, pd.DataFrame)
        assert len(result.columns) == len(loader.cfg["data"]["tickers"])


# ── IressLoader Tests ─────────────────────────────────────────────────────────

class TestIressLoader:

    def test_load_file_parses_csv(self, config, tmp_path):
        """IressLoader should correctly parse a standard Iress export format."""
        config["paths"]["raw_data"] = str(tmp_path)
        iress_dir = tmp_path / "iress"
        iress_dir.mkdir()

        # Create fake Iress-format CSV
        csv_content = """Date,Open,High,Low,Close,Volume
01/01/2020,50000,51000,49000,50500,1000000
02/01/2020,50500,52000,50000,51000,1200000
03/01/2020,51000,51500,50000,50800,900000
"""
        (iress_dir / "J200.csv").write_text(csv_content)

        loader = IressLoader(config)
        result = loader.load_file("J200.csv", "index")

        assert isinstance(result, pd.Series)
        assert result.name == "index"
        assert len(result) == 3

    def test_load_file_raises_for_missing(self, config, tmp_path):
        config["paths"]["raw_data"] = str(tmp_path)
        loader = IressLoader(config)
        with pytest.raises(FileNotFoundError):
            loader.load_file("missing.csv", "index")


# ── Data Quality Tests ────────────────────────────────────────────────────────

class TestDataQuality:

    def test_no_negative_prices(self, sample_prices):
        assert (sample_prices > 0).all().all(), "Prices should always be positive"

    def test_date_index_is_sorted(self, sample_prices):
        assert sample_prices.index.is_monotonic_increasing, "Dates should be sorted ascending"

    def test_no_large_gaps(self, sample_prices):
        """No gap between consecutive rows longer than 5 calendar days (JSE holiday tolerance)."""
        gaps = sample_prices.index.to_series().diff().dt.days.dropna()
        assert (gaps <= 5).all(), f"Unexpected large gap in data: {gaps.max()} days"

    def test_returns_not_extreme(self, sample_prices):
        """Daily returns should stay within reasonable bounds (±30%)."""
        returns = sample_prices.pct_change().dropna()
        assert (returns.abs() < 0.30).all().all(), "Extreme daily returns detected"

    def test_vix_positive(self, sample_prices):
        assert (sample_prices["vix"] > 0).all(), "VIX should always be positive"


# ── Integration Tests (require network) ──────────────────────────────────────

@pytest.mark.integration
class TestIntegration:

    def test_download_jse_top40(self, config):
        """Sanity check: can we actually download JSE Top 40 data?"""
        loader = YFinanceLoader(config)
        # Override to short date range for speed
        loader.start = "2023-01-01"
        loader.end = "2023-06-30"
        result = loader.download_ticker("^J200", "index")
        assert not result.empty
        assert "index" in result.columns
        assert len(result) > 50  # At least 50 trading days

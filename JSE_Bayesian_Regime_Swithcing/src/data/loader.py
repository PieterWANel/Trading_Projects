"""
src/data/loader.py
------------------
Data ingestion pipeline for the JSE Regime Strategy.
Supports yfinance (primary) and Iress CSV exports (alternative).

Usage:
    python -m src.data.loader
"""

import os
import logging
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import yfinance as yf
import yaml

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Config ───────────────────────────────────────────────────────────────────
def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── yfinance Loader ───────────────────────────────────────────────────────────
class YFinanceLoader:
    """
    Downloads OHLCV data for all strategy tickers via yfinance.
    Handles JSE-specific quirks: public holidays, rand gaps, missing data.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.start = config["data"]["start_date"]
        self.end = (
            datetime.today().strftime("%Y-%m-%d")
            if config["data"]["end_date"] == "today"
            else config["data"]["end_date"]
        )
        self.raw_path = Path(config["paths"]["raw_data"])
        self.raw_path.mkdir(parents=True, exist_ok=True)

    def download_ticker(self, ticker: str, name: str) -> pd.DataFrame:
        """Download single ticker, return cleaned daily close series."""
        logger.info(f"Downloading {name} ({ticker})...")
        try:
            df = yf.download(
                ticker,
                start=self.start,
                end=self.end,
                auto_adjust=True,
                progress=False,
            )
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            df = df[["Close"]].rename(columns={"Close": name})
            df.index = pd.to_datetime(df.index)
            df.index.name = "date"

            # Forward fill up to 5 days (handles JSE public holidays)
            df = df.ffill(limit=5)

            logger.info(f"  └─ {name}: {len(df)} rows | {df.index[0].date()} → {df.index[-1].date()}")
            return df

        except Exception as e:
            logger.error(f"Failed to download {ticker}: {e}")
            return pd.DataFrame()

    def download_all(self) -> pd.DataFrame:
        """Download all configured tickers and merge into a single DataFrame."""
        tickers = self.cfg["data"]["tickers"]

        frames = []
        for name, ticker in tickers.items():
            df = self.download_ticker(ticker, name)
            if not df.empty:
                frames.append(df)

        if not frames:
            raise RuntimeError("No data downloaded. Check ticker configuration.")

        # Align on common business day index
        merged = pd.concat(frames, axis=1)
        merged = merged.dropna(subset=[merged.columns[0]])  # Require first ticker data at minimum

        logger.info(f"\nMerged dataset: {merged.shape[0]} rows × {merged.shape[1]} columns")
        logger.info(f"Date range: {merged.index[0].date()} → {merged.index[-1].date()}")
        logger.info(f"Missing values:\n{merged.isnull().sum()}")

        return merged

    def save_raw(self, df: pd.DataFrame, filename: str = "raw_prices.csv") -> None:
        path = self.raw_path / filename
        df.to_csv(path)
        logger.info(f"Raw data saved to {path}")

    def load_raw(self, filename: str = "raw_prices.csv") -> pd.DataFrame:
        path = self.raw_path / filename
        if not path.exists():
            raise FileNotFoundError(f"Raw data not found at {path}. Run download_all() first.")
        df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
        df.columns = df.columns.get_level_values(0)
        df.index.name = "date"
        logger.info(f"Loaded raw data: {df.shape} from {path}")
        return df


# ── Iress CSV Loader ──────────────────────────────────────────────────────────
class IressLoader:
    """
    Loads data exported from Iress as CSV files.
    Iress export format: Date, Open, High, Low, Close, Volume
    
    Usage:
        Place CSV exports in data/raw/iress/
        Name files as: J200.csv, USDZAR.csv, etc.
    """

    COLUMN_MAP = {
        "Date": "date",
        "Close": "close",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Volume": "volume",
    }

    def __init__(self, config: dict):
        self.cfg = config
        self.iress_path = Path(config["paths"]["raw_data"]) / "iress"
        self.iress_path.mkdir(parents=True, exist_ok=True)

    def load_file(self, filename: str, name: str) -> pd.Series:
        """Load single Iress CSV export, return named close price series."""
        path = self.iress_path / filename
        if not path.exists():
            raise FileNotFoundError(f"Iress file not found: {path}")

        df = pd.read_csv(path)
        df.columns = [self.COLUMN_MAP.get(c, c.lower()) for c in df.columns]
        df["date"] = pd.to_datetime(df["date"], dayfirst=True)  # Iress uses DD/MM/YYYY
        df = df.set_index("date").sort_index()

        series = df["close"].rename(name)
        logger.info(f"Loaded {name} from Iress: {len(series)} rows")
        return series

    def load_all(self, file_map: dict) -> pd.DataFrame:
        """
        Load multiple Iress files.
        
        Args:
            file_map: dict mapping {name: filename}, e.g. {"index": "J200.csv"}
        
        Returns:
            Merged DataFrame of close prices
        """
        frames = [self.load_file(fname, name) for name, fname in file_map.items()]
        merged = pd.concat(frames, axis=1).ffill(limit=5)
        logger.info(f"Merged Iress data: {merged.shape}")
        return merged


# ── Unified Interface ─────────────────────────────────────────────────────────
def get_price_data(
    config: dict,
    source: str = "yfinance",
    force_download: bool = False,
    iress_file_map: dict = None,
) -> pd.DataFrame:
    """
    Unified data loading interface. Checks for cached data before downloading.
    
    Args:
        config:          Loaded settings.yaml config dict
        source:          "yfinance" or "iress"
        force_download:  Re-download even if cached data exists
        iress_file_map:  Required if source="iress": {name: filename} mapping
    
    Returns:
        DataFrame of daily close prices, columns named by asset
    """
    raw_path = Path(config["paths"]["raw_data"])
    cache_file = raw_path / "raw_prices.csv"

    if source == "yfinance":
        loader = YFinanceLoader(config)

        if cache_file.exists() and not force_download:
            logger.info(f"Cache found at {cache_file}. Loading cached data.")
            logger.info("Use force_download=True to refresh.")
            return loader.load_raw()

        df = loader.download_all()
        loader.save_raw(df)
        return df

    elif source == "iress":
        if iress_file_map is None:
            raise ValueError("iress_file_map required when source='iress'")
        loader = IressLoader(config)
        return loader.load_all(iress_file_map)

    else:
        raise ValueError(f"Unknown source: {source}. Use 'yfinance' or 'iress'.")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    config = load_config()
    df = get_price_data(config, source="yfinance", force_download=True)
    print("\nPrice data preview:")
    print(df.tail(10))
    print(f"\nShape: {df.shape}")
    print(f"\nMissing values:\n{df.isnull().sum()}")

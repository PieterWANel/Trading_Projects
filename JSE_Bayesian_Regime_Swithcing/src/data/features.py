"""
src/data/features.py
--------------------
Feature engineering pipeline for JSE regime detection.
Transforms raw price data into model-ready features.

All features are designed to be forward-looking safe (no look-ahead bias):
each feature at time t uses only data available at time t.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/settings.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ── Core Feature Functions ────────────────────────────────────────────────────

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Log returns for all price series."""
    return np.log(prices / prices.shift(1))


def realized_volatility(returns: pd.Series, window: int) -> pd.Series:
    """
    Annualised rolling realised volatility.
    Computed as rolling std of log returns × sqrt(252).
    """
    return returns.rolling(window).std() * np.sqrt(252)


def vol_of_vol(vol_series: pd.Series, window: int) -> pd.Series:
    """
    Volatility of volatility — second-order uncertainty signal.
    Elevated vol-of-vol indicates regime uncertainty / transition.
    """
    return vol_series.rolling(window).std()


def rolling_autocorrelation(returns: pd.Series, window: int, lag: int = 1) -> pd.Series:
    """
    Rolling lag-1 autocorrelation of returns.
    - Positive autocorr → trending / momentum regime
    - Negative autocorr → mean-reverting / choppy regime
    """
    return returns.rolling(window).apply(
        lambda x: pd.Series(x).autocorr(lag=lag), raw=False
    )


def rolling_skewness(returns: pd.Series, window: int) -> pd.Series:
    """Rolling skewness — elevated negative skew signals stress regime."""
    return returns.rolling(window).skew()


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Rolling drawdown from recent peak."""
    rolling_max = prices.expanding().max()
    return (prices / rolling_max) - 1


def zscore(series: pd.Series, window: int) -> pd.Series:
    """Rolling z-score normalisation."""
    mu = series.rolling(window).mean()
    sigma = series.rolling(window).std()
    return (series - mu) / (sigma + 1e-8)


# ── SA-Specific Features ──────────────────────────────────────────────────────

def rand_stress_indicator(usdzar_returns: pd.Series, vix: pd.Series, window: int = 21) -> pd.Series:
    """
    Composite SA-specific stress indicator.
    Combines USDZAR volatility with VIX level — both elevated = EM crisis regime.
    Z-scored and averaged.
    """
    usdzar_vol_z = zscore(realized_volatility(usdzar_returns, window), window * 3)
    vix_z = zscore(vix, window * 3)
    return (usdzar_vol_z + vix_z) / 2


def commodity_regime_signal(gold_returns: pd.Series, brent_returns: pd.Series, window: int = 21) -> pd.Series:
    """
    Commodity cycle signal for JSE resources sector.
    Rolling average of gold + brent returns, z-scored.
    Positive → commodity tailwind (resources bid) | Negative → headwind
    """
    combined = (gold_returns + brent_returns) / 2
    return zscore(combined.rolling(window).mean(), window * 3)


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def build_features(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    
    Args:
        prices:  DataFrame with columns [jse_top40, usdzar, gold, brent, vix]
        config:  Settings dict from settings.yaml
    
    Returns:
        DataFrame of features, aligned to prices jse_top40, no look-ahead bias
    """
    vol_win = config["features"]["vol_window"]         # 21
    ac_win = config["features"]["autocorr_window"]     # 21
    vov_win = config["features"]["vol_of_vol_window"]  # 63

    logger.info("Building features...")

    # ── Returns ──────────────────────────────────────────────────────────────
    ret = compute_log_returns(prices)

    feat = pd.DataFrame(index=prices.index)

    # ── JSE Top 40 features ───────────────────────────────────────────────────
    feat["jse_return"] = ret["jse_top40"]
    feat["jse_rv"] = realized_volatility(ret["jse_top40"], vol_win)
    feat["jse_rv_long"] = realized_volatility(ret["jse_top40"], vov_win)
    feat["jse_vov"] = vol_of_vol(feat["jse_rv"], vov_win)
    feat["jse_autocorr"] = rolling_autocorrelation(ret["jse_top40"], ac_win)
    feat["jse_skew"] = rolling_skewness(ret["jse_top40"], vol_win)
    feat["jse_drawdown"] = drawdown_series(prices["jse_top40"])

    # Volatility ratio: short-term vs long-term vol (regime transition signal)
    feat["jse_vol_ratio"] = feat["jse_rv"] / (feat["jse_rv_long"] + 1e-8)

    # ── USDZAR features ───────────────────────────────────────────────────────
    feat["zar_return"] = ret["usdzar"]
    feat["zar_rv"] = realized_volatility(ret["usdzar"], vol_win)
    feat["zar_rv_z"] = zscore(feat["zar_rv"], vov_win)

    # ── VIX features ──────────────────────────────────────────────────────────
    feat["vix_level"] = prices["vix"]
    feat["vix_change"] = ret["vix"]
    feat["vix_z"] = zscore(prices["vix"], vov_win)

    # ── Commodity features ────────────────────────────────────────────────────
    feat["gold_return"] = ret["gold"]
    feat["brent_return"] = ret["brent"]
    feat["commodity_signal"] = commodity_regime_signal(ret["gold"], ret["brent"], vol_win)

    # ── Composite SA stress indicator ─────────────────────────────────────────
    feat["sa_stress"] = rand_stress_indicator(ret["usdzar"], prices["vix"], vol_win)

    # ── Normalised features for MCMC input ───────────────────────────────────
    # Model features: subset used directly in Markov-switching model
    model_features = [
        "jse_rv",
        "jse_vov",
        "jse_autocorr",
        "jse_vol_ratio",
        "zar_rv",
        "vix_z",
        "commodity_signal",
        "sa_stress",
    ]

    for col in model_features:
        feat[f"{col}_z"] = zscore(feat[col], vov_win)

    logger.info(f"Features built: {feat.shape[1]} columns")
    logger.info(f"Dropping NaN rows from feature warm-up period...")

    feat = feat.dropna(subset=["jse_return", "jse_rv", "zar_rv", "vix_level"])
    feat = feat.fillna(0)
    logger.info(f"Feature matrix: {feat.shape[0]} rows × {feat.shape[1]} columns")
    logger.info(f"Date range: {feat.index[0].date()} → {feat.index[-1].date()}")

    return feat


def get_model_input(features: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the subset of features used as direct inputs to the regime model.
    Normalised (z-scored) versions only.
    """
    z_cols = [c for c in features.columns if c.endswith("_z")]
    return features[z_cols].copy()


def save_features(features: pd.DataFrame, config: dict, filename: str = "features.csv") -> None:
    path = Path(config["paths"]["processed_data"])
    path.mkdir(parents=True, exist_ok=True)
    features.to_csv(path / filename)
    logger.info(f"Features saved to {path / filename}")


def load_features(config: dict, filename: str = "features.csv") -> pd.DataFrame:
    path = Path(config["paths"]["processed_data"]) / filename
    if not path.exists():
        raise FileNotFoundError(f"Features not found at {path}. Run build_features() first.")
    return pd.read_csv(path, jse_top40_col="date", parse_dates=True)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data.loader import get_price_data

    config = load_config()
    prices = get_price_data(config)
    features = build_features(prices, config)
    save_features(features, config)

    print("\nFeature preview:")
    print(features.tail(5).to_string())
    print(f"\nModel input columns:")
    print(get_model_input(features).columns.tolist())

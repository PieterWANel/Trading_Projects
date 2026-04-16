"""
src/strategy/router.py
----------------------
Regime-to-strategy routing logic.

Takes regime probabilities + confidence from the Bayesian model
and maps them to concrete position sizing decisions.

Separation of concerns:
- router.py  → decides WHAT position to take (regime logic)
- backtest.py → simulates HOW that plays out (P&L, costs, metrics)

Usage:
    from src.strategy.router import RegimeRouter
    router = RegimeRouter(config)
    positions = router.compute_positions(regime_probs, confidence, prices)
"""

import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Signal Generators ─────────────────────────────────────────────────────────

class SignalGenerator:
    """
    Collection of directional signals used by the regime router.
    Each signal returns a value in [0, 1]:
        0 = flat / no exposure
        1 = full long exposure
    """

    @staticmethod
    def momentum(prices: pd.Series, lookback: int = 5) -> pd.Series:
        """
        Time-series momentum signal.
        Long when recent return is positive, flat otherwise.
        Shifted by 1 to avoid look-ahead bias (signal set at close, traded next open).
        """
        ret = prices.pct_change(lookback)
        signal = (ret > 0).astype(float)
        return signal.shift(1).rename("momentum_signal")

    @staticmethod
    def mean_reversion(prices: pd.Series, window: int = 21) -> pd.Series:
        """
        Z-score mean reversion signal.
        Long when price is below rolling mean (oversold), flat when above.
        Used in choppy / high-vol regimes where momentum breaks down.
        """
        rolling_mean = prices.rolling(window).mean()
        rolling_std = prices.rolling(window).std()
        zscore = (prices - rolling_mean) / (rolling_std + 1e-8)
        # Long when z-score < -0.5 (oversold), flat otherwise
        signal = (zscore < -0.5).astype(float)
        return signal.shift(1).rename("mean_reversion_signal")

    @staticmethod
    def volatility_scaled(signal: pd.Series, returns: pd.Series,
                          target_vol: float = 0.15, window: int = 21) -> pd.Series:
        """
        Scale any signal inversely to realised volatility.
        Targets a constant annualised portfolio volatility of target_vol.
        Clips scaling between 0.1x and 2.0x to avoid extreme leverage.
        """
        rv = returns.rolling(window).std() * np.sqrt(252)
        scale = (target_vol / (rv + 1e-8)).clip(0.1, 2.0)
        return (signal * scale).clip(0, 1).rename(f"{signal.name}_volscaled")

    @staticmethod
    def trend_strength(prices: pd.Series, short: int = 10, long: int = 50) -> pd.Series:
        """
        Moving average crossover strength signal.
        Returns fractional signal proportional to MA spread (not just binary cross).
        """
        ma_short = prices.ewm(span=short).mean()
        ma_long = prices.ewm(span=long).mean()
        spread = (ma_short - ma_long) / (ma_long + 1e-8)
        # Normalise to [0, 1]: positive spread = bullish
        signal = (spread > 0).astype(float)
        return signal.shift(1).rename("trend_signal")


# ── Regime Router ─────────────────────────────────────────────────────────────

class RegimeRouter:
    """
    Maps regime states to position sizes.

    Logic:
        1. Identify dominant regime from posterior probabilities
        2. Select appropriate sub-signal for that regime
        3. Scale by posterior confidence (uncertainty penalty)
        4. Apply regime-specific position limits from config

    The key insight: in a Risk-Off regime, we don't just flip to short —
    we reduce exposure and wait. In SA markets, timing the rebound is
    more valuable than trying to profit from the drawdown itself.
    """

    REGIME_SIGNAL_MAP = {
        "Risk-On":      "momentum",
        "Risk-Off":     "defensive",
        "Stagflation":  "mean_reversion",
    }

    def __init__(self, config: dict):
        self.cfg = config
        self.limits = config["strategy"]["position_limits"]
        self.min_conf = config["strategy"]["min_regime_confidence"]
        self.signals = SignalGenerator()

    def _get_base_limit(self, regime_label: str) -> float:
        """Look up max long allocation for a given regime label."""
        # Map label to config key (handles partial matches)
        label_lower = regime_label.lower()
        if "risk-on" in label_lower or "growth" in label_lower:
            return self.limits["risk_on"]["max_long"]
        elif "risk-off" in label_lower or "stress" in label_lower:
            return self.limits["risk_off"]["max_long"]
        else:  # Stagflation, mixed, unknown
            return self.limits["stagflation"]["max_long"]

    def _select_signal(
        self,
        regime_label: str,
        prices: pd.Series,
        returns: pd.Series,
    ) -> pd.Series:
        """Select and return the appropriate directional signal for the regime."""
        if "Risk-On" in regime_label:
            raw = self.signals.momentum(prices, lookback=5)
            return self.signals.volatility_scaled(raw, returns)
        elif "Risk-Off" in regime_label:
            # Defensive: flat unless trend is strongly positive
            return self.signals.trend_strength(prices, short=5, long=20) * 0.3
        else:  # Stagflation
            mr = self.signals.mean_reversion(prices, window=21)
            return self.signals.volatility_scaled(mr, returns)

    def compute_positions(
        self,
        regime_probs: pd.DataFrame,
        confidence: pd.Series,
        prices: pd.Series,
    ) -> pd.DataFrame:
        """
        Main position computation method.

        Args:
            regime_probs: DataFrame (T × K) of posterior regime probabilities
            confidence:   Series of position confidence scores [0,1] from Bayesian model
            prices:       JSE Top 40 price series

        Returns:
            DataFrame with columns:
                position        → final position weight [0, 1]
                base_limit      → regime max allocation
                signal          → directional signal value
                confidence      → posterior confidence
                dominant_regime → regime label at each time step
        """
        # Align indices
        common_idx = prices.index.intersection(regime_probs.index)
        prices = prices.reindex(common_idx)
        regime_probs = regime_probs.reindex(common_idx)
        confidence = confidence.reindex(common_idx).fillna(0.5)

        returns = np.log(prices / prices.shift(1)).fillna(0)

        dominant_regime = regime_probs.idxmax(axis=1)
        dominant_prob = regime_probs.max(axis=1)

        # Pre-compute signals for each regime type
        signals_by_regime = {
            label: self._select_signal(label, prices, returns)
            for label in regime_probs.columns
        }

        results = pd.DataFrame(index=common_idx)
        results["dominant_regime"] = dominant_regime
        results["regime_prob"] = dominant_prob
        results["confidence"] = confidence

        positions = np.zeros(len(common_idx))
        base_limits = np.zeros(len(common_idx))
        signal_values = np.zeros(len(common_idx))

        for i, date in enumerate(common_idx):
            regime = dominant_regime.iloc[i]
            conf = confidence.iloc[i]

            # Gate: don't trade if confidence is too low
            if conf < self.min_conf:
                positions[i] = 0.0
                continue

            base_limit = self._get_base_limit(regime)
            signal = signals_by_regime[regime].get(date, 0.0) if date in signals_by_regime[regime].index else 0.0

            # Final position = base limit × signal × confidence
            positions[i] = base_limit * signal * conf
            base_limits[i] = base_limit
            signal_values[i] = signal

        results["base_limit"] = base_limits
        results["signal"] = signal_values
        results["position"] = np.clip(positions, 0, 1)

        logger.info(f"Positions computed: {len(results)} periods")
        logger.info(f"  Average position: {results['position'].mean():.2%}")
        logger.info(f"  Days fully invested (>90%): {(results['position'] > 0.9).sum()}")
        logger.info(f"  Days in cash (<10%): {(results['position'] < 0.1).sum()}")
        logger.info(f"\n  Regime breakdown:")
        logger.info(f"\n{dominant_regime.value_counts().to_string()}")

        return results

    def regime_transition_summary(self, regime_probs: pd.DataFrame) -> pd.DataFrame:
        """
        Summarise regime transition frequencies from the data.
        Useful for validating that the model produces realistic sticky regimes
        (you shouldn't be switching regime every day).
        """
        regimes = regime_probs.idxmax(axis=1)
        transitions = pd.crosstab(
            regimes.shift(1).rename("From"),
            regimes.rename("To"),
            normalize="index",
        )
        return transitions.round(3)


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data.loader import get_price_data, load_config

    config = load_config()
    prices = get_price_data(config)

    processed = Path(config["paths"]["processed_data"])

    try:
        regime_probs = pd.read_csv(
            processed / "regime_probs_bayesian.csv", index_col="date", parse_dates=True
        )
        confidence = pd.read_csv(
            processed / "position_confidence.csv", index_col="date", parse_dates=True
        ).squeeze()
        logger.info("Using Bayesian regime probabilities.")
    except FileNotFoundError:
        logger.warning("Bayesian results not found. Using classical output.")
        regime_probs = pd.read_csv(
            processed / "regime_probs_classical.csv", index_col="date", parse_dates=True
        )
        confidence = pd.Series(1.0, index=regime_probs.index, name="confidence")

    router = RegimeRouter(config)
    positions = router.compute_positions(regime_probs, confidence, prices["index"])

    print("\nPosition snapshot (last 10 rows):")
    print(positions.tail(10).to_string())

    print("\nRegime transition matrix:")
    print(router.regime_transition_summary(regime_probs).to_string())

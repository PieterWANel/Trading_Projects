"""
src/strategy/backtest.py
------------------------
Regime-conditional strategy router and vectorbt backtesting engine.

Strategy logic:
    Each period, the regime model outputs:
    1. Regime label (Risk-On / Risk-Off / Stagflation)
    2. Posterior confidence score [0, 1]

    Position size = Regime base size × Confidence score × Signal strength

    Regime routing:
    - Risk-On:      Momentum signal → long JSE Top 40
    - Risk-Off:     Defensive → reduce to 30% or cash
    - Stagflation:  Mixed → 50% exposure, tighter stops

Usage:
    python -m src.strategy.backtest
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from src.utils.metrics import performance_summary, regime_performance_breakdown, drawdown_series

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Signal Generation ─────────────────────────────────────────────────────────

def momentum_signal(prices: pd.Series, lookback: int = 63) -> pd.Series:
    """63-day momentum — quarterly trend signal, much more stable."""
    ret = prices.pct_change(lookback)
    signal = (ret > 0).astype(float)
    return signal.shift(1)


def volatility_scaled_signal(signal: pd.Series, returns: pd.Series, target_vol: float = 0.15, window: int = 21) -> pd.Series:
    """
    Scale position size inversely to realised volatility.
    Target annualised volatility = target_vol.
    """
    rv = returns.rolling(window).std() * np.sqrt(252)
    scale = (target_vol / (rv + 1e-8)).clip(0.1, 2.0)
    return (signal * scale).clip(0, 1)


# ── Position Sizing ───────────────────────────────────────────────────────────

def compute_positions(
    regime_probs: pd.DataFrame,
    confidence: pd.Series,
    prices: pd.Series,
    config: dict,
) -> pd.Series:
    returns = np.log(prices / prices.shift(1))
    mom_signal = momentum_signal(prices, lookback=5)
    vol_signal = volatility_scaled_signal(mom_signal, returns)

    dominant_regime = regime_probs.idxmax(axis=1)
    positions = pd.Series(0.0, index=regime_probs.index, name="position")

    for date in regime_probs.index:
        regime = dominant_regime.get(date, "Risk-On")
        conf = confidence.get(date, 0.5)
        signal = vol_signal.get(date, 0.0)

        if "Risk-On" in regime:
            # Full exposure scaled by vol signal
            positions[date] = 1.0 * signal * conf
        elif "Risk-Off" in regime:
            # Reduced but not zero — stay 30% invested
            positions[date] = 0.3 * conf
        else:  # Stagflation
            # Half exposure
            positions[date] = 0.5 * signal * conf

    return positions.clip(0, 1)


# ── Transaction Costs ─────────────────────────────────────────────────────────

def apply_transaction_costs(
    returns: pd.Series,
    positions: pd.Series,
    cost_bps: float = 30,
    slippage_bps: float = 5,
) -> pd.Series:
    """
    Apply realistic JSE transaction costs.
    
    Costs triggered on position changes (turnover).
    30bps round trip + 5bps slippage = ~35bps per trade.
    """
    total_cost_pct = (cost_bps + slippage_bps) / 10000
    position_changes = positions.diff().abs().fillna(0)
    cost_drag = position_changes * total_cost_pct
    return returns - cost_drag


# ── Core Backtest Engine ──────────────────────────────────────────────────────

class RegimeBacktest:
    """
    Backtesting engine for the regime-switching strategy.
    """

    def __init__(self, config: dict):
        self.cfg = config
        self.results = {}

    def run(
        self,
        prices: pd.Series,
        regime_probs: pd.DataFrame,
        confidence: pd.Series,
        label: str = "Bayesian Regime Strategy",
    ) -> pd.DataFrame:
        """
        Execute the full backtest.
        
        Args:
            prices:       JSE Top 40 price series
            regime_probs: Posterior regime probability DataFrame
            confidence:   Position confidence series from Bayesian model
            label:        Strategy label for results
        
        Returns:
            DataFrame of daily strategy returns
        """
        logger.info(f"Running backtest: {label}")

        # Align all series
        common_idx = prices.index.intersection(regime_probs.index)
        prices = prices.reindex(common_idx)
        regime_probs = regime_probs.reindex(common_idx)
        confidence = confidence.reindex(common_idx).fillna(0.5)

        # Daily index returns
        index_returns = prices.pct_change().fillna(0)

        # Compute positions
        positions = compute_positions(regime_probs, confidence, prices, self.cfg)

        # Strategy returns (positions are set at close, applied next day)
        strategy_returns = (positions.shift(1) * index_returns)

        # Apply transaction costs
        strategy_returns = apply_transaction_costs(
            strategy_returns,
            positions,
            cost_bps=self.cfg["strategy"]["transaction_cost_bps"],
            slippage_bps=self.cfg["backtest"]["slippage_bps"],
        )

        # Compile results
        self.results[label] = {
            "prices": prices,
            "positions": positions,
            "strategy_returns": strategy_returns,
            "benchmark_returns": index_returns,
            "regime_probs": regime_probs,
            "confidence": confidence,
        }

        # Cumulative wealth
        strategy_wealth = (1 + strategy_returns).cumprod()
        benchmark_wealth = (1 + index_returns).cumprod()

        logger.info(f"Backtest complete: {len(strategy_returns)} days")
        logger.info(f"  Start: {strategy_returns.index[0].date()} | End: {strategy_returns.index[-1].date()}")

        return strategy_returns

    def performance_report(self, label: str = None) -> None:
        """Print comprehensive performance report."""
        if not self.results:
            logger.warning("No backtest results. Run .run() first.")
            return

        key = label or list(self.results.keys())[0]
        res = self.results[key]

        strat_ret = res["strategy_returns"]
        bench_ret = res["benchmark_returns"]
        regime_labels = res["regime_probs"].idxmax(axis=1)

        rf = self.cfg["performance"]["risk_free_rate"]
        ann = self.cfg["performance"]["annualisation_factor"]

        print("\n" + "="*60)
        print(f"  PERFORMANCE REPORT: {key}")
        print("="*60)

        summary = performance_summary(strat_ret, bench_ret, rf, ann, key)
        print(summary.to_string())

        print("\n" + "-"*60)
        print("  PERFORMANCE BY REGIME")
        print("-"*60)
        breakdown = regime_performance_breakdown(strat_ret, regime_labels, ann)
        print(breakdown.to_string())

        # Regime time allocation
        print("\n" + "-"*60)
        print("  REGIME TIME ALLOCATION")
        print("-"*60)
        print(regime_labels.value_counts(normalize=True).map("{:.1%}".format))

    def plot_results(self, label: str = None, config: dict = None, save: bool = True) -> None:
        """Four-panel results plot."""
        if not self.results:
            return

        key = label or list(self.results.keys())[0]
        res = self.results[key]
        cfg = config or self.cfg

        strat_ret = res["strategy_returns"]
        bench_ret = res["benchmark_returns"]
        positions = res["positions"]
        confidence = res["confidence"]
        regime_probs = res["regime_probs"]

        fig, axes = plt.subplots(4, 1, figsize=(16, 18), sharex=True)
        fig.suptitle(f"JSE Top 40 Regime Strategy — Backtest Results", fontsize=14, fontweight="bold")

        # Panel 1: Cumulative returns
        ax1 = axes[0]
        strat_cum = (1 + strat_ret).cumprod()
        bench_cum = (1 + bench_ret).cumprod()
        ax1.plot(strat_cum.index, strat_cum.values, label=key, color="#2980b9", lw=1.5)
        ax1.plot(bench_cum.index, bench_cum.values, label="JSE Top 40 (B&H)", color="#7f8c8d", lw=1.2, ls="--")
        ax1.set_ylabel("Cumulative Wealth (R1 = 1.0)", fontsize=10)
        ax1.set_title("Cumulative Returns", fontsize=10)
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale("log")

        # Panel 2: Drawdown
        ax2 = axes[1]
        strat_dd = drawdown_series(strat_ret)
        bench_dd = drawdown_series(bench_ret)
        ax2.fill_between(strat_dd.index, strat_dd.values * 100, 0, alpha=0.6, color="#2980b9", label=key)
        ax2.fill_between(bench_dd.index, bench_dd.values * 100, 0, alpha=0.3, color="#7f8c8d", label="Benchmark")
        ax2.set_ylabel("Drawdown (%)", fontsize=10)
        ax2.set_title("Drawdown", fontsize=10)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)

        # Panel 3: Position sizing
        ax3 = axes[2]
        ax3.fill_between(positions.index, positions.values * 100, 0, alpha=0.6, color="#27ae60")
        ax3.set_ylabel("Position Size (%)", fontsize=10)
        ax3.set_title("Position Sizing (Regime + Confidence Adjusted)", fontsize=10)
        ax3.set_ylim(0, 110)
        ax3.grid(True, alpha=0.3)

        # Panel 4: Regime probabilities
        ax4 = axes[3]
        colours = ["#2ecc71", "#e74c3c", "#f39c12"]
        for col, colour in zip(regime_probs.columns, colours):
            ax4.plot(regime_probs.index, regime_probs[col], label=col, color=colour, lw=1.0, alpha=0.8)
        ax4.plot(confidence.index, confidence.values, label="Confidence", color="#8e44ad", lw=1.5, ls="--")
        ax4.set_ylabel("Probability", fontsize=10)
        ax4.set_title("Regime Probabilities + Position Confidence", fontsize=10)
        ax4.legend(fontsize=9, loc="upper left")
        ax4.set_ylim(0, 1)
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save and cfg:
            fig_path = Path(cfg["paths"]["figures"])
            fig_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(fig_path / "backtest_results.png", dpi=150, bbox_inches="tight")
            logger.info("Backtest chart saved.")

        plt.show()


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data.loader import get_price_data
    from src.data.features import build_features, load_features

    config = load_config()

    # Load price data
    prices = get_price_data(config)

    # Load regime probabilities (output from bayesian_regime.py or markov_switching.py)
    processed = Path(config["paths"]["processed_data"])

    try:
        regime_probs = pd.read_csv(processed / "regime_probs_bayesian.csv", index_col="date", parse_dates=True)
        confidence = pd.read_csv(processed / "position_confidence.csv", index_col="date", parse_dates=True).squeeze()
        logger.info("Using Bayesian regime probabilities.")
    except FileNotFoundError:
        logger.warning("Bayesian results not found. Using classical model output.")
        regime_probs = pd.read_csv(processed / "regime_probs_classical.csv", index_col="date", parse_dates=True)
        confidence = pd.Series(1.0, index=regime_probs.index, name="confidence")

    # Run backtest
    backtest = RegimeBacktest(config)
    backtest.run(prices["jse_top40"], regime_probs, confidence)
    backtest.performance_report()
    backtest.plot_results(config=config)

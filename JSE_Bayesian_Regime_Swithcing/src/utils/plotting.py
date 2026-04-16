"""
src/utils/plotting.py
---------------------
Standardised visualisation library for the JSE Regime Strategy.

All functions follow a consistent style:
- Dark grid background, clean axis spines
- Colour palette: green (Risk-On), red (Risk-Off), orange (Stagflation)
- Save to results/figures/ when save=True
- Return the figure object for further customisation

Usage:
    from src.utils.plotting import RegimePlotter
    plotter = RegimePlotter(config)
    plotter.regime_overview(prices, regime_probs)
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import seaborn as sns
import yaml

logger = logging.getLogger(__name__)

# ── Style Constants ───────────────────────────────────────────────────────────

PALETTE = {
    "Risk-On":     "#2ecc71",   # Green
    "Risk-Off":    "#e74c3c",   # Red
    "Stagflation": "#f39c12",   # Orange
    "strategy":    "#2980b9",   # Blue
    "benchmark":   "#7f8c8d",   # Grey
    "confidence":  "#8e44ad",   # Purple
    "neutral":     "#2c3e50",   # Dark navy
}

# Known SA market events for annotation
SA_EVENTS = {
    "GFC":         "2008-09-15",
    "Nenegate":    "2015-12-09",
    "Zuma recall": "2018-02-14",
    "COVID":       "2020-03-23",
    "Fed pivot":   "2022-03-16",
}


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _set_style():
    """Apply consistent matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "#f8f9fa",
        "axes.grid":         True,
        "grid.color":        "#dee2e6",
        "grid.linewidth":    0.6,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "font.family":       "DejaVu Sans",
        "font.size":         10,
    })


def _save_fig(fig: plt.Figure, filename: str, config: dict) -> None:
    fig_path = Path(config["paths"]["figures"])
    fig_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path / filename, dpi=150, bbox_inches="tight", facecolor="white")
    logger.info(f"Figure saved: {fig_path / filename}")


def _annotate_events(ax: plt.Axes, y_frac: float = 0.97, date_range: tuple = None) -> None:
    """Add vertical lines for known SA market events within the chart's date range."""
    for label, date_str in SA_EVENTS.items():
        dt = pd.Timestamp(date_str)
        if date_range:
            if not (date_range[0] <= dt <= date_range[1]):
                continue
        ax.axvline(dt, color="#636e72", ls="--", lw=0.8, alpha=0.7)
        ax.text(dt, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0]) * y_frac,
                label, fontsize=7, rotation=90, va="top", ha="right", color="#636e72")


def _shade_regimes(ax: plt.Axes, regime_series: pd.Series, alpha: float = 0.12) -> None:
    """Shade background of axis by regime classification."""
    in_regime = False
    current = None
    start = None
    for date, regime in regime_series.items():
        colour = PALETTE.get(regime, "#bdc3c7")
        if regime != current:
            if in_regime and start is not None:
                ax.axvspan(start, date, alpha=alpha, color=PALETTE.get(current, "#bdc3c7"), lw=0)
            current = regime
            start = date
            in_regime = True
    if in_regime and start is not None:
        ax.axvspan(start, regime_series.index[-1], alpha=alpha,
                   color=PALETTE.get(current, "#bdc3c7"), lw=0)


# ── Main Plotter Class ────────────────────────────────────────────────────────

class RegimePlotter:
    """Centralised plotting interface for the JSE regime strategy."""

    def __init__(self, config: dict):
        self.cfg = config
        _set_style()

    def regime_overview(
        self,
        prices: pd.Series,
        regime_probs: pd.DataFrame,
        confidence: pd.Series = None,
        save: bool = True,
        filename: str = "01_regime_overview.png",
    ) -> plt.Figure:
        """
        Four-panel overview:
        1. JSE Top 40 price level with regime shading
        2. Regime probabilities
        3. Regime classification bar
        4. Confidence score (if provided)
        """
        n_panels = 4 if confidence is not None else 3
        fig, axes = plt.subplots(n_panels, 1, figsize=(16, 4 * n_panels), sharex=True)
        fig.suptitle("JSE Top 40 — Bayesian Regime Classification Overview",
                     fontsize=14, fontweight="bold", y=1.01)

        regimes = regime_probs.idxmax(axis=1)
        date_range = (regime_probs.index[0], regime_probs.index[-1])
        colours = list(PALETTE.values())[:3]

        # Panel 1: Price level
        ax1 = axes[0]
        prices_aligned = prices.reindex(regime_probs.index)
        ax1.plot(prices_aligned.index, prices_aligned.values,
                 color=PALETTE["neutral"], lw=1.3, label="JSE Top 40")
        _shade_regimes(ax1, regimes)
        _annotate_events(ax1, date_range=date_range)
        ax1.set_ylabel("Index Level", fontsize=10)
        ax1.set_title("JSE Top 40 Price Level", fontsize=10, fontweight="bold")

        # Legend patches for regimes
        patches = [mpatches.Patch(color=PALETTE[r], label=r, alpha=0.6)
                   for r in regime_probs.columns if r in PALETTE]
        ax1.legend(handles=patches, loc="upper left", fontsize=9)

        # Panel 2: Regime probabilities
        ax2 = axes[1]
        for col in regime_probs.columns:
            colour = PALETTE.get(col, "#bdc3c7")
            ax2.plot(regime_probs.index, regime_probs[col],
                     label=col, color=colour, lw=1.2, alpha=0.9)
        ax2.set_ylabel("Probability", fontsize=10)
        ax2.set_title("Posterior Regime Probabilities", fontsize=10, fontweight="bold")
        ax2.set_ylim(0, 1)
        ax2.legend(loc="upper left", fontsize=9)

        # Panel 3: Hard classification
        ax3 = axes[2]
        regime_numeric = regimes.map({v: k for k, v in {0: "Risk-On", 1: "Risk-Off", 2: "Stagflation"}.items()})
        for i, (label, colour) in enumerate(zip(["Risk-On", "Risk-Off", "Stagflation"], colours)):
            mask = regimes == label
            if mask.any():
                ax3.fill_between(regimes.index, 0, 1, where=mask.values,
                                 color=colour, alpha=0.7, step="post", label=label)
        ax3.set_yticks([])
        ax3.set_ylabel("Regime", fontsize=10)
        ax3.set_title("Hard Regime Classification", fontsize=10, fontweight="bold")
        ax3.legend(loc="upper left", fontsize=9)

        # Panel 4: Confidence (optional)
        if confidence is not None and n_panels == 4:
            ax4 = axes[3]
            conf_aligned = confidence.reindex(regime_probs.index).fillna(0.5)
            ax4.fill_between(conf_aligned.index, 0, conf_aligned.values,
                             color=PALETTE["confidence"], alpha=0.5)
            ax4.plot(conf_aligned.index, conf_aligned.values,
                     color=PALETTE["confidence"], lw=1.2)
            ax4.axhline(self.cfg["strategy"]["min_regime_confidence"],
                        color="red", ls="--", lw=0.8, alpha=0.7, label="Min threshold")
            ax4.set_ylabel("Confidence", fontsize=10)
            ax4.set_title("Bayesian Posterior Confidence (1 − Normalised Entropy)",
                          fontsize=10, fontweight="bold")
            ax4.set_ylim(0, 1)
            ax4.legend(loc="lower left", fontsize=9)

        # X-axis formatting
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
        plt.xticks(rotation=45)

        plt.tight_layout()
        if save:
            _save_fig(fig, filename, self.cfg)
        return fig

    def backtest_results(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        positions: pd.Series,
        regime_probs: pd.DataFrame,
        confidence: pd.Series = None,
        save: bool = True,
        filename: str = "02_backtest_results.png",
    ) -> plt.Figure:
        """
        Four-panel backtest results:
        1. Cumulative returns (log scale)
        2. Underwater / drawdown chart
        3. Position sizing over time
        4. Regime probabilities
        """
        from src.utils.metrics import drawdown_series as dd_series

        fig = plt.figure(figsize=(16, 18))
        gs = gridspec.GridSpec(4, 1, hspace=0.35)
        axes = [fig.add_subplot(gs[i]) for i in range(4)]
        fig.suptitle("JSE Top 40 Regime Strategy — Backtest Results",
                     fontsize=14, fontweight="bold")

        # Align
        common = strategy_returns.index.intersection(benchmark_returns.index)
        sr = strategy_returns.reindex(common)
        br = benchmark_returns.reindex(common)
        pos = positions.reindex(common).fillna(0)

        strat_cum = (1 + sr).cumprod()
        bench_cum = (1 + br).cumprod()

        # Panel 1: Cumulative returns
        ax1 = axes[0]
        ax1.plot(strat_cum.index, strat_cum.values,
                 color=PALETTE["strategy"], lw=1.8, label="Regime Strategy")
        ax1.plot(bench_cum.index, bench_cum.values,
                 color=PALETTE["benchmark"], lw=1.3, ls="--", label="JSE Top 40 (B&H)", alpha=0.8)
        ax1.set_yscale("log")
        ax1.set_ylabel("Cumulative Return (log scale)", fontsize=10)
        ax1.set_title("Cumulative Wealth (R1 invested)", fontsize=10, fontweight="bold")
        ax1.legend(fontsize=10)
        _annotate_events(ax1, date_range=(common[0], common[-1]))

        # Panel 2: Drawdown
        ax2 = axes[1]
        strat_dd = dd_series(sr) * 100
        bench_dd = dd_series(br) * 100
        ax2.fill_between(strat_dd.index, strat_dd.values, 0,
                         color=PALETTE["strategy"], alpha=0.5, label="Strategy")
        ax2.fill_between(bench_dd.index, bench_dd.values, 0,
                         color=PALETTE["benchmark"], alpha=0.3, label="Benchmark")
        ax2.set_ylabel("Drawdown (%)", fontsize=10)
        ax2.set_title("Underwater Chart", fontsize=10, fontweight="bold")
        ax2.legend(fontsize=9)

        # Panel 3: Position sizing
        ax3 = axes[2]
        regime_aligned = regime_probs.reindex(common).idxmax(axis=1) if regime_probs is not None else None
        if regime_aligned is not None:
            _shade_regimes(ax3, regime_aligned, alpha=0.15)
        ax3.fill_between(pos.index, pos.values * 100, 0,
                         color="#27ae60", alpha=0.6, label="Position %")
        ax3.set_ylabel("Allocation (%)", fontsize=10)
        ax3.set_title("Dynamic Position Sizing (Regime + Confidence Weighted)",
                      fontsize=10, fontweight="bold")
        ax3.set_ylim(0, 110)
        ax3.legend(fontsize=9)

        # Panel 4: Regime probabilities
        ax4 = axes[3]
        rp = regime_probs.reindex(common) if regime_probs is not None else None
        if rp is not None:
            for col in rp.columns:
                ax4.plot(rp.index, rp[col], label=col,
                         color=PALETTE.get(col, "#bdc3c7"), lw=1.1, alpha=0.85)
        if confidence is not None:
            conf_aligned = confidence.reindex(common).fillna(0.5)
            ax4.plot(conf_aligned.index, conf_aligned.values,
                     color=PALETTE["confidence"], lw=1.5, ls="--", label="Confidence", alpha=0.9)
        ax4.set_ylabel("Probability", fontsize=10)
        ax4.set_title("Regime Probabilities + Confidence", fontsize=10, fontweight="bold")
        ax4.set_ylim(0, 1)
        ax4.legend(fontsize=9, loc="upper left")

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
        plt.xticks(rotation=45)

        if save:
            _save_fig(fig, filename, self.cfg)
        return fig

    def regime_return_distributions(
        self,
        returns: pd.Series,
        regime_probs: pd.DataFrame,
        annualisation: int = 252,
        save: bool = True,
        filename: str = "03_regime_distributions.png",
    ) -> plt.Figure:
        """Violin + box plot of returns per regime."""
        regimes = regime_probs.idxmax(axis=1)
        common = returns.index.intersection(regimes.index)
        df = pd.DataFrame({
            "return_ann": returns.reindex(common) * annualisation,
            "regime": regimes.reindex(common),
        }).dropna()

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle("Return Distributions by Regime", fontsize=13, fontweight="bold")

        order = [r for r in ["Risk-On", "Risk-Off", "Stagflation"] if r in df["regime"].unique()]
        palette = {r: PALETTE.get(r, "#bdc3c7") for r in order}

        # Violin
        sns.violinplot(data=df, x="regime", y="return_ann", order=order,
                       palette=palette, ax=axes[0], inner="box", alpha=0.75)
        axes[0].set_title("Annualised Return Distribution", fontsize=11, fontweight="bold")
        axes[0].set_xlabel("")
        axes[0].set_ylabel("Annualised Return", fontsize=10)
        axes[0].axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)

        # Summary stats
        stats = df.groupby("regime")["return_ann"].agg(["mean", "std", "skew", "count"])
        stats.columns = ["Mean", "Std Dev", "Skew", "N"]
        stats = stats.reindex(order)
        axes[1].axis("off")
        table = axes[1].table(
            cellText=stats.round(4).values,
            rowLabels=stats.index,
            colLabels=stats.columns,
            cellLoc="center",
            loc="center",
            bbox=[0, 0, 1, 1],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        for (row, col), cell in table.get_celld().items():
            if row == 0 or col == -1:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
        axes[1].set_title("Summary Statistics by Regime", fontsize=11, fontweight="bold")

        plt.tight_layout()
        if save:
            _save_fig(fig, filename, self.cfg)
        return fig

    def feature_correlation_heatmap(
        self,
        features: pd.DataFrame,
        save: bool = True,
        filename: str = "04_feature_correlations.png",
    ) -> plt.Figure:
        """Correlation heatmap of model features."""
        z_cols = [c for c in features.columns if c.endswith("_z")]
        corr = features[z_cols].corr()

        fig, ax = plt.subplots(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
            center=0, vmin=-1, vmax=1, ax=ax,
            annot_kws={"size": 8}, linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title("Feature Correlation Matrix (Z-scored Inputs)", fontsize=13, fontweight="bold")
        plt.tight_layout()
        if save:
            _save_fig(fig, filename, self.cfg)
        return fig

    def rolling_performance(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 252,
        save: bool = True,
        filename: str = "05_rolling_performance.png",
    ) -> plt.Figure:
        """Rolling Sharpe ratio and active return vs benchmark."""
        from src.utils.metrics import sharpe_ratio

        common = strategy_returns.index.intersection(benchmark_returns.index)
        sr = strategy_returns.reindex(common)
        br = benchmark_returns.reindex(common)
        active = sr - br

        rolling_sharpe_strat = sr.rolling(window).apply(
            lambda x: sharpe_ratio(pd.Series(x), adjust_autocorr=False), raw=False
        )
        rolling_sharpe_bench = br.rolling(window).apply(
            lambda x: sharpe_ratio(pd.Series(x), adjust_autocorr=False), raw=False
        )
        rolling_active_return = active.rolling(window).mean() * 252

        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        fig.suptitle(f"Rolling {window}-Day Performance", fontsize=13, fontweight="bold")

        # Rolling Sharpe
        ax1 = axes[0]
        ax1.plot(rolling_sharpe_strat.index, rolling_sharpe_strat.values,
                 color=PALETTE["strategy"], lw=1.5, label="Strategy Sharpe")
        ax1.plot(rolling_sharpe_bench.index, rolling_sharpe_bench.values,
                 color=PALETTE["benchmark"], lw=1.2, ls="--", label="Benchmark Sharpe")
        ax1.axhline(0, color="black", lw=0.8, alpha=0.5)
        ax1.axhline(1, color="green", lw=0.8, ls=":", alpha=0.5, label="Sharpe = 1")
        ax1.set_ylabel("Sharpe Ratio", fontsize=10)
        ax1.set_title(f"Rolling {window}-Day Sharpe Ratio", fontsize=10, fontweight="bold")
        ax1.legend(fontsize=9)

        # Rolling active return
        ax2 = axes[1]
        ax2.fill_between(rolling_active_return.index, rolling_active_return.values, 0,
                         where=rolling_active_return.values >= 0,
                         color=PALETTE["Risk-On"], alpha=0.5, label="Outperforming")
        ax2.fill_between(rolling_active_return.index, rolling_active_return.values, 0,
                         where=rolling_active_return.values < 0,
                         color=PALETTE["Risk-Off"], alpha=0.5, label="Underperforming")
        ax2.axhline(0, color="black", lw=1)
        ax2.set_ylabel("Active Return (ann.)", fontsize=10)
        ax2.set_title(f"Rolling {window}-Day Active Return vs Benchmark", fontsize=10, fontweight="bold")
        ax2.legend(fontsize=9)

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axes[-1].xaxis.set_major_locator(mdates.YearLocator(2))
        plt.xticks(rotation=45)

        plt.tight_layout()
        if save:
            _save_fig(fig, filename, self.cfg)
        return fig

    def plot_all(
        self,
        prices: pd.Series,
        regime_probs: pd.DataFrame,
        confidence: pd.Series,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        positions: pd.Series,
        features: pd.DataFrame = None,
    ) -> None:
        """Generate the full suite of charts in one call."""
        logger.info("Generating full chart suite...")
        self.regime_overview(prices, regime_probs, confidence)
        self.backtest_results(strategy_returns, benchmark_returns, positions, regime_probs, confidence)
        self.regime_return_distributions(benchmark_returns, regime_probs)
        self.rolling_performance(strategy_returns, benchmark_returns)
        if features is not None:
            self.feature_correlation_heatmap(features)
        logger.info("All charts generated.")

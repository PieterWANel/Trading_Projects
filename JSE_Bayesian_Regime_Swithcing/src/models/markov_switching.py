"""
src/models/markov_switching.py
------------------------------
Classical Hamilton (1989) Markov-switching model baseline.
Estimated via EM algorithm using statsmodels.

This serves as:
1. Interpretable baseline for the Bayesian MCMC model
2. Fast regime labelling for EDA
3. Benchmark to demonstrate improvement from Bayesian approach

Usage:
    python -m src.models.markov_switching
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression
import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Model Fitting ─────────────────────────────────────────────────────────────

class HamiltonMarkovSwitching:
    """
    Wraps statsmodels MarkovRegression for regime detection on JSE returns.
    
    Fits a k-regime model where each regime has its own:
    - Mean return
    - Variance (switching variance)
    
    The transition probability matrix P[i,j] = P(regime_t=j | regime_{t-1}=i)
    is estimated from data.
    """

    def __init__(self, n_regimes: int = 3, config: dict = None):
        self.k = n_regimes
        self.cfg = config or load_config()
        self.model = None
        self.result = None
        self.regime_labels = self.cfg["regimes"]["labels"]

    def fit(self, returns: pd.Series, switching_variance: bool = True) -> "HamiltonMarkovSwitching":
        """
        Fit the Markov-switching model.
        
        Args:
            returns:            Daily log returns of JSE Top 40
            switching_variance: If True, each regime has its own variance (recommended)
        """
        logger.info(f"Fitting {self.k}-regime Markov-switching model...")
        logger.info(f"  Data: {len(returns)} observations | {returns.index[0].date()} → {returns.index[-1].date()}")

        self.returns = returns.dropna()

        self.model = MarkovRegression(
            endog=self.returns,
            k_regimes=self.k,
            trend="c",                          # Constant mean per regime
            switching_variance=switching_variance,
        )

        self.result = self.model.fit(
            search_reps=20,     # Multiple starting points for EM
            maxiter=1000,
            disp=False,
        )

        logger.info("Model fitted successfully.")
        self._log_summary()
        return self

    def _log_summary(self) -> None:
        """Log key model parameters."""
        res = self.result
        logger.info(f"\n{'='*50}")
        logger.info(f"  Log-likelihood: {res.llf:.2f}")
        logger.info(f"  AIC: {res.aic:.2f} | BIC: {res.bic:.2f}")
        logger.info(f"\n  Regime parameters:")
        for k in range(self.k):
            label = self.regime_labels.get(k, f"Regime {k}")
            mu = res.params[f"const[{k}]"]
            sigma = np.sqrt(res.params[f"sigma2[{k}]"]) * np.sqrt(252)
            mu_ann = mu * 252
            logger.info(f"    {label}: μ={mu_ann:.1%} ann | σ={sigma:.1%} ann")
        logger.info(f"\n  Transition matrix:")
        logger.info(f"  Transition probs (p_ii): {[round(float(res.params[i]), 3) for i in range(self.k)]}")
    
    @property
    def smoothed_probs(self) -> pd.DataFrame:
        raw = self.result.smoothed_marginal_probabilities
        if hasattr(raw, 'values'):
            data = raw.values
        else:
            data = np.array(raw)
        cols = [self.regime_labels.get(i, f"Regime_{i}") for i in range(self.k)]
        return pd.DataFrame(data, index=self.returns.index, columns=cols)

    @property
    def filtered_probs(self) -> pd.DataFrame:
        raw = self.result.filtered_marginal_probabilities
        if hasattr(raw, 'values'):
            data = raw.values
        else:
            data = np.array(raw)
        cols = [self.regime_labels.get(i, f"Regime_{i}") for i in range(self.k)]
        return pd.DataFrame(data, index=self.returns.index, columns=cols)

    @property
    def regime_classification(self) -> pd.Series:
        """Hard regime classification: argmax of smoothed probabilities."""
        probs = self.smoothed_probs
        classified = probs.idxmax(axis=1)
        classified.name = "regime"
        return classified

    def expected_durations(self) -> pd.Series:
        """Expected duration in each regime (in trading days)."""
        trans = self.result.params[:self.k * self.k].values.reshape(self.k, self.k)
        durations = {}
        for i in range(self.k):
            p_stay = trans[i, i]
            durations[self.regime_labels.get(i, f"Regime_{i}")] = 1 / (1 - p_stay)
        return pd.Series(durations, name="Expected Duration (days)")

    def save_results(self, config: dict) -> None:
        """Save regime probabilities and classifications to processed data."""
        out_path = Path(config["paths"]["processed_data"])
        out_path.mkdir(parents=True, exist_ok=True)

        self.smoothed_probs.to_csv(out_path / "regime_probs_classical.csv")
        self.regime_classification.to_csv(out_path / "regime_classification_classical.csv")
        logger.info(f"Results saved to {out_path}")


# ── Visualisation ─────────────────────────────────────────────────────────────

def plot_regime_probabilities(
    model: HamiltonMarkovSwitching,
    prices: pd.Series,
    config: dict,
    save: bool = True,
) -> None:
    """
    Three-panel plot:
    1. JSE Top 40 index level
    2. Smoothed regime probabilities
    3. Hard regime classification (coloured background)
    """
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.suptitle("JSE Top 40 — Hamilton Markov-Switching Regime Model", fontsize=14, fontweight="bold")

    probs = model.smoothed_probs
    regimes = model.regime_classification
    colours = ["#2ecc71", "#e74c3c", "#f39c12"]  # Green, Red, Orange

    # Panel 1: Index level
    ax1 = axes[0]
    prices_aligned = prices.reindex(probs.index)
    ax1.plot(prices_aligned.index, prices_aligned.values, color="#2c3e50", lw=1.2)
    ax1.set_ylabel("JSE Top 40 Level", fontsize=10)
    ax1.set_title("Index Level", fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Shade regimes on panel 1
    for i, (label, colour) in enumerate(zip(probs.columns, colours)):
        regime_mask = regimes == label
        _shade_regimes(ax1, regime_mask, colour, alpha=0.15)

    # Panel 2: Regime probabilities
    ax2 = axes[1]
    for col, colour in zip(probs.columns, colours):
        ax2.plot(probs.index, probs[col], label=col, color=colour, lw=1.2)
    ax2.set_ylabel("Probability", fontsize=10)
    ax2.set_title("Smoothed Regime Probabilities", fontsize=10)
    ax2.legend(loc="upper left", fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Hard regime classification
    ax3 = axes[2]
    regime_numeric = regimes.map({v: k for k, v in model.regime_labels.items()})
    ax3.fill_between(regime_numeric.index, regime_numeric.values, step="post", alpha=0.7, color="#3498db")
    ax3.set_yticks(list(model.regime_labels.keys()))
    ax3.set_yticklabels(list(model.regime_labels.values()), fontsize=9)
    ax3.set_ylabel("Regime", fontsize=10)
    ax3.set_title("Hard Regime Classification", fontsize=10)
    ax3.grid(True, alpha=0.3)

    # Format x-axis
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax3.xaxis.set_major_locator(mdates.YearLocator(2))
    plt.xticks(rotation=45)

    # Annotate known SA events
    events = {
        "GFC": "2008-09-15",
        "Nenegate": "2015-12-09",
        "COVID": "2020-03-23",
        "Fed Pivot": "2022-03-16",
    }
    for event, date in events.items():
        try:
            dt = pd.Timestamp(date)
            if dt in probs.index or (probs.index[0] <= dt <= probs.index[-1]):
                ax1.axvline(dt, color="gray", ls="--", lw=0.8, alpha=0.6)
                ax1.text(dt, ax1.get_ylim()[1] * 0.95, event, fontsize=7, rotation=90,
                         va="top", ha="right", color="gray")
        except Exception:
            pass

    plt.tight_layout()

    if save:
        fig_path = Path(config["paths"]["figures"])
        fig_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path / "regime_classical.png", dpi=150, bbox_inches="tight")
        logger.info(f"Figure saved to {fig_path / 'regime_classical.png'}")

    plt.show()


def _shade_regimes(ax, mask: pd.Series, colour: str, alpha: float = 0.2) -> None:
    """Shade background of axis where mask is True."""
    in_regime = False
    start = None
    for date, val in mask.items():
        if val and not in_regime:
            start = date
            in_regime = True
        elif not val and in_regime:
            ax.axvspan(start, date, alpha=alpha, color=colour, lw=0)
            in_regime = False
    if in_regime:
        ax.axvspan(start, mask.index[-1], alpha=alpha, color=colour, lw=0)


def plot_regime_statistics(model: HamiltonMarkovSwitching, returns: pd.Series, config: dict) -> None:
    """Distribution of returns per regime."""
    fig, axes = plt.subplots(1, model.k, figsize=(14, 5), sharey=False)
    fig.suptitle("Return Distribution by Regime", fontsize=13, fontweight="bold")

    colours = ["#2ecc71", "#e74c3c", "#f39c12"]
    regimes = model.regime_classification

    for i, (label, ax, colour) in enumerate(zip(model.regime_labels.values(), axes, colours)):
        mask = regimes == label
        regime_returns = returns[mask] * 252  # Annualised

        sns.histplot(regime_returns, ax=ax, bins=50, color=colour, alpha=0.7, kde=True)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_xlabel("Annualised Return", fontsize=9)

        stats_text = (
            f"n = {mask.sum()}\n"
            f"μ = {regime_returns.mean():.1%}\n"
            f"σ = {regime_returns.std():.1%}\n"
            f"Skew = {regime_returns.skew():.2f}"
        )
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
                fontsize=9, va="top", bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig_path = Path(config["paths"]["figures"])
    fig_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path / "regime_return_distributions.png", dpi=150, bbox_inches="tight")
    plt.show()


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data.loader import get_price_data
    from src.data.features import build_features

    config = load_config()

    # Load data
    prices = get_price_data(config)
    returns = np.log(prices["jse_top40"] / prices["jse_top40"].shift(1)).dropna()

    # Fit model
    model = HamiltonMarkovSwitching(n_regimes=config["regimes"]["n_regimes"], config=config)
    model.fit(returns)

    # Print regime summary
    print("\nRegime Classification Summary:")
    print(model.regime_classification.value_counts())
    print("\nExpected Regime Durations:")
    print(model.expected_durations())

    # Save and plot
    model.save_results(config)
    plot_regime_probabilities(model, prices["jse_top40"], config)
    plot_regime_statistics(model, returns, config)

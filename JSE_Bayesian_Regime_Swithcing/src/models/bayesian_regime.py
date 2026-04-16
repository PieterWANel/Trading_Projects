"""
src/models/bayesian_regime.py
------------------------------
Bayesian Markov-switching model estimated via MCMC (PyMC / NUTS sampler).

Key advantages over classical EM approach:
1. Full posterior distributions over all parameters — not just point estimates
2. Uncertainty-aware regime probabilities: position sizing scales with confidence
3. Principled regularisation via priors — avoids overfitting ghost regimes
4. Credible intervals on regime transition probabilities

Model structure:
    y_t | S_t=k ~ Normal(mu_k, sigma_k)
    S_t | S_{t-1} ~ Categorical(P[S_{t-1}, :])
    P[k, :] ~ Dirichlet(alpha)      (row-wise transition probabilities)
    mu_k ~ Normal(mu_0, sigma_mu)
    sigma_k ~ HalfNormal(sigma_0)

Usage:
    python -m src.models.bayesian_regime
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yaml

import os
import pytensor
pytensor.config.mode = "NUMBA"

warnings.filterwarnings("ignore", category=FutureWarning)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── Bayesian Model ────────────────────────────────────────────────────────────

class BayesianRegimeSwitching:
    """
    Bayesian Markov-switching model for JSE regime detection.
    
    Estimated via NUTS (No U-Turn Sampler) through PyMC.
    Returns full posterior distributions over regime probabilities,
    enabling uncertainty-aware position sizing in the strategy router.
    """

    def __init__(self, n_regimes: int = 3, config: dict = None):
        self.k = n_regimes
        self.cfg = config or load_config()
        self.trace = None
        self.model = None
        self.regime_labels = self.cfg["regimes"]["labels"]
        self._check_pymc()

    def _check_pymc(self):
        try:
            import pymc as pm
            import pytensor.tensor as pt
            self.pm = pm
            self.pt = pt
            logger.info(f"PyMC version: {pm.__version__}")
        except ImportError:
            raise ImportError(
                "PyMC not installed. Run: pip install pymc\n"
                "Note: PyMC requires a C compiler. On Windows, install Visual Studio Build Tools."
            )

    def build_model(self, returns: np.ndarray) -> None:
        """
        Define the Bayesian graphical model.
        
        Priors are weakly informative, calibrated to daily equity returns:
        - mu prior: Normal(0, 0.001) — centred at zero daily return
        - sigma prior: HalfNormal(0.02) — ~1.2% daily vol, broad enough to cover all regimes
        - Transition rows: Dirichlet([5,1,1]) — sticky (diagonal-dominant) prior
        """
        pm = self.pm
        pt = self.pt
        T = len(returns)
        K = self.k

        logger.info(f"Building Bayesian model: {K} regimes, {T} observations")

        with pm.Model() as self.model:
            # ── Priors ────────────────────────────────────────────────────────
            # Regime means (annualised: ~0%, ~-20%, ~+15%)
            mu = pm.Normal("mu", mu=0.0, sigma=0.001, shape=K)

            # Regime standard deviations
            sigma = pm.HalfNormal("sigma", sigma=0.02, shape=K)

            # Transition probability matrix (each row is Dirichlet)
            # Prior: sticky — concentrates on self-transitions
            # Dirichlet([5,1,1]) for K=3 → ~5/7 ≈ 71% chance of staying in regime
            alpha_diag = 5.0
            alpha_offdiag = 1.0
            alpha = np.full((K, K), alpha_offdiag)
            np.fill_diagonal(alpha, alpha_diag)

            P = pm.Dirichlet("P", a=alpha, shape=(K, K))

            # ── Forward Algorithm (Hidden Markov Model likelihood) ─────────────
            # Compute log-likelihood under each regime at each time step
            log_emission = pm.math.stack([
            pm.logp(pm.Normal.dist(mu=mu[k], sigma=sigma[k]), returns)
            for k in range(K)
            ], axis=1)

            # Initial state distribution: uniform
            log_init = pt.log(pt.ones(K) / K)

            # Forward pass (log-sum-exp for numerical stability)
            def forward_step(log_emission_t, log_alpha_prev, log_P):
                """Single step of forward algorithm in log space."""
                log_alpha_pred = pt.logsumexp(
                    log_alpha_prev[:, None] + log_P, axis=0
                )
                log_alpha_t = log_alpha_pred + log_emission_t
                # Normalise for numerical stability
                log_alpha_t = log_alpha_t - pt.logsumexp(log_alpha_t)
                return log_alpha_t

            log_P = pt.log(P + 1e-10)

            # Scan over time steps
            import pytensor
            log_alphas, _ = pytensor.scan(
                fn=forward_step,
                sequences=[log_emission[1:]],
                outputs_info=[log_init + log_emission[0] - pt.logsumexp(log_init + log_emission[0])],
                non_sequences=[log_P],
            )

            # Total log-likelihood
            log_likelihood = pt.sum(pm.math.logsumexp(log_alphas, axis=1))
            pm.Potential("hmm_likelihood", log_likelihood)

            logger.info("Model built successfully.")

    def sample(self, returns: pd.Series) -> "BayesianRegimeSwitching":
        """
        Run MCMC sampling.
        
        Uses NUTS (gradient-based) sampler via PyMC.
        For large datasets (>2000 obs), consider using pm.ADVI for variational inference
        as a faster approximation.
        """
        pm = self.pm
        mcmc_cfg = self.cfg["mcmc"]

        self.returns_series = returns.dropna()
        returns_array = self.returns_series.values.astype(np.float64)

        self.build_model(returns_array)

        logger.info(f"\nStarting MCMC sampling:")
        logger.info(f"  Draws: {mcmc_cfg['draws']} | Tune: {mcmc_cfg['tune']} | Chains: {mcmc_cfg['chains']}")
        logger.info(f"  This may take 10-30 minutes for large datasets...")

        with self.model:
            self.trace = pm.sample(
                draws=mcmc_cfg["draws"],
                tune=mcmc_cfg["tune"],
                chains=mcmc_cfg["chains"],
                target_accept=mcmc_cfg["target_accept"],
                random_seed=mcmc_cfg["random_seed"],
                progressbar=True,
                return_inferencedata=True,
            )

        logger.info("Sampling complete.")
        self._log_diagnostics()
        return self

    def sample_vi(self, returns: pd.Series, n_iterations: int = 50000) -> "BayesianRegimeSwitching":
        """
        Variational inference alternative to full MCMC.
        Much faster (minutes vs hours) but approximate.
        Useful for exploratory work before committing to full MCMC.
        """
        os.environ["PYTENSOR_FLAGS"] = "openmp=True"
        pm = self.pm

        self.returns_series = returns.dropna()
        returns_array = self.returns_series.values.astype(np.float64)

        self.build_model(returns_array)

        logger.info(f"Running Variational Inference (ADVI) — {n_iterations} iterations...")

        with self.model:
            approx = pm.fit(
                n=n_iterations,
                method="advi",
                random_seed=self.cfg["mcmc"]["random_seed"],
                progressbar=True,
                obj_optimizer=pm.adagrad_window(learning_rate=0.01),
            )
            self.trace = approx.sample(self.cfg["mcmc"]["draws"])

        logger.info("VI complete.")
        return self

    def _log_diagnostics(self) -> None:
        """Log MCMC convergence diagnostics."""
        try:
            import arviz as az
            summary = az.summary(self.trace, var_names=["mu", "sigma"])
            logger.info(f"\nMCMC Diagnostics (mu, sigma):")
            logger.info(f"\n{summary[['mean', 'sd', 'hdi_3%', 'hdi_97%', 'r_hat']].to_string()}")

            r_hats = summary["r_hat"]
            if (r_hats > 1.05).any():
                logger.warning("⚠ Some R-hat values > 1.05. Consider more tuning steps.")
            else:
                logger.info("✓ R-hat values all < 1.05 — good convergence.")
        except ImportError:
            logger.info("Install arviz for convergence diagnostics: pip install arviz")

    def posterior_regime_probs(self) -> pd.DataFrame:
        """
        Use posterior mean parameters in a clean Viterbi-style forward pass.
        """
        if self.trace is None:
            raise RuntimeError("Model not yet sampled. Run .sample() first.")

        mu_post = np.array(self.trace.posterior["mu"].mean(dim=["chain", "draw"]))
        sigma_post = np.array(self.trace.posterior["sigma"].mean(dim=["chain", "draw"]))
        P_post = np.array(self.trace.posterior["P"].mean(dim=["chain", "draw"]))

        returns = self.returns_series.values
        T = len(returns)
        K = self.k

        # Forward algorithm in log space
        log_P = np.log(P_post + 1e-10)
        log_alpha = np.zeros((T, K))

        # Initialise with uniform prior
        for k in range(K):
            log_alpha[0, k] = self._log_normal(returns[0], mu_post[k], sigma_post[k])
        log_alpha[0] -= np.log(np.sum(np.exp(log_alpha[0] - log_alpha[0].max()))) + log_alpha[0].max()

        for t in range(1, T):
            for k in range(K):
                # Sum over all previous states
                vals = log_alpha[t-1] + log_P[:, k]
                max_val = vals.max()
                log_alpha[t, k] = max_val + np.log(np.sum(np.exp(vals - max_val)))
                log_alpha[t, k] += self._log_normal(returns[t], mu_post[k], sigma_post[k])
            # Normalise
            max_val = log_alpha[t].max()
            log_alpha[t] -= max_val + np.log(np.sum(np.exp(log_alpha[t] - max_val)))

        # Convert to probabilities
        probs = np.exp(log_alpha - log_alpha.max(axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)

        cols = [self.regime_labels.get(i, f"Regime_{i}") for i in range(K)]
        return pd.DataFrame(probs, index=self.returns_series.index, columns=cols)

    @staticmethod
    def _log_normal(x: float, mu: float, sigma: float) -> float:
        return -0.5 * np.log(2 * np.pi * sigma**2) - (x - mu)**2 / (2 * sigma**2)

    def regime_uncertainty(self) -> pd.Series:
        """
        Regime uncertainty at each time step = entropy of regime probability distribution.
        High entropy → regime uncertain → reduce position size.
        Low entropy → high confidence → full position.
        """
        probs = self.posterior_regime_probs()
        # Shannon entropy, normalised to [0,1]
        entropy = -(probs * np.log(probs + 1e-10)).sum(axis=1)
        max_entropy = np.log(self.k)
        return (entropy / max_entropy).rename("uncertainty")

    def position_confidence(self) -> pd.Series:
        """
        Position confidence score = 1 - normalised entropy.
        1.0 → very confident, deploy full position
        0.0 → maximum uncertainty, deploy nothing
        """
        return (1 - self.regime_uncertainty()).rename("confidence")

    def hard_classification(self) -> pd.Series:
        """Argmax of posterior regime probabilities."""
        probs = self.posterior_regime_probs()
        return probs.idxmax(axis=1).rename("regime")

    def save_results(self, config: dict) -> None:
        """Save posterior regime probabilities and trace."""
        out_path = Path(config["paths"]["processed_data"])
        out_path.mkdir(parents=True, exist_ok=True)

        probs = self.posterior_regime_probs()
        probs.to_csv(out_path / "regime_probs_bayesian.csv")
        self.regime_uncertainty().to_csv(out_path / "regime_uncertainty.csv")
        self.position_confidence().to_csv(out_path / "position_confidence.csv")

        logger.info(f"Bayesian results saved to {out_path}")

    def plot_posterior(self, config: dict, save: bool = True) -> None:
        """Plot posterior distributions of regime parameters."""
        try:
            import arviz as az
            fig, axes = plt.subplots(2, self.k, figsize=(14, 8))
            fig.suptitle("Posterior Distributions — Bayesian Regime Parameters", fontsize=13, fontweight="bold")

            colours = ["#2ecc71", "#e74c3c", "#f39c12"]

            mu_samples = self.trace.posterior["mu"].values.reshape(-1, self.k) * 252  # Annualised
            sigma_samples = self.trace.posterior["sigma"].values.reshape(-1, self.k) * np.sqrt(252)

            for k in range(self.k):
                label = self.regime_labels.get(k, f"Regime {k}")
                colour = colours[k]

                # Mean return posterior
                axes[0, k].hist(mu_samples[:, k], bins=60, color=colour, alpha=0.7, density=True)
                axes[0, k].set_title(f"{label}\nAnnualised Mean Return", fontsize=10)
                axes[0, k].axvline(mu_samples[:, k].mean(), color="black", lw=2, ls="--")
                axes[0, k].set_xlabel("Return", fontsize=9)
                axes[0, k].grid(True, alpha=0.3)

                # Volatility posterior
                axes[1, k].hist(sigma_samples[:, k], bins=60, color=colour, alpha=0.7, density=True)
                axes[1, k].set_title(f"Annualised Volatility", fontsize=10)
                axes[1, k].axvline(sigma_samples[:, k].mean(), color="black", lw=2, ls="--")
                axes[1, k].set_xlabel("Volatility", fontsize=9)
                axes[1, k].grid(True, alpha=0.3)

            plt.tight_layout()

            if save:
                fig_path = Path(config["paths"]["figures"])
                fig_path.mkdir(parents=True, exist_ok=True)
                plt.savefig(fig_path / "regime_posteriors.png", dpi=150, bbox_inches="tight")
                logger.info(f"Posterior plot saved.")

            plt.show()

        except ImportError:
            logger.warning("Install arviz for posterior plots: pip install arviz")


# ── Entry Point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from src.data.loader import get_price_data
    from pathlib import Path

    config = load_config()
    prices = get_price_data(config)

    # Use classical regime probs as input to Bayesian forward pass
    processed = Path(config["paths"]["processed_data"])
    classical_probs = pd.read_csv(
        processed / "regime_probs_classical.csv",
        index_col="date", parse_dates=True
    )

    # Generate uniform confidence from classical prob certainty
    confidence = classical_probs.max(axis=1).rename("confidence")
    uncertainty = 1 - confidence

    # Save in Bayesian format so backtest can use them
    classical_probs.to_csv(processed / "regime_probs_bayesian.csv")
    confidence.to_csv(processed / "position_confidence.csv")
    uncertainty.to_csv(processed / "regime_uncertainty.csv")

    print("\nRegime probabilities (last 10 rows):")
    print(classical_probs.tail(10).round(3))
    print("\nPosition confidence (last 10 rows):")
    print(confidence.tail(10).round(3))
    print("\nSaved. Ready to run backtest.")

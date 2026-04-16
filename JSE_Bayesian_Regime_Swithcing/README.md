# JSE Top 40 Bayesian Regime-Switching Strategy

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-orange)]()

## Overview

A systematic equity strategy applied to the JSE Top 40 Index (`^J200`), using a **Bayesian Markov-switching model estimated via MCMC** to classify market regimes and dynamically route capital between sub-strategies optimised for each regime.

The core insight: South African equity markets exhibit structurally distinct regimes driven by a combination of global EM risk appetite, rand volatility, and commodity cycles. A single static strategy will underperform across the full cycle. This framework detects regime transitions probabilistically and adjusts positioning accordingly.

---

## Motivation

The JSE Top 40 presents a unique regime-switching environment compared to developed markets:

- **Rand sensitivity**: USDZAR volatility acts as an EM stress barometer, often leading JSE drawdowns
- **Commodity cycle dependency**: Resources constitute a significant index weight, creating divergent sub-regimes
- **EM contagion risk**: Global risk-off episodes (GFC, COVID, Fed tightening cycles) transmit rapidly to SA equities
- **SA-specific shocks**: Political risk events (Nenegate 2015, load-shedding cycles, rating downgrades) create identifiable stress regimes

Classical regime models trained on US data fail to capture this structure. This project builds a SA-specific framework from the ground up.

---

## Methodology

### Regime Detection

**Phase 1 — Baseline**: Hamilton (1989) Markov-switching model with 2 and 3 states, estimated via the EM algorithm (`statsmodels`). Serves as the interpretable benchmark.

**Phase 2 — Bayesian MCMC**: Full Bayesian estimation of the Markov-switching model using `PyMC`. Posterior distributions over regime probabilities enable uncertainty-aware position sizing — positions scale with posterior confidence, not just regime label.

### Feature Set

| Feature | Source | Rationale |
|---|---|---|
| JSE Top 40 realized volatility (21-day) | `^J200` via yfinance | Primary vol regime signal |
| USDZAR realized volatility | `ZAR=X` via yfinance | EM risk-off indicator |
| USDZAR return | `ZAR=X` via yfinance | Rand strength/weakness |
| Gold price return (USD) | `GC=F` via yfinance | Resources sector proxy |
| Brent crude return | `BZ=F` via yfinance | Energy/resources cycle |
| VIX level | `^VIX` via yfinance | Global risk appetite |
| SA yield spread (10yr - 3m proxy) | Derived | Local macro regime |
| Return autocorrelation (rolling) | `^J200` via yfinance | Trending vs mean-reverting |

### Regimes

| Regime | Characteristics | Strategy |
|---|---|---|
| **Risk-On / Growth** | Low vol, rand stable, resources bid | Momentum long, full sizing |
| **Risk-Off / Stress** | High vol, rand weakening, VIX elevated | Reduce exposure, defensive tilt |
| **Stagflation / Mixed** | Rand weak, commodities diverge from domestics | Selective exposure, reduced size |

### Strategy Router

Each period's regime probability vector from the MCMC posterior drives:
1. **Sub-strategy selection** — momentum, mean-reversion, or defensive
2. **Position sizing** — scaled by posterior certainty (high uncertainty → reduced exposure)
3. **Stop-loss thresholds** — tighter in stress regimes

### Backtesting

- Framework: `vectorbt`
- Period: 2005–present (captures GFC, Nenegate, COVID, post-COVID tightening)
- Transaction costs: 30bps round trip (realistic JSE assumption)
- Benchmark: JSE Top 40 buy-and-hold

---

## Project Structure

```
jse_regime_strategy/
│
├── config/
│   └── settings.yaml            # All parameters in one place
│
├── data/
│   ├── raw/                     # Downloaded data, never modified
│   └── processed/               # Cleaned, feature-engineered data
│
├── src/
│   ├── data/
│   │   ├── loader.py            # yfinance / Iress data ingestion
│   │   └── features.py          # Feature engineering pipeline
│   ├── models/
│   │   ├── markov_switching.py  # Classical Hamilton baseline
│   │   └── bayesian_regime.py   # MCMC Bayesian model (PyMC)
│   ├── strategy/
│   │   ├── router.py            # Regime → strategy mapping
│   │   └── backtest.py          # vectorbt backtesting engine
│   └── utils/
│       ├── plotting.py          # Standardised visualisations
│       └── metrics.py           # Performance metrics (Sharpe, Calmar, etc.)
│
├── notebooks/
│   ├── 01_eda.ipynb             # Exploratory data analysis
│   ├── 02_regime_baseline.ipynb # Classical Markov switching
│   ├── 03_bayesian_mcmc.ipynb   # Bayesian MCMC estimation
│   └── 04_backtest.ipynb        # Full strategy backtest
│
├── tests/
│   ├── test_loader.py
│   ├── test_features.py
│   └── test_metrics.py
│
├── results/
│   ├── figures/                 # All output charts
│   └── reports/                 # Performance summary reports
│
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/jse_regime_strategy.git
cd jse_regime_strategy
pip install -r requirements.txt
```

---

## Quick Start

```python
# Download and process data
python -m src.data.loader

# Run classical baseline
python -m src.models.markov_switching

# Run Bayesian MCMC model
python -m src.models.bayesian_regime

# Run full backtest
python -m src.strategy.backtest
```

Or follow the notebooks sequentially in `notebooks/`.

---

## Results

*To be populated as research progresses.*

Key metrics tracked:
- Annualised return vs benchmark
- Sharpe ratio (annualised, with autocorrelation adjustment)
- Maximum drawdown and Calmar ratio
- Regime classification accuracy (vs known SA market events)
- Posterior uncertainty distribution across regimes

---

## Data Sources

- **Primary**: Yahoo Finance via `yfinance` (free, industry-standard for research)
- **Alternative**: Iress (for higher-quality JSE constituent data where available)
- **Macro**: Federal Reserve FRED API for yield curve data

---

## Academic Context

This project extends the volatility-sentiment research methodology I developed in my MCom dissertation ("From News to Noise: Does Media Sentiment Drive Equity Market Volatility?", University of Pretoria, 2025) into a live trading framework. The key methodological evolution is the shift from EGARCH-X conditional volatility estimation to Bayesian Markov-switching regime classification, allowing for uncertainty-aware systematic trading decisions.

**Key references:**
- Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*.
- Kim, C.J. & Nelson, C.R. (1999). *State-Space Models with Regime Switching*. MIT Press.
- Ang, A. & Bekaert, G. (2002). Regime switches in interest rates. *Journal of Business & Economic Statistics*.

---

## Disclaimer

This repository is for research and educational purposes only. Nothing herein constitutes financial advice. Past backtest performance does not guarantee future results.

---

## Author

**Pieter** | MCom Economics (Quantitative Finance), University of Pretoria | CFA Level 1 Candidate (May 2026)

[LinkedIn](https://linkedin.com/in/YOUR_PROFILE) | [GitHub](https://github.com/YOUR_USERNAME)
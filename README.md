# Systematic Trading Strategies

Quantitative trading strategies with documented backtests and performance analysis. Built on a dual-framework validation approach (vectorized and event-driven backtesting), with a focus on regime-aware systematic methods applied to both US and South African equity markets.

## 🎯 Project Overview

This repository contains systematic trading strategies developed as part of my quantitative finance research. Strategies are backtested using both vectorized (vectorbt) and event-driven (backtrader) frameworks to validate robustness and identify implementation differences.

A core theme across this work is **market regime detection** — the recognition that a single static strategy underperforms across a full market cycle, and that dynamic capital routing based on probabilistic regime classification improves risk-adjusted returns.

---

## 📊 Strategy 1: Momentum Crossover (US Equities)

**Description:**
- Simple Moving Average (SMA) crossover: 10-day vs 50-day
- Long-only positions on bullish crossovers
- Asset: AAPL (Apple Inc.)
- Test Period: 2020–2025

**Performance Summary:**

| Metric | Vectorized | Event-Driven |
|---|---|---|
| **Total Return** | 148.6% | 140.9% |
| **Sharpe Ratio** | 1.05 | 0.77 |
| **Max Drawdown** | 26.1% | 26.3% |
| **Win Rate** | 35% | 38% |
| **Profit Factor** | 2.20 | 2.12 |
| **Total Trades** | 21 | 21 |

**Key Finding:** 8% performance difference between frameworks attributable to position sizing assumptions rather than strategy logic — vectorized backtests overestimate returns due to implicit execution assumptions.

### Results

**Vectorized Backtest**
![Vectorized Backtest Results](momentum_strategy/results/backtest_vectorized.png)

**Event-Driven Backtest**
![Event-Driven Backtest Results](momentum_strategy/results/backtest_event_driven.png)

---

## 📊 Strategy 2: JSE Top 40 Bayesian Regime-Switching Strategy

[![Status](https://img.shields.io/badge/Status-Research-orange)]()

**Description:**

A systematic equity strategy applied to the JSE Top 40 Index (`^J200`), using a Bayesian Markov-switching model estimated via MCMC to classify market regimes and dynamically route capital between sub-strategies optimised for each regime.

The core insight: South African equity markets exhibit structurally distinct regimes driven by EM risk appetite, rand volatility, and commodity cycles. A single static strategy underperforms across the full cycle. This framework detects regime transitions probabilistically and adjusts positioning accordingly.

**Why JSE-specific?**

The JSE Top 40 presents a unique regime-switching environment:
- **Rand sensitivity**: USDZAR volatility acts as an EM stress barometer, often leading JSE drawdowns
- **Commodity cycle dependency**: Resources constitute a significant index weight, creating divergent sub-regimes
- **EM contagion risk**: Global risk-off episodes transmit rapidly to SA equities
- **SA-specific shocks**: Political risk events (Nenegate 2015, load-shedding cycles, rating downgrades) create identifiable stress regimes

Classical regime models trained on US data fail to capture this structure.

**Methodology:**

| Phase | Model | Purpose |
|---|---|---|
| Baseline | Hamilton (1989) Markov-switching, 2–3 states via EM algorithm | Interpretable benchmark |
| Primary | Bayesian MCMC via PyMC | Uncertainty-aware position sizing |

**Regimes:**

| Regime | Characteristics | Strategy |
|---|---|---|
| Risk-On / Growth | Low vol, rand stable, resources bid | Momentum long, full sizing |
| Risk-Off / Stress | High vol, rand weakening, VIX elevated | Reduce exposure, defensive tilt |
| Stagflation / Mixed | Rand weak, commodities diverge from domestics | Selective exposure, reduced size |

**Feature Set:**

| Feature | Source | Rationale |
|---|---|---|
| JSE Top 40 realized volatility (21-day) | `^J200` | Primary vol regime signal |
| USDZAR realized volatility | `ZAR=X` | EM risk-off indicator |
| Gold price return (USD) | `GC=F` | Resources sector proxy |
| Brent crude return | `BZ=F` | Energy/resources cycle |
| VIX level | `^VIX` | Global risk appetite |
| Return autocorrelation (rolling) | `^J200` | Trending vs mean-reverting |

**Backtesting:**
- Framework: `vectorbt`
- Period: 2005–present (captures GFC, Nenegate, COVID, post-COVID tightening)
- Transaction costs: 30bps round trip (realistic JSE assumption)
- Benchmark: JSE Top 40 buy-and-hold

**Results:** *To be populated as research progresses.*

---

## 🔧 Technologies

- **Python 3.13**
- **Backtesting:** vectorbt, backtrader
- **Bayesian Modelling:** PyMC
- **Regime Detection:** statsmodels (Markov-switching), hmmlearn
- **Data:** yfinance, FRED API
- **Analysis:** pandas, numpy, scikit-learn
- **Visualization:** matplotlib

---

## 📁 Repository Structure
momentum_strategy/              # Strategy 1 — US momentum
├── vectorized_backtest.py
├── event_driven_backtest.py
└── results/
jse_regime_strategy/            # Strategy 2 — JSE regime-switching
├── config/
│   └── settings.yaml
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data/
│   │   ├── loader.py
│   │   └── features.py
│   ├── models/
│   │   ├── markov_switching.py
│   │   └── bayesian_regime.py
│   ├── strategy/
│   │   ├── router.py
│   │   └── backtest.py
│   └── utils/
│       ├── plotting.py
│       └── metrics.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_regime_baseline.ipynb
│   ├── 03_bayesian_mcmc.ipynb
│   └── 04_backtest.ipynb
└── results/
utils/                          # Shared utilities
requirements.txt

---

## 🔬 Methodology

**Dual-Framework Validation (Strategy 1):**
1. **Vectorized** (vectorbt): Optimised for speed, assumes perfect execution
2. **Event-driven** (backtrader): More realistic, accounts for order execution details

**Bayesian Regime Detection (Strategy 2):**
Posterior distributions over regime probabilities enable uncertainty-aware position sizing — positions scale with posterior confidence, not just the regime label. This is a meaningful improvement over hard-switching models.

---

## ⚠️ Risk Disclosure

This repository is for research and educational purposes only. Nothing herein constitutes financial advice. Past backtest performance does not guarantee future results.

---

## 📝 Academic Context

This work extends the volatility-sentiment research methodology developed in my MCom dissertation (*"From News to Noise: Does Media Sentiment Drive Equity Market Volatility?"*, University of Pretoria, 2025) into a live trading framework. The key methodological evolution is from EGARCH-X conditional volatility estimation to Bayesian Markov-switching regime classification, enabling uncertainty-aware systematic trading decisions.

**Key references:**
- Hamilton, J.D. (1989). A new approach to the economic analysis of nonstationary time series and the business cycle. *Econometrica*.
- Kim, C.J. & Nelson, C.R. (1999). *State-Space Models with Regime Switching*. MIT Press.
- Ang, A. & Bekaert (2002). Regime switches in interest rates. *Journal of Business & Economic Statistics*.

See also: [Working Papers](https://github.com/PieterWANel/Working_Papers)

---

## 📧 Contact

**Pieter Nel** | MCom Economics (Quantitative Finance), University of Pretoria | CFA Level 1 Candidate (May 2026)

Email: piedanel@gmail.com  
LinkedIn: [pieterwanel](https://linkedin.com/in/pieterwanel)  
GitHub: [PieterWANel](https://github.com/PieterWANel)

---

**License:** MIT | **Last Updated:** April 2026
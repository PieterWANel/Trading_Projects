# Systematic Trading Strategies

Quantitative trading strategies with documented backtests and performance analysis. Focus on momentum-based approaches using dual-framework validation (vectorized and event-driven backtesting).

## 🎯 Project Overview

This repository contains systematic trading strategies developed as part of my quantitative finance research. All strategies are backtested using both vectorized (vectorbt) and event-driven (backtrader) frameworks to validate robustness and identify implementation differences.

## 📊 Featured Strategy: Momentum Crossover

**Strategy Description:**
- Simple Moving Average (SMA) crossover: 10-day vs 50-day
- Long-only positions on bullish crossovers
- Asset: AAPL (Apple Inc.)
- Test Period: 2020-01-01 to 2025-12-31 (5 years)

**Performance Summary:**

| Metric | Vectorized | Event-Driven |
|--------|-----------|--------------|
| **Total Return** | 148.6% | 140.9% |
| **Sharpe Ratio** | 1.05 | 0.77 |
| **Max Drawdown** | 26.1% | 26.3% |
| **Win Rate** | 35% | 38% |
| **Profit Factor** | 2.20 | 2.12 |
| **Total Trades** | 21 | 21 |

**Key Finding:** 8% performance difference between frameworks attributable to position sizing assumptions rather than strategy logic.

## 🔧 Technologies

- **Python 3.13**
- **Backtesting Frameworks:** vectorbt, backtrader
- **Data:** yfinance
- **Analysis:** pandas, numpy
- **Visualization:** matplotlib

## 📁 Repository Structure
```
momentum_strategy/     # Main trading strategy implementation
├── vectorized_backtest.py    # Vectorbt implementation
├── event_driven_backtest.py  # Backtrader implementation
└── results/                  # Performance charts and analysis

utils/                 # Helper functions
requirements.txt       # Python dependencies
```

## 📈 Results Visualization

### Vectorized Backtest
![Vectorized Backtest Results](momentum_strategy/results/backtest_vectorized.png)

### Event-Driven Backtest
![Event-Driven Backtest Results](momentum_strategy/results/backtest_event_driven.png)

## 🔬 Methodology

**Dual-Framework Validation:**
1. **Vectorized approach** (vectorbt): Optimized for speed, assumes perfect execution
2. **Event-driven approach** (backtrader): More realistic, accounts for order execution details

**Key Insight:** The 8% performance gap reveals that vectorized backtests may overestimate returns due to implicit assumptions about position sizing and trade execution. Event-driven testing provides more conservative estimates closer to live trading reality.

## ⚠️ Risk Disclosure

This is educational research demonstrating backtesting methodology. Past performance does not guarantee future results. The strategy underperformed buy-and-hold (AAPL: +276% over same period), highlighting the importance of:
- Transaction costs
- Slippage
- Market regime dependency
- Single-asset concentration risk

## 🎯 Future Work

- [ ] Implement walk-forward optimization to address forward-looking bias
- [ ] Expand to multi-asset portfolio (10-20 stocks)
- [ ] Add machine learning-based entry/exit signals
- [ ] Integrate Interactive Brokers API for paper trading
- [ ] Implement risk parity position sizing

## 📝 Related Research

See my Master's dissertation on volatility modeling: [Working Papers](https://github.com/PieterWANel/Working_Papers)

## 📧 Contact

Pieter Nel  
Email: piedanel@gmail.com  
LinkedIn: [pieterwanel](https://linkedin.com/in/pieterwanel)

---

**License:** MIT  
**Last Updated:** January 2026

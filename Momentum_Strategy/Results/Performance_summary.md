# Performance Analysis

## Test Period
- Start: 2020-01-02
- End: 2025-12-30
- Duration: 1,507 days (~5 years)
- Asset: AAPL

## Vectorized Results (vectorbt)
- Total Return: 148.6%
- Sharpe Ratio: 1.05
- Max Drawdown: 26.1%
- Calmar Ratio: 0.94
- Win Rate: 35%
- Profit Factor: 2.20

## Event-Driven Results (backtrader)
- Total Return: 140.9%
- Sharpe Ratio: 0.77
- Max Drawdown: 26.3%
- Win Rate: 38%
- Profit Factor: 2.12
- Avg Win: $2,117
- Avg Loss: $664

## Framework Comparison

| Aspect | Difference | Explanation |
|--------|-----------|-------------|
| Total Return | 8% | Position sizing assumptions |
| Sharpe Ratio | 0.28 | Vectorized assumes perfect execution |
| Win Rate | 3% | Order fill timing differences |
| Profit Factor | 0.08 | Minor differences in trade accounting |

## Key Takeaway

Vectorized backtesting provides optimistic upper bound, while event-driven testing offers conservative lower bound. Real-world performance likely falls between these estimates.
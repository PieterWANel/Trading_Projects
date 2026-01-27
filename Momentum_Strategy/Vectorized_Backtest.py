import vectorbt as vbt
import yfinance as yf

# Load Data
data = yf.download("AAPL", start="2020-01-01", end="2025-12-31")

# Moving Average Crossover
fast_ma = vbt.MA.run(data["Close"], 10)
slow_ma = vbt.MA.run(data["Close"], 50)

# Generate Signals
enter = fast_ma.ma_crossed_above(slow_ma)
exit = fast_ma.ma_crossed_below(slow_ma)

# Run Backtest
Portfolio = vbt.Portfolio.from_signals(
    data["Close"],
    enter, 
    exit, 
    init_cash=10000, 
    fees=0.001,
    freq="1D"
    )

# Results
print("=" * 60)
print("PERFORMANCE METRICS")
print("=" * 60)

print(Portfolio.stats())
#Portfolio.plot().show()

print("Creating visualization...")

best_col = Portfolio.sharpe_ratio().idxmax()
fig = Portfolio[best_col].plot().show()

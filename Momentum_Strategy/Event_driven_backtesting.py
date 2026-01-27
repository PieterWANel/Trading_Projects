import backtrader as bt
import yfinance as yf
import pandas as pd

class SimpleStrategy(bt.Strategy):
    params = (("fast_period", 10), ("slow_period", 50),)

    def __init__(self):
        self.fast_ma = bt.indicators.SMA(
            self.data.close, period=self.params.fast_period
        )
        self.slow_ma = bt.indicators.SMA(
            self.data.close, period=self.params.slow_period
        )
        self.crossover = bt.indicators.CrossOver(
            self.fast_ma, self.slow_ma
        )
    
    def next(self):
        if not self.position:
            if self.crossover > 0:
                # BUY WITH ALL AVAILABLE CASH
                size = self.broker.getcash() / self.data.close[0] * 0.95  # Use 95% of cash
                self.buy(size=size)
        elif self.crossover < 0:
            self.close()

# Setup
cerebro = bt.Cerebro()

# Add Data
dataframe = yf.download("AAPL", start="2020-01-01", end="2025-12-31", progress=False)
print(f"Downloaded {len(dataframe)} rows of dataframe")

print(f"Column type: {type(dataframe.columns)}")
print(f"Columns: {dataframe.columns.tolist()}")

if isinstance(dataframe.columns, pd.MultiIndex):
    dataframe.columns = dataframe.columns.droplevel(1)  # Drop the ticker level
    print(f"Fixed columns! Now: {dataframe.columns.tolist()}")

data = bt.feeds.PandasData(dataname=dataframe)
cerebro.adddata(data)
cerebro.addstrategy(SimpleStrategy)

# Run
cerebro.broker.setcash(10000)
cerebro.broker.setcommission(commission=0.001)

# Add Trading Stats
cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', riskfreerate=0.0)
cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
cerebro.addanalyzer(bt.analyzers.SQN, _name='sqn')

print("\n" + "="*70)
print("RUNNING BACKTEST")
print("="*70)

starting_value = cerebro.broker.getvalue()
print(f"Starting Portfolio Value: ${starting_value:,.2f}\n")

results = cerebro.run()
strat = results[0]

ending_value = cerebro.broker.getvalue()

print("\n" + "="*70)
print("PERFORMANCE STATISTICS")
print("="*70)

# Portfolio values
print(f"\nPortfolio:")
print(f"  Starting Value:        ${starting_value:,.2f}")
print(f"  Ending Value:          ${ending_value:,.2f}")
print(f"  Total P&L:             ${ending_value - starting_value:,.2f}")
print(f"  Total Return:          {((ending_value / starting_value) - 1) * 100:.2f}%")

# Returns analysis
returns_analysis = strat.analyzers.returns.get_analysis()
print(f"\nReturns:")
print(f"  Total Return:          {returns_analysis['rtot'] * 100:.2f}%")
print(f"  Average Return:        {returns_analysis['ravg'] * 100:.2f}%")
if 'rnorm' in returns_analysis:
    print(f"  Normalized Return:     {returns_analysis['rnorm']:.2f}%")

# Sharpe Ratio
sharpe = strat.analyzers.sharpe.get_analysis()
sharpe_ratio = sharpe.get('sharperatio', None)
if sharpe_ratio is not None:
    print(f"\nRisk-Adjusted Performance:")
    print(f"  Sharpe Ratio:          {sharpe_ratio:.2f}")
else:
    print(f"\nRisk-Adjusted Performance:")
    print(f"  Sharpe Ratio:          N/A")

# Drawdown
dd = strat.analyzers.drawdown.get_analysis()
print(f"\nDrawdown:")
print(f"  Max Drawdown:          {dd['max']['drawdown']:.2f}%")
print(f"  Max Drawdown Period:   {dd['max']['len']} days")
if dd['max']['moneydown'] > 0:
    print(f"  Max Money Down:        ${dd['max']['moneydown']:,.2f}")

# Trade Analysis
trades = strat.analyzers.trades.get_analysis()
total_trades = trades.total.get('total', 0)

if total_trades > 0:
    print(f"\nTrade Analysis:")
    print(f"  Total Trades:          {total_trades}")
    
    # Won trades
    won_total = trades.won.get('total', 0)
    won_pnl_total = trades.won.get('pnl', {}).get('total', 0)
    won_pnl_avg = trades.won.get('pnl', {}).get('average', 0)
    
    print(f"  Winning Trades:        {won_total}")
    print(f"  Total Win Amount:      ${won_pnl_total:,.2f}")
    print(f"  Average Win:           ${won_pnl_avg:,.2f}")
    
    # Lost trades
    lost_total = trades.lost.get('total', 0)
    lost_pnl_total = trades.lost.get('pnl', {}).get('total', 0)
    lost_pnl_avg = trades.lost.get('pnl', {}).get('average', 0)
    
    print(f"  Losing Trades:         {lost_total}")
    print(f"  Total Loss Amount:     ${lost_pnl_total:,.2f}")
    print(f"  Average Loss:          ${lost_pnl_avg:,.2f}")
    
    # Win rate and profit factor
    win_rate = (won_total / total_trades) * 100 if total_trades > 0 else 0
    print(f"  Win Rate:              {win_rate:.2f}%")
    
    if abs(lost_pnl_total) > 0:
        profit_factor = abs(won_pnl_total / lost_pnl_total)
        print(f"  Profit Factor:         {profit_factor:.2f}")
    
    # Longest winning/losing streaks
    if 'streak' in trades.won:
        print(f"  Longest Win Streak:    {trades.won.streak.get('longest', 0)}")
    if 'streak' in trades.lost:
        print(f"  Longest Loss Streak:   {trades.lost.streak.get('longest', 0)}")

# System Quality Number
sqn = strat.analyzers.sqn.get_analysis()
if 'sqn' in sqn:
    print(f"\nSystem Quality Number:")
    print(f"  SQN:                   {sqn['sqn']:.2f}")
    print(f"  Number of Trades:      {sqn.get('trades', 0)}")
    
    # SQN interpretation
    sqn_value = sqn['sqn']
    if sqn_value < 1.6:
        quality = "Poor"
    elif sqn_value < 1.9:
        quality = "Below Average"
    elif sqn_value < 2.4:
        quality = "Average"
    elif sqn_value < 2.9:
        quality = "Good"
    elif sqn_value < 5.0:
        quality = "Excellent"
    else:
        quality = "Superb"
    print(f"  Quality:               {quality}")

print("\n" + "="*70)

# Plot
cerebro.plot()

#cerebro.run()
#cerebro.plot()


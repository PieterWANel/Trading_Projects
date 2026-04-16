"""
src/utils/metrics.py
--------------------
Performance metrics for strategy evaluation.
Industry-standard metrics with SA-specific adjustments.
"""

import numpy as np
import pandas as pd


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.075,
    annualisation: int = 252,
    adjust_autocorr: bool = True,
) -> float:
    """
    Annualised Sharpe ratio with optional autocorrelation adjustment.
    
    Standard Sharpe understates risk when returns are autocorrelated
    (as in momentum strategies). The Lo (2002) adjustment corrects for this.
    
    Args:
        risk_free_rate:   Annual risk-free rate (SARB repo rate proxy: 7.5%)
        adjust_autocorr:  Apply Lo (2002) autocorrelation adjustment
    """
    excess = returns - risk_free_rate / annualisation
    mean_excess = excess.mean() * annualisation
    vol = returns.std() * np.sqrt(annualisation)

    if adjust_autocorr and len(returns) > 10:
        # Lo (2002) adjustment factor
        autocorrs = [returns.autocorr(lag=i) for i in range(1, min(13, len(returns) // 4))]
        q = len(autocorrs)
        adj_factor = np.sqrt(
            1 + 2 * sum(
                (1 - k / (q + 1)) * autocorrs[k - 1]
                for k in range(1, q + 1)
            )
        )
        vol_adj = vol * adj_factor
        return mean_excess / (vol_adj + 1e-10)

    return mean_excess / (vol + 1e-10)


def calmar_ratio(returns: pd.Series, annualisation: int = 252) -> float:
    """Annualised return / Maximum drawdown."""
    ann_return = returns.mean() * annualisation
    mdd = max_drawdown(returns)
    return ann_return / (abs(mdd) + 1e-10)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.075,
    annualisation: int = 252,
) -> float:
    """Sharpe ratio using downside deviation only."""
    excess = returns - risk_free_rate / annualisation
    downside = returns[returns < 0].std() * np.sqrt(annualisation)
    return (excess.mean() * annualisation) / (downside + 1e-10)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown."""
    cum = (1 + returns).cumprod()
    rolling_max = cum.expanding().max()
    drawdowns = cum / rolling_max - 1
    return drawdowns.min()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Full drawdown time series."""
    cum = (1 + returns).cumprod()
    rolling_max = cum.expanding().max()
    return cum / rolling_max - 1


def annualised_return(returns: pd.Series, annualisation: int = 252) -> float:
    return returns.mean() * annualisation


def annualised_volatility(returns: pd.Series, annualisation: int = 252) -> float:
    return returns.std() * np.sqrt(annualisation)


def hit_rate(returns: pd.Series) -> float:
    """Percentage of positive return periods."""
    return (returns > 0).mean()


def profit_factor(returns: pd.Series) -> float:
    """Gross profit / Gross loss."""
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    return gross_profit / (gross_loss + 1e-10)


def information_ratio(strategy_returns: pd.Series, benchmark_returns: pd.Series, annualisation: int = 252) -> float:
    """Annualised active return / tracking error."""
    active = strategy_returns - benchmark_returns
    return (active.mean() * annualisation) / (active.std() * np.sqrt(annualisation) + 1e-10)


def performance_summary(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series = None,
    risk_free_rate: float = 0.075,
    annualisation: int = 252,
    label: str = "Strategy",
) -> pd.DataFrame:
    """
    Full performance summary table.
    
    Returns:
        DataFrame with key metrics for strategy (and benchmark if provided)
    """
    def _metrics(ret: pd.Series, name: str) -> dict:
        return {
            "Name": name,
            "Ann. Return": f"{annualised_return(ret, annualisation):.2%}",
            "Ann. Volatility": f"{annualised_volatility(ret, annualisation):.2%}",
            "Sharpe (adj.)": f"{sharpe_ratio(ret, risk_free_rate, annualisation):.3f}",
            "Sortino": f"{sortino_ratio(ret, risk_free_rate, annualisation):.3f}",
            "Calmar": f"{calmar_ratio(ret, annualisation):.3f}",
            "Max Drawdown": f"{max_drawdown(ret):.2%}",
            "Hit Rate": f"{hit_rate(ret):.2%}",
            "Profit Factor": f"{profit_factor(ret):.2f}",
        }

    rows = [_metrics(strategy_returns, label)]

    if benchmark_returns is not None:
        rows.append(_metrics(benchmark_returns, "Benchmark (JSE Top 40)"))
        ir = information_ratio(strategy_returns, benchmark_returns, annualisation)
        rows[0]["Information Ratio"] = f"{ir:.3f}"

    return pd.DataFrame(rows).set_index("Name").T


def regime_performance_breakdown(
    returns: pd.Series,
    regime_labels: pd.Series,
    annualisation: int = 252,
) -> pd.DataFrame:
    """
    Performance statistics broken down by regime.
    Useful for validating regime model quality.
    """
    results = {}
    for regime in regime_labels.unique():
        mask = regime_labels == regime
        regime_ret = returns[mask]
        results[regime] = {
            "Ann. Return": annualised_return(regime_ret, annualisation),
            "Ann. Vol": annualised_volatility(regime_ret, annualisation),
            "Sharpe": sharpe_ratio(regime_ret, annualisation=annualisation, adjust_autocorr=False),
            "Hit Rate": hit_rate(regime_ret),
            "N Periods": len(regime_ret),
            "% of Sample": len(regime_ret) / len(returns),
        }

    df = pd.DataFrame(results).T
    df["Ann. Return"] = df["Ann. Return"].map("{:.2%}".format)
    df["Ann. Vol"] = df["Ann. Vol"].map("{:.2%}".format)
    df["Sharpe"] = df["Sharpe"].map("{:.3f}".format)
    df["Hit Rate"] = df["Hit Rate"].map("{:.2%}".format)
    df["% of Sample"] = df["% of Sample"].map("{:.1%}".format)
    return df

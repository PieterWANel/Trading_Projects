"""
Simulation Runner & Analytics
==============================
High-level interface to run experiments, batch simulations,
and compute equilibrium analytics.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from .engine import GlostenMilgromModel, SimulationState, Trade
from .players import MarketMaker, InformedTrader, NoiseTrader


# ---------------------------------------------------------------------------
# Simulation Runner
# ---------------------------------------------------------------------------

class Simulation:
    """
    Orchestrates a full Glosten-Milgrom simulation run.

    Parameters
    ----------
    v_low, v_high       : True asset value support
    prob_v_high         : Prior probability V = V_H
    mu                  : P(any arriving trader is informed)
    informed_strategy   : 'aggressive' | 'mixed' | 'patient'
    noise_strategy      : 'random' | 'momentum' | 'contrarian'
    n_rounds            : Number of trading rounds
    seed                : Random seed for reproducibility
    """

    def __init__(
        self,
        v_low:             float = 40.0,
        v_high:            float = 60.0,
        prob_v_high:       float = 0.5,
        mu:                float = 0.3,
        informed_strategy: str   = "aggressive",
        noise_strategy:    str   = "random",
        n_rounds:          int   = 100,
        seed:              Optional[int] = 42,
    ):
        self.state = SimulationState(
            v_low            = v_low,
            v_high           = v_high,
            prob_v_high      = prob_v_high,
            mu               = mu,
            informed_strategy= informed_strategy,
        )
        self.model        = GlostenMilgromModel(self.state)
        self.mm           = MarketMaker()
        self.informed     = InformedTrader(strategy=informed_strategy)
        self.noise        = NoiseTrader(strategy=noise_strategy)
        self.n_rounds     = n_rounds
        self.seed         = seed

    def run(self) -> "Simulation":
        self.state.reset(seed=self.seed)
        self.mm.reset()
        self.informed.reset()
        self.noise.reset()
        for _ in range(self.n_rounds):
            self.model.step()
        return self

    def _avg_spread_decomp(self) -> dict:
        """
        Spread decomposition at the AVERAGE belief over the simulation,
        not at the terminal (converged) belief. This gives a meaningful
        decomposition rather than near-zero values after convergence.
        """
        df = self.trades_df()
        avg_belief = float(df["belief_high"].mean())
        # Temporarily set belief to average for decomposition
        orig = self.state.belief_high
        self.state.belief_high = avg_belief
        decomp = self.model.spread_decomposition()
        decomp["note"] = f"Computed at avg belief={avg_belief:.3f} (not terminal belief)"
        self.state.belief_high = orig
        return decomp

    def trades_df(self) -> pd.DataFrame:
        """Return all trades as a tidy DataFrame."""
        rows = []
        for t in self.state.trades:
            rows.append({
                "round":        t.round_num,
                "trader_type":  t.trader_type,
                "action":       t.action,
                "price":        t.price,
                "bid":          t.quote.bid,
                "ask":          t.quote.ask,
                "spread":       t.quote.spread,
                "mid":          t.quote.mid,
                "true_value":   t.true_value,
                "belief_high":  t.belief_high,
                "mm_pnl":       t.mm_pnl,
                "it_pnl":       t.it_pnl,
                "nt_pnl":       t.nt_pnl,
                "mm_cum_pnl":   sum(x.mm_pnl for x in self.state.trades[:t.round_num]),
                "it_cum_pnl":   sum(x.it_pnl for x in self.state.trades[:t.round_num]),
                "nt_cum_pnl":   sum(x.nt_pnl for x in self.state.trades[:t.round_num]),
                "note":         t.note,
            })
        return pd.DataFrame(rows)

    def summary(self) -> Dict:
        """Return summary statistics."""
        df = self.trades_df()
        executed = df[df["action"] != "hold"]

        return {
            "true_value":         self.state.true_value,
            "final_belief_high":  self.state.belief_high,
            "final_mid":          self.model.compute_quotes().mid,
            "n_rounds":           self.n_rounds,
            "n_informed_trades":  int((executed["trader_type"] == "informed").sum()),
            "n_noise_trades":     int((executed["trader_type"] == "noise").sum()),
            "avg_spread":         round(float(df["spread"].mean()), 4),
            "final_spread":       round(float(df["spread"].iloc[-1]), 4),
            "mm_total_pnl":       round(float(self.state.mm_total_pnl), 4),
            "it_total_pnl":       round(float(self.state.it_total_pnl), 4),
            "nt_total_pnl":       round(float(self.state.nt_total_pnl), 4),
            "spread_decomp":      self._avg_spread_decomp(),
        }


# ---------------------------------------------------------------------------
# Batch Experiments
# ---------------------------------------------------------------------------

class BatchExperiment:
    """
    Run parameter sweeps over mu or other variables.
    Useful for equilibrium analysis and plotting.
    """

    @staticmethod
    def sweep_mu(
        mu_values: Optional[List[float]] = None,
        n_rounds:  int   = 200,
        n_seeds:   int   = 10,
        **kwargs
    ) -> pd.DataFrame:
        """
        Sweep over mu (probability of informed trader) and record
        average spread, P&L, and belief convergence speed.
        """
        if mu_values is None:
            mu_values = list(np.linspace(0.0, 0.9, 19))

        records = []
        for mu in mu_values:
            for seed in range(n_seeds):
                sim = Simulation(mu=mu, n_rounds=n_rounds, seed=seed, **kwargs)
                sim.run()
                s = sim.summary()
                records.append({
                    "mu":              mu,
                    "seed":            seed,
                    "avg_spread":      s["avg_spread"],
                    "final_spread":    s["final_spread"],
                    "mm_pnl":          s["mm_total_pnl"],
                    "it_pnl":          s["it_total_pnl"],
                    "nt_pnl":          s["nt_total_pnl"],
                    "final_belief":    s["final_belief_high"],
                    "belief_error":    abs(s["final_belief_high"] -
                                          (1.0 if sim.state.true_value == sim.state.v_high else 0.0)),
                })

        df = pd.DataFrame(records)
        # Average across seeds
        return df.groupby("mu").mean(numeric_only=True).reset_index()

    @staticmethod
    def strategy_comparison(
        strategies: Optional[List[str]] = None,
        n_rounds:   int   = 100,
        n_seeds:    int   = 20,
        **kwargs
    ) -> pd.DataFrame:
        """Compare informed trader strategies head-to-head."""
        if strategies is None:
            strategies = ["aggressive", "mixed", "patient"]

        records = []
        for strat in strategies:
            for seed in range(n_seeds):
                sim = Simulation(informed_strategy=strat, n_rounds=n_rounds,
                                 seed=seed, **kwargs)
                sim.run()
                s = sim.summary()
                records.append({
                    "strategy":    strat,
                    "it_pnl":      s["it_total_pnl"],
                    "mm_pnl":      s["mm_total_pnl"],
                    "nt_pnl":      s["nt_total_pnl"],
                    "avg_spread":  s["avg_spread"],
                    "n_it_trades": s["n_informed_trades"],
                })

        df = pd.DataFrame(records)
        return df.groupby("strategy").mean(numeric_only=True).reset_index()
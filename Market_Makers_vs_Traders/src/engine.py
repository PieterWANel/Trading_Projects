"""
Glosten-Milgrom Market Microstructure Model
============================================
Three-player sequential trading game:
  - Market Maker  : sets bid/ask spread, updates beliefs via Bayes' rule
  - Informed Trader: knows true asset value V, trades to maximise profit
  - Noise Trader  : trades randomly (liquidity-driven, no information)

Reference: Glosten & Milgrom (1985), "Bid, Ask and Transaction Prices in a
Specialist Market with Heterogeneously Informed Traders", JFE 14(1), 71-100.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Literal, List, Optional
import random


# ---------------------------------------------------------------------------
# Data Structures
# ---------------------------------------------------------------------------

TraderType = Literal["informed", "noise"]
Action     = Literal["buy", "sell", "hold"]


@dataclass
class Quote:
    """Market maker's posted bid and ask prices."""
    bid: float
    ask: float

    @property
    def spread(self) -> float:
        return round(self.ask - self.bid, 4)

    @property
    def mid(self) -> float:
        return round((self.bid + self.ask) / 2, 4)


@dataclass
class Trade:
    """A single executed trade."""
    round_num:   int
    trader_type: TraderType
    action:      Action
    price:       float          # transaction price (ask if buy, bid if sell)
    true_value:  float
    quote:       Quote
    mm_pnl:      float          # market maker P&L this trade
    it_pnl:      float          # informed trader P&L this trade
    nt_pnl:      float          # noise trader P&L this trade
    mu_before:   float          # P(informed) before trade
    mu_after:    float          # P(informed) after trade
    belief_high: float          # P(V = V_H) after trade
    note:        str = ""


@dataclass
class SimulationState:
    """Full simulation state."""
    # Parameters
    v_low:        float = 40.0   # low true value
    v_high:       float = 60.0   # high true value
    prob_v_high:  float = 0.5    # prior P(V = V_H)
    mu:           float = 0.3    # P(any trader is informed)
    informed_strategy: str = "aggressive"  # aggressive | mixed

    # Running state
    true_value:   float = 0.0
    belief_high:  float = 0.5    # market maker's posterior P(V = V_H)
    round_num:    int   = 0
    trades:       List[Trade] = field(default_factory=list)

    # Cumulative P&L
    mm_total_pnl: float = 0.0
    it_total_pnl: float = 0.0
    nt_total_pnl: float = 0.0

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        self.true_value  = self.v_high if random.random() < self.prob_v_high else self.v_low
        self.belief_high = self.prob_v_high
        self.round_num   = 0
        self.trades      = []
        self.mm_total_pnl = 0.0
        self.it_total_pnl = 0.0
        self.nt_total_pnl = 0.0


# ---------------------------------------------------------------------------
# Core Model
# ---------------------------------------------------------------------------

class GlostenMilgromModel:
    """
    Implements the Glosten-Milgrom (1985) sequential trade model.

    Equilibrium conditions
    ----------------------
    The market maker's zero-profit ask and bid prices satisfy:

        ask = E[V | buy order]
            = P(informed|buy)*V_H  +  (1-P(informed|buy)) * E[V|belief]
            (informed always buys if V=V_H; noise trader buys w.p. 0.5)

        bid = E[V | sell order]
            (informed always sells if V=V_L; noise trader sells w.p. 0.5)

    Bayesian updating after observing a buy order:
        P(V=V_H | buy) = P(buy|V=V_H)*P(V=V_H) / P(buy)

    where P(buy) mixes over informed and noise trader actions.
    """

    def __init__(self, state: SimulationState):
        self.s = state

    # ------------------------------------------------------------------
    # Quote Calculation (Zero-Profit Market Maker)
    # ------------------------------------------------------------------

    def compute_quotes(self) -> Quote:
        """
        Compute zero-profit bid and ask given current belief P(V=V_H).
        """
        mu   = self.s.mu
        p_h  = self.s.belief_high
        v_h  = self.s.v_high
        v_l  = self.s.v_low
        ev   = p_h * v_h + (1 - p_h) * v_l   # unconditional expected value

        # --- ASK PRICE (MM sells, trader buys) ---
        # P(buy | V=V_H): informed buys with prob 1 if aggressive, 0.5 if mixed
        p_buy_given_vh = 1.0 if self.s.informed_strategy == "aggressive" else 0.5
        # P(buy | V=V_L): informed holds (or sells), noise buys w.p. 0.5
        p_buy_given_vl = 0.0 if self.s.informed_strategy == "aggressive" else 0.5

        # Total buy probability
        p_buy = (mu * (p_h * p_buy_given_vh + (1 - p_h) * p_buy_given_vl)
                 + (1 - mu) * 0.5)

        if p_buy > 1e-9:
            # P(V=V_H | buy order arrived) via Bayes
            p_vh_given_buy = (mu * p_h * p_buy_given_vh
                              + (1 - mu) * p_h * 0.5) / p_buy
            ask = p_vh_given_buy * v_h + (1 - p_vh_given_buy) * v_l
        else:
            ask = ev

        # --- BID PRICE (MM buys, trader sells) ---
        p_sell_given_vh = 0.0 if self.s.informed_strategy == "aggressive" else 0.5
        p_sell_given_vl = 1.0 if self.s.informed_strategy == "aggressive" else 0.5

        p_sell = (mu * (p_h * p_sell_given_vh + (1 - p_h) * p_sell_given_vl)
                  + (1 - mu) * 0.5)

        if p_sell > 1e-9:
            p_vh_given_sell = (mu * p_h * p_sell_given_vh
                               + (1 - mu) * p_h * 0.5) / p_sell
            bid = p_vh_given_sell * v_h + (1 - p_vh_given_sell) * v_l
        else:
            bid = ev

        return Quote(bid=round(bid, 4), ask=round(ask, 4))

    # ------------------------------------------------------------------
    # Bayesian Belief Update
    # ------------------------------------------------------------------

    def update_belief(self, action: Action) -> float:
        """
        Update market maker's belief P(V=V_H) after observing a trade direction.
        Returns new belief.
        """
        mu  = self.s.mu
        p_h = self.s.belief_high
        v_h = self.s.v_high
        v_l = self.s.v_low

        aggressive = self.s.informed_strategy == "aggressive"

        if action == "buy":
            p_buy_given_vh = 1.0 if aggressive else 0.5
            p_buy_given_vl = 0.0 if aggressive else 0.5
            # P(buy | V=V_H) mixing informed and noise
            like_h = mu * p_buy_given_vh + (1 - mu) * 0.5
            like_l = mu * p_buy_given_vl + (1 - mu) * 0.5
        elif action == "sell":
            p_sell_given_vh = 0.0 if aggressive else 0.5
            p_sell_given_vl = 1.0 if aggressive else 0.5
            like_h = mu * p_sell_given_vh + (1 - mu) * 0.5
            like_l = mu * p_sell_given_vl + (1 - mu) * 0.5
        else:
            return p_h  # hold: no information

        denom = like_h * p_h + like_l * (1 - p_h)
        if denom < 1e-12:
            return p_h
        new_belief = (like_h * p_h) / denom
        return round(min(max(new_belief, 0.0), 1.0), 6)

    # ------------------------------------------------------------------
    # Trader Actions
    # ------------------------------------------------------------------

    def informed_action(self) -> Action:
        """Informed trader's optimal action given true value vs current mid."""
        quote = self.compute_quotes()
        v = self.s.true_value

        if self.s.informed_strategy == "aggressive":
            # Buy if true value > ask (profit), sell if true value < bid
            if v > quote.ask:
                return "buy"
            elif v < quote.bid:
                return "sell"
            else:
                return "hold"
        else:
            # Mixed strategy: randomise to hide information
            if v > quote.mid:
                return "buy" if random.random() < 0.7 else "hold"
            elif v < quote.mid:
                return "sell" if random.random() < 0.7 else "hold"
            else:
                return "hold"

    def noise_action(self) -> Action:
        """Noise trader acts randomly: 50/50 buy or sell."""
        return "buy" if random.random() < 0.5 else "sell"

    # ------------------------------------------------------------------
    # Single Round
    # ------------------------------------------------------------------

    def step(self) -> Trade:
        """Execute one round of the game."""
        self.s.round_num += 1
        quote = self.compute_quotes()
        mu_before = self.s.mu

        # Determine trader type this round
        trader_type: TraderType = "informed" if random.random() < self.s.mu else "noise"

        # Trader chooses action
        if trader_type == "informed":
            action = self.informed_action()
        else:
            action = self.noise_action()

        # Transaction price and P&L
        v = self.s.true_value
        mm_pnl = it_pnl = nt_pnl = 0.0

        if action == "buy":
            price  = quote.ask
            mm_pnl = price - v          # MM sold at ask, asset worth V
            if trader_type == "informed":
                it_pnl = v - price      # bought at ask, asset worth V
            else:
                nt_pnl = v - price
        elif action == "sell":
            price  = quote.bid
            mm_pnl = v - price          # MM bought at bid, asset worth V
            if trader_type == "informed":
                it_pnl = price - v      # sold at bid, asset worth V
            else:
                nt_pnl = price - v
        else:
            price = quote.mid

        # Bayesian update
        if action in ("buy", "sell"):
            new_belief = self.update_belief(action)
        else:
            new_belief = self.s.belief_high

        # Note
        ev_new = new_belief * self.s.v_high + (1 - new_belief) * self.s.v_low
        note = (f"{'Informed' if trader_type=='informed' else 'Noise'} trader "
                f"{action}s at {price:.2f}. "
                f"MM belief P(V_H) {self.s.belief_high:.3f}→{new_belief:.3f}. "
                f"Implied EV={ev_new:.2f}.")

        trade = Trade(
            round_num   = self.s.round_num,
            trader_type = trader_type,
            action      = action,
            price       = round(price, 4),
            true_value  = v,
            quote       = quote,
            mm_pnl      = round(mm_pnl, 4),
            it_pnl      = round(it_pnl, 4),
            nt_pnl      = round(nt_pnl, 4),
            mu_before   = mu_before,
            mu_after    = self.s.mu,     # mu is a parameter (not updated here)
            belief_high = new_belief,
            note        = note,
        )

        # Persist state
        self.s.belief_high    = new_belief
        self.s.mm_total_pnl  += mm_pnl
        self.s.it_total_pnl  += it_pnl
        self.s.nt_total_pnl  += nt_pnl
        self.s.trades.append(trade)

        return trade

    # ------------------------------------------------------------------
    # Run Full Simulation
    # ------------------------------------------------------------------

    def run(self, n_rounds: int = 50, seed: Optional[int] = None) -> SimulationState:
        """Run n_rounds of the game from a fresh state."""
        self.s.reset(seed=seed)
        for _ in range(n_rounds):
            self.step()
        return self.s

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def spread_decomposition(self) -> dict:
        """
        Decompose the equilibrium spread into:
          - Adverse selection component  (informed trader cost)
          - Inventory / order-processing component (residual)
        """
        quote = self.compute_quotes()
        ev    = self.s.belief_high * self.s.v_high + (1 - self.s.belief_high) * self.s.v_low
        half_spread = quote.spread / 2

        # Adverse selection = E[loss to informed | trade]
        mu  = self.s.mu
        v_h = self.s.v_high
        v_l = self.s.v_low
        p_h = self.s.belief_high

        # Expected adverse selection cost on ask side
        adv_sel_ask = mu * p_h * (v_h - quote.ask)
        adv_sel_bid = mu * (1 - p_h) * (quote.bid - v_l)
        adv_sel = abs(adv_sel_ask) + abs(adv_sel_bid)

        return {
            "total_spread":     round(quote.spread, 4),
            "half_spread":      round(half_spread, 4),
            "adverse_selection":round(adv_sel, 4),
            "order_processing": round(max(half_spread - adv_sel / 2, 0), 4),
            "ask":              quote.ask,
            "bid":              quote.bid,
            "mid":              quote.mid,
        }

    def equilibrium_spread_vs_mu(self, mu_range=None) -> List[dict]:
        """Compute equilibrium spread across a range of mu values."""
        if mu_range is None:
            mu_range = np.linspace(0.0, 1.0, 51)
        original_mu = self.s.mu
        results = []
        for mu in mu_range:
            self.s.mu = mu
            q = self.compute_quotes()
            results.append({"mu": round(mu, 3), "spread": round(q.spread, 4),
                            "ask": round(q.ask, 4), "bid": round(q.bid, 4)})
        self.s.mu = original_mu
        return results

    def information_leakage(self) -> List[dict]:
        """Track how belief converges to truth over rounds."""
        return [
            {
                "round":       t.round_num,
                "belief_high": t.belief_high,
                "true_value":  t.true_value,
                "spread":      t.quote.spread,
                "trader_type": t.trader_type,
                "action":      t.action,
            }
            for t in self.s.trades
        ]

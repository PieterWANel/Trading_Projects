"""
Player Strategies
=================
Encapsulates strategy logic for each of the three players.
Allows easy extension with alternative strategies (e.g. RL agents).
"""

import random
from abc import ABC, abstractmethod
from typing import Optional
from .engine import Quote, SimulationState, Action, TraderType


# ---------------------------------------------------------------------------
# Abstract Base
# ---------------------------------------------------------------------------

class Player(ABC):
    """Base class for all market participants."""

    def __init__(self, name: str):
        self.name = name
        self.total_pnl: float = 0.0
        self.trade_count: int = 0

    @abstractmethod
    def act(self, state: SimulationState, quote: Quote) -> Action:
        """Return the player's chosen action given current state and quote."""
        ...

    def record_pnl(self, pnl: float):
        self.total_pnl += pnl
        if pnl != 0:
            self.trade_count += 1

    def reset(self):
        self.total_pnl  = 0.0
        self.trade_count = 0

    def __repr__(self):
        return f"{self.name}(pnl={self.total_pnl:.2f}, trades={self.trade_count})"


# ---------------------------------------------------------------------------
# Market Maker
# ---------------------------------------------------------------------------

class MarketMaker(Player):
    """
    Zero-profit competitive market maker.
    Sets bid/ask to break even in expectation given beliefs.
    Does NOT know whether any given trader is informed.
    """

    def __init__(self):
        super().__init__("Market Maker")

    def act(self, state: SimulationState, quote: Quote) -> Action:
        # Market maker is passive — always quotes, never initiates
        return "hold"

    def quote(self, model) -> Quote:
        """Delegate to the model's equilibrium quote computation."""
        return model.compute_quotes()


# ---------------------------------------------------------------------------
# Informed Trader
# ---------------------------------------------------------------------------

class InformedTrader(Player):
    """
    Trader with perfect knowledge of the true asset value V.
    Strategies:
      - aggressive : always buy if V > ask, sell if V < bid
      - mixed      : randomises to partially conceal information
      - patient    : waits for spread to narrow before trading
    """

    STRATEGIES = ("aggressive", "mixed", "patient")

    def __init__(self, strategy: str = "aggressive"):
        assert strategy in self.STRATEGIES, f"Strategy must be one of {self.STRATEGIES}"
        super().__init__(f"Informed Trader [{strategy}]")
        self.strategy = strategy

    def act(self, state: SimulationState, quote: Quote) -> Action:
        v = state.true_value

        if self.strategy == "aggressive":
            if v > quote.ask:
                return "buy"
            elif v < quote.bid:
                return "sell"
            return "hold"

        elif self.strategy == "mixed":
            # Randomise direction to hide information, but bias toward profit
            if v > quote.mid:
                r = random.random()
                if r < 0.65:  return "buy"
                if r < 0.80:  return "hold"
                return "sell"
            elif v < quote.mid:
                r = random.random()
                if r < 0.65:  return "sell"
                if r < 0.80:  return "hold"
                return "buy"
            return "hold"

        elif self.strategy == "patient":
            # Only trade when spread is tight enough that profit > threshold
            threshold = (state.v_high - state.v_low) * 0.1
            if v > quote.ask and (v - quote.ask) > threshold:
                return "buy"
            elif v < quote.bid and (quote.bid - v) > threshold:
                return "sell"
            return "hold"

        return "hold"

    def expected_profit(self, quote: Quote, true_value: float) -> float:
        """Expected profit from the best available action."""
        buy_profit  = true_value - quote.ask
        sell_profit = quote.bid  - true_value
        return max(buy_profit, sell_profit, 0.0)


# ---------------------------------------------------------------------------
# Noise (Uninformed) Trader
# ---------------------------------------------------------------------------

class NoiseTrader(Player):
    """
    Uninformed trader with no informational edge.
    Trades for exogenous liquidity reasons (rebalancing, hedging, cash needs).
    Strategies:
      - random     : 50/50 buy or sell each round
      - momentum   : follows recent price direction
      - contrarian : fades recent price moves
    """

    STRATEGIES = ("random", "momentum", "contrarian")

    def __init__(self, strategy: str = "random"):
        assert strategy in self.STRATEGIES, f"Strategy must be one of {self.STRATEGIES}"
        super().__init__(f"Noise Trader [{strategy}]")
        self.strategy = strategy
        self._price_history: list = []

    def act(self, state: SimulationState, quote: Quote) -> Action:
        self._price_history.append(quote.mid)

        if self.strategy == "random":
            return "buy" if random.random() < 0.5 else "sell"

        elif self.strategy == "momentum":
            if len(self._price_history) < 3:
                return "buy" if random.random() < 0.5 else "sell"
            recent_move = self._price_history[-1] - self._price_history[-3]
            if recent_move > 0:
                return "buy"  if random.random() < 0.7 else "sell"
            elif recent_move < 0:
                return "sell" if random.random() < 0.7 else "buy"
            return "buy" if random.random() < 0.5 else "sell"

        elif self.strategy == "contrarian":
            if len(self._price_history) < 3:
                return "buy" if random.random() < 0.5 else "sell"
            recent_move = self._price_history[-1] - self._price_history[-3]
            if recent_move > 0:
                return "sell" if random.random() < 0.7 else "buy"
            elif recent_move < 0:
                return "buy"  if random.random() < 0.7 else "sell"
            return "buy" if random.random() < 0.5 else "sell"

        return "buy" if random.random() < 0.5 else "sell"

    def reset(self):
        super().reset()
        self._price_history = []

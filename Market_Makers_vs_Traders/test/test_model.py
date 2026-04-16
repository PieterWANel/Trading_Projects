"""
Unit Tests — Glosten-Milgrom Model
"""

import pytest
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.engine import GlostenMilgromModel, SimulationState
from src.players import InformedTrader, NoiseTrader
from src.simulation import Simulation


# ---------------------------------------------------------------------------
# Quote Tests
# ---------------------------------------------------------------------------

class TestQuotes:

    def setup_method(self):
        self.state = SimulationState(v_low=40, v_high=60, prob_v_high=0.5, mu=0.3)
        self.model = GlostenMilgromModel(self.state)

    def test_ask_above_bid(self):
        q = self.model.compute_quotes()
        assert q.ask > q.bid, "Ask must be above bid"

    def test_spread_positive(self):
        q = self.model.compute_quotes()
        assert q.spread > 0, "Spread must be positive when mu > 0"

    def test_zero_mu_gives_zero_spread(self):
        self.state.mu = 0.0
        q = self.model.compute_quotes()
        assert abs(q.spread) < 1e-6, "Zero informed traders => zero spread"

    def test_spread_increases_with_mu(self):
        spreads = []
        for mu in [0.1, 0.3, 0.5, 0.7]:
            self.state.mu = mu
            spreads.append(self.model.compute_quotes().spread)
        assert spreads == sorted(spreads), "Spread should increase monotonically with mu"

    def test_mid_equals_expected_value_at_prior(self):
        """At prior belief (0.5), mid should equal EV = 50."""
        self.state.mu = 0.0  # no adverse selection
        q = self.model.compute_quotes()
        ev = 0.5 * 60 + 0.5 * 40
        assert abs(q.mid - ev) < 0.01


# ---------------------------------------------------------------------------
# Bayesian Belief Tests
# ---------------------------------------------------------------------------

class TestBeliefUpdate:

    def setup_method(self):
        self.state = SimulationState(v_low=40, v_high=60, prob_v_high=0.5, mu=0.5,
                                     informed_strategy="aggressive")
        self.model = GlostenMilgromModel(self.state)

    def test_buy_increases_belief_high(self):
        initial = self.state.belief_high
        new_belief = self.model.update_belief("buy")
        assert new_belief > initial, "Buy order should increase P(V=V_H)"

    def test_sell_decreases_belief_high(self):
        initial = self.state.belief_high
        new_belief = self.model.update_belief("sell")
        assert new_belief < initial, "Sell order should decrease P(V=V_H)"

    def test_belief_bounded(self):
        for action in ["buy", "sell"]:
            b = self.model.update_belief(action)
            assert 0.0 <= b <= 1.0

    def test_repeated_buys_converge_to_one(self):
        """Repeated buy signals should drive P(V=V_H) toward 1."""
        for _ in range(30):
            self.state.belief_high = self.model.update_belief("buy")
        assert self.state.belief_high > 0.9


# ---------------------------------------------------------------------------
# Player Tests
# ---------------------------------------------------------------------------

class TestInformedTrader:

    def test_aggressive_buys_when_v_above_ask(self):
        from src.engine import Quote
        state = SimulationState(v_low=40, v_high=60, prob_v_high=0.5, mu=0.3)
        state.true_value = 60
        it = InformedTrader(strategy="aggressive")
        q  = Quote(bid=48.0, ask=52.0)
        assert it.act(state, q) == "buy"

    def test_aggressive_sells_when_v_below_bid(self):
        from src.engine import Quote
        state = SimulationState(v_low=40, v_high=60, prob_v_high=0.5, mu=0.3)
        state.true_value = 40
        it = InformedTrader(strategy="aggressive")
        q  = Quote(bid=48.0, ask=52.0)
        assert it.act(state, q) == "sell"

    def test_invalid_strategy_raises(self):
        with pytest.raises(AssertionError):
            InformedTrader(strategy="telepathy")


class TestNoiseTrader:

    def test_random_strategy_trades(self):
        from src.engine import Quote
        state = SimulationState()
        nt = NoiseTrader(strategy="random")
        q  = Quote(bid=49.0, ask=51.0)
        actions = {nt.act(state, q) for _ in range(50)}
        assert "buy" in actions and "sell" in actions


# ---------------------------------------------------------------------------
# Simulation Tests
# ---------------------------------------------------------------------------

class TestSimulation:

    def test_run_completes(self):
        sim = Simulation(n_rounds=50, seed=0)
        sim.run()
        assert len(sim.state.trades) == 50

    def test_mm_near_zero_pnl(self):
        """MM should break approximately even over many rounds."""
        sim = Simulation(n_rounds=500, seed=42)
        sim.run()
        # MM profit should be small relative to value range
        assert abs(sim.state.mm_total_pnl) < 200

    def test_informed_profitable_on_average(self):
        """Informed trader should profit on average."""
        total_it_pnl = 0
        for seed in range(10):
            sim = Simulation(n_rounds=100, seed=seed)
            sim.run()
            total_it_pnl += sim.state.it_total_pnl
        assert total_it_pnl > 0, "Informed trader should be profitable in aggregate"

    def test_noise_trader_loses(self):
        """Noise trader should lose to the spread on average."""
        total_nt_pnl = 0
        for seed in range(10):
            sim = Simulation(n_rounds=100, seed=seed)
            sim.run()
            total_nt_pnl += sim.state.nt_total_pnl
        assert total_nt_pnl < 0, "Noise trader should lose to bid-ask spread"

    def test_trades_df_shape(self):
        sim = Simulation(n_rounds=30, seed=1)
        sim.run()
        df = sim.trades_df()
        assert len(df) == 30
        assert "spread" in df.columns
        assert "belief_high" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""Tests for Strategy Builder & Simulator."""
import pytest
from services.strategies import list_strategies, simulate_strategy, build_strategy_legs, STRATEGY_TEMPLATES


class TestStrategyBuilder:
    def test_list_strategies(self):
        strategies = list_strategies()
        assert len(strategies) == len(STRATEGY_TEMPLATES)
        for s in strategies:
            assert "id" in s and "name" in s and "n_legs" in s

    def test_build_straddle_legs(self):
        legs = build_strategy_legs("straddle", S=100, K=100, T=1, r=0.05, sigma=0.2)
        assert len(legs) == 2
        types = [l["type"] for l in legs]
        assert "call" in types and "put" in types

    def test_build_iron_condor_legs(self):
        legs = build_strategy_legs("iron_condor", S=24000, K=24000, T=0.08, r=0.069, sigma=0.14)
        assert len(legs) == 4

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            build_strategy_legs("invalid", S=100, K=100, T=1)


class TestStrategySimulation:
    def test_straddle_v_shape(self):
        result = simulate_strategy("straddle", S=100, K=100, T=1, r=0.05, sigma=0.2)
        assert "spots" in result and "pnl" in result
        assert len(result["spots"]) == 100
        # Straddle: max loss at ATM, profit on both sides
        assert result["max_loss"] < 0
        assert result["breakevens"]  # should have 2 breakevens
        assert len(result["breakevens"]) == 2

    def test_long_call_payoff(self):
        result = simulate_strategy("long_call", S=100, K=100, T=1, r=0.05, sigma=0.2)
        # Max loss = premium paid
        assert result["max_loss"] < 0
        assert abs(result["max_loss"] + result["entry_premium"]) < 0.1
        assert result["max_profit"] > 0

    def test_bull_call_spread_capped(self):
        result = simulate_strategy("bull_call_spread", S=100, K=100, T=1, r=0.05, sigma=0.2)
        # Both profit and loss are capped
        assert result["max_profit"] < 100  # bounded
        assert result["max_loss"] > -100   # bounded

    def test_greeks_present(self):
        result = simulate_strategy("straddle", S=100, K=100, T=1, r=0.05, sigma=0.2)
        g = result["greeks"]
        assert "delta" in g and "gamma" in g and "vega" in g and "theta" in g and "rho" in g

    def test_strategy_inputs_returned(self):
        result = simulate_strategy("long_put", S=200, K=200, T=0.5, r=0.05, sigma=0.3)
        assert result["inputs"]["S"] == 200
        assert result["strategy"]["id"] == "long_put"

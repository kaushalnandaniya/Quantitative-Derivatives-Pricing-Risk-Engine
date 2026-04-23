"""
Strategy Builder & Simulator
===============================
Pre-built multi-leg option strategy templates with P&L simulation.

Strategies: Long Call, Long Put, Bull Call Spread, Bear Put Spread,
Straddle, Strangle, Iron Condor, Butterfly.
"""

import logging
import numpy as np
from typing import Dict, List

from pricing.black_scholes import black_scholes_price
from pricing.greeks import GreeksCalculator

logger = logging.getLogger(__name__)

STRATEGY_TEMPLATES = {
    "long_call": {
        "name": "Long Call", "description": "Bullish bet — unlimited upside, limited downside",
        "legs": [{"type": "call", "strike_offset": 0, "qty": 1}],
    },
    "long_put": {
        "name": "Long Put", "description": "Bearish bet — profit when price falls",
        "legs": [{"type": "put", "strike_offset": 0, "qty": 1}],
    },
    "bull_call_spread": {
        "name": "Bull Call Spread", "description": "Moderate bull — buy lower call, sell higher call",
        "legs": [
            {"type": "call", "strike_offset": -50, "qty": 1},
            {"type": "call", "strike_offset": 50, "qty": -1},
        ],
    },
    "bear_put_spread": {
        "name": "Bear Put Spread", "description": "Moderate bear — buy higher put, sell lower put",
        "legs": [
            {"type": "put", "strike_offset": 50, "qty": 1},
            {"type": "put", "strike_offset": -50, "qty": -1},
        ],
    },
    "straddle": {
        "name": "Long Straddle", "description": "Volatility play — profit on large moves either way",
        "legs": [
            {"type": "call", "strike_offset": 0, "qty": 1},
            {"type": "put", "strike_offset": 0, "qty": 1},
        ],
    },
    "strangle": {
        "name": "Long Strangle", "description": "Cheaper vol play — OTM call + OTM put",
        "legs": [
            {"type": "call", "strike_offset": 100, "qty": 1},
            {"type": "put", "strike_offset": -100, "qty": 1},
        ],
    },
    "iron_condor": {
        "name": "Iron Condor", "description": "Sell volatility — profit if price stays in range",
        "legs": [
            {"type": "put", "strike_offset": -200, "qty": 1},
            {"type": "put", "strike_offset": -100, "qty": -1},
            {"type": "call", "strike_offset": 100, "qty": -1},
            {"type": "call", "strike_offset": 200, "qty": 1},
        ],
    },
    "butterfly": {
        "name": "Butterfly Spread", "description": "Profit if price stays near strike at expiry",
        "legs": [
            {"type": "call", "strike_offset": -100, "qty": 1},
            {"type": "call", "strike_offset": 0, "qty": -2},
            {"type": "call", "strike_offset": 100, "qty": 1},
        ],
    },
}


def list_strategies() -> List[Dict]:
    """Return available strategy templates."""
    return [
        {"id": k, "name": v["name"], "description": v["description"],
         "n_legs": len(v["legs"])}
        for k, v in STRATEGY_TEMPLATES.items()
    ]


def build_strategy_legs(
    strategy_id: str, S: float, K: float, T: float,
    r: float = 0.05, sigma: float = 0.2, lot_size: int = 1,
) -> List[Dict]:
    """
    Generate position legs from a strategy template.

    Args:
        strategy_id: Key from STRATEGY_TEMPLATES.
        S: Spot price.
        K: Base strike price (ATM).
        T: Time to maturity (years).
        r: Risk-free rate.
        sigma: Volatility.
        lot_size: Number of lots.

    Returns:
        List of position dicts ready for pricing.
    """
    if strategy_id not in STRATEGY_TEMPLATES:
        raise ValueError(f"Unknown strategy '{strategy_id}'. Available: {list(STRATEGY_TEMPLATES.keys())}")

    template = STRATEGY_TEMPLATES[strategy_id]
    positions = []

    for leg in template["legs"]:
        leg_K = K + leg["strike_offset"]
        positions.append({
            "type": leg["type"],
            "S": S, "K": leg_K, "T": T, "r": r, "sigma": sigma,
            "qty": leg["qty"] * lot_size,
        })

    return positions


def simulate_strategy(
    strategy_id: str, S: float, K: float, T: float,
    r: float = 0.05, sigma: float = 0.2, lot_size: int = 1,
    n_points: int = 100,
) -> Dict:
    """
    Simulate P&L profile for a strategy across spot prices.

    Returns:
        Dictionary with spots, payoffs, greeks, max profit/loss, breakevens.
    """
    positions = build_strategy_legs(strategy_id, S, K, T, r, sigma, lot_size)
    template = STRATEGY_TEMPLATES[strategy_id]

    # Current cost (entry premium)
    total_premium = 0.0
    for pos in positions:
        price = float(black_scholes_price(pos["S"], pos["K"], T, r, sigma, pos["type"]))
        total_premium += price * pos["qty"]

    # Spot range for P&L
    spot_low = K * 0.7
    spot_high = K * 1.3
    spots = np.linspace(spot_low, spot_high, n_points)

    # P&L at expiry
    payoffs = np.zeros(n_points)
    for pos in positions:
        if pos["type"] == "call":
            leg_payoff = np.maximum(spots - pos["K"], 0.0) * pos["qty"]
        else:
            leg_payoff = np.maximum(pos["K"] - spots, 0.0) * pos["qty"]
        payoffs += leg_payoff

    # Net P&L = payoff at expiry - premium paid
    pnl = payoffs - total_premium

    # Max profit / max loss
    max_profit = float(np.max(pnl))
    max_loss = float(np.min(pnl))

    # Breakeven points
    breakevens = []
    for i in range(len(pnl) - 1):
        if pnl[i] * pnl[i+1] < 0:
            # Linear interpolation
            x = spots[i] + (spots[i+1] - spots[i]) * abs(pnl[i]) / (abs(pnl[i]) + abs(pnl[i+1]))
            breakevens.append(round(float(x), 2))

    # Current Greeks (aggregate)
    calc = GreeksCalculator(method="analytical")
    net_greeks = {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0, "rho": 0.0}
    for pos in positions:
        g = calc.calculate(S, pos["K"], T, r, sigma, pos["type"])
        for key in net_greeks:
            net_greeks[key] += float(np.asarray(g[key])) * pos["qty"]

    net_greeks = {k: round(v, 6) for k, v in net_greeks.items()}

    return {
        "strategy": {"id": strategy_id, "name": template["name"], "description": template["description"]},
        "legs": positions,
        "entry_premium": round(total_premium, 4),
        "spots": spots.tolist(),
        "pnl": pnl.tolist(),
        "max_profit": round(max_profit, 4),
        "max_loss": round(max_loss, 4),
        "breakevens": breakevens,
        "greeks": net_greeks,
        "inputs": {"S": S, "K": K, "T": T, "r": r, "sigma": sigma, "lot_size": lot_size},
    }

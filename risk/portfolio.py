"""
Portfolio Representation & Valuation
======================================
Multi-position portfolio system integrated with the pricing engine.

A portfolio is a collection of option positions, each defined by:
    type, strike, maturity, vol, rate, quantity, and exercise style.

Valuation is fully vectorized: pass an array of spot prices and get
back an array of portfolio values — critical for Monte Carlo P&L.
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Union
from copy import deepcopy

from pricing.black_scholes import black_scholes_price

logger = logging.getLogger(__name__)


class Portfolio:
    """
    Portfolio of European/American option positions.

    Each position is a dict with keys:
        type   : 'call' or 'put'
        S      : current spot price of the underlying
        K      : strike price
        T      : time to maturity (years)
        r      : risk-free rate
        sigma  : volatility
        qty    : quantity (positive = long, negative = short)
        asset  : (optional) asset identifier for multi-asset portfolios
        style  : (optional) 'european' or 'american' (default: 'european')

    Usage:
        port = Portfolio()
        port.add_position(type="call", S=100, K=100, T=1, r=0.05, sigma=0.2, qty=10)
        port.add_position(type="put",  S=100, K=90,  T=1, r=0.05, sigma=0.25, qty=5)
        v = port.value()  # current portfolio value
    """

    def __init__(self, positions: Optional[List[Dict]] = None):
        self._positions: List[Dict] = []
        if positions:
            for pos in positions:
                self.add_position(**pos)

    # -----------------------------------------------------------------
    # Position Management
    # -----------------------------------------------------------------

    def add_position(
        self,
        type: str,
        S: float,
        K: float,
        T: float,
        r: float = 0.05,
        sigma: float = 0.2,
        qty: int = 1,
        asset: str = "default",
        style: str = "european",
        **kwargs,
    ) -> None:
        """Add an option position to the portfolio."""
        otype = type.lower()
        if otype not in ("call", "put"):
            raise ValueError(f"type must be 'call' or 'put', got '{type}'")

        pos = {
            "type": otype,
            "S": float(S),
            "K": float(K),
            "T": float(T),
            "r": float(r),
            "sigma": float(sigma),
            "qty": int(qty),
            "asset": str(asset),
            "style": style.lower(),
        }
        self._positions.append(pos)
        logger.debug(
            f"Added {qty}x {otype} K={K} T={T:.3f} σ={sigma:.2f} (asset={asset})"
        )

    @property
    def positions(self) -> List[Dict]:
        """Return a deep copy of all positions."""
        return deepcopy(self._positions)

    @property
    def n_positions(self) -> int:
        return len(self._positions)

    def get_unique_assets(self) -> List[str]:
        """Return sorted list of unique asset identifiers."""
        return sorted(set(p["asset"] for p in self._positions))

    # -----------------------------------------------------------------
    # Valuation
    # -----------------------------------------------------------------

    def value(
        self,
        spot_overrides: Optional[Dict[str, Union[float, np.ndarray]]] = None,
        T_offset: float = 0.0,
    ) -> Union[float, np.ndarray]:
        """
        Compute total portfolio value.

        Args:
            spot_overrides: Dict mapping asset name -> spot price(s).
                           If a value is an ndarray of shape (n_sims,),
                           the returned value will also be (n_sims,).
                           If None, uses each position's stored S.
            T_offset:      Time elapsed since position inception (years).
                           Remaining maturity = max(T - T_offset, 0).

        Returns:
            Scalar or np.ndarray of portfolio values.
        """
        if not self._positions:
            raise ValueError("Portfolio has no positions")

        total = 0.0

        for pos in self._positions:
            # Determine spot price
            if spot_overrides and pos["asset"] in spot_overrides:
                S = spot_overrides[pos["asset"]]
            else:
                S = pos["S"]

            # Remaining maturity (floor at tiny positive for BS stability)
            T_remaining = max(pos["T"] - T_offset, 1e-10)

            # Price via Black-Scholes (vectorized if S is an array)
            price = black_scholes_price(
                S=S,
                K=pos["K"],
                T=T_remaining,
                r=pos["r"],
                sigma=pos["sigma"],
                option_type=pos["type"],
            )

            total = total + pos["qty"] * price

        return total

    def value_at_spots(
        self,
        spots: np.ndarray,
        T_offset: float = 0.0,
        asset: str = "default",
    ) -> np.ndarray:
        """
        Convenience: value the portfolio across an array of spot prices
        for a single-asset portfolio.

        Args:
            spots:    1-D array of spot prices, shape (n_sims,).
            T_offset: Time elapsed.
            asset:    Asset identifier to override.

        Returns:
            np.ndarray of shape (n_sims,) — portfolio values.
        """
        return self.value(
            spot_overrides={asset: spots},
            T_offset=T_offset,
        )

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def summary(self) -> List[Dict]:
        """Return a human-readable summary of all positions."""
        rows = []
        for i, pos in enumerate(self._positions):
            bs_val = float(
                black_scholes_price(
                    pos["S"], pos["K"], pos["T"], pos["r"], pos["sigma"], pos["type"]
                )
            )
            rows.append({
                "idx": i,
                "type": pos["type"],
                "asset": pos["asset"],
                "S": pos["S"],
                "K": pos["K"],
                "T": pos["T"],
                "sigma": pos["sigma"],
                "qty": pos["qty"],
                "unit_price": bs_val,
                "total_value": bs_val * pos["qty"],
            })
        return rows

    def __repr__(self) -> str:
        total = float(np.sum(self.value())) if self._positions else 0.0
        return (
            f"Portfolio({self.n_positions} positions, "
            f"value=${total:,.2f})"
        )

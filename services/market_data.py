"""
Market Data Service
====================
Dual-mode market data provider:
    - Mock mode: realistic NIFTY/BANKNIFTY option chain data
    - Live mode: Zerodha Kite API integration (requires credentials)
"""

import logging
import time
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from pricing.black_scholes import black_scholes_price
from pricing.greeks import GreeksCalculator

logger = logging.getLogger(__name__)


class MarketDataProvider(ABC):
    @abstractmethod
    def get_quote(self, symbol: str) -> Dict: ...
    @abstractmethod
    def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> Dict: ...
    @abstractmethod
    def get_status(self) -> Dict: ...


class MockMarketProvider(MarketDataProvider):
    """Generates realistic Indian market option chain data using BS pricing."""

    INSTRUMENTS = {
        "NIFTY": {"name": "NIFTY 50", "spot": 24050.60, "lot_size": 25, "strike_step": 50, "base_iv": 0.14, "iv_skew": 0.02},
        "BANKNIFTY": {"name": "BANK NIFTY", "spot": 51280.35, "lot_size": 15, "strike_step": 100, "base_iv": 0.16, "iv_skew": 0.025},
        "RELIANCE": {"name": "Reliance Industries", "spot": 2845.70, "lot_size": 250, "strike_step": 20, "base_iv": 0.22, "iv_skew": 0.015},
    }

    def __init__(self):
        self._calc = GreeksCalculator(method="analytical")
        rng = np.random.default_rng(int(time.time()) % 10000)
        self._jitter = {sym: float(rng.normal(0, 0.003)) for sym in self.INSTRUMENTS}

    def get_quote(self, symbol: str) -> Dict:
        symbol = symbol.upper()
        if symbol not in self.INSTRUMENTS:
            raise ValueError(f"Symbol '{symbol}' not found. Available: {list(self.INSTRUMENTS.keys())}")
        inst = self.INSTRUMENTS[symbol]
        jitter = self._jitter.get(symbol, 0)
        spot = inst["spot"] * (1 + jitter)
        return {
            "symbol": symbol, "name": inst["name"],
            "last_price": round(spot, 2),
            "change": round(spot * jitter, 2),
            "change_pct": round(jitter * 100, 2),
            "open": round(spot * (1 - abs(jitter) * 0.5), 2),
            "high": round(spot * (1 + abs(jitter) * 1.2), 2),
            "low": round(spot * (1 - abs(jitter) * 1.2), 2),
            "volume": int(np.random.default_rng(42).integers(5_000_000, 25_000_000)),
            "lot_size": inst["lot_size"],
            "timestamp": datetime.now().isoformat(),
            "provider": "mock",
        }

    def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> Dict:
        symbol = symbol.upper()
        if symbol not in self.INSTRUMENTS:
            raise ValueError(f"Symbol '{symbol}' not found.")
        inst = self.INSTRUMENTS[symbol]
        spot = inst["spot"] * (1 + self._jitter.get(symbol, 0))
        step, base_iv, skew, r = inst["strike_step"], inst["base_iv"], inst["iv_skew"], 0.069

        today = datetime.now()
        if symbol in ("NIFTY", "BANKNIFTY"):
            expiries = [today + timedelta(days=d) for d in [2, 9, 16, 30]]
        else:
            expiries = [today + timedelta(days=d) for d in [7, 30, 60]]
        expiry_labels = [e.strftime("%Y-%m-%d") for e in expiries]

        exp_date = expiries[0] if expiry is None else datetime.strptime(expiry, "%Y-%m-%d")
        if expiry is None:
            expiry = expiry_labels[0]
        T = max((exp_date - today).days / 365.0, 1/365.0)

        atm_strike = round(spot / step) * step
        n_strikes = 10
        strikes = [atm_strike + (i - n_strikes) * step for i in range(2 * n_strikes + 1)]

        chain = []
        for K in strikes:
            moneyness = np.log(spot / K)
            call_iv = max(base_iv + skew * max(-moneyness * 10, 0) + skew * abs(moneyness) * 2, 0.05)
            put_iv = max(base_iv + skew * max(moneyness * 10, 0) + skew * abs(moneyness) * 2, 0.05)

            call_price = float(black_scholes_price(spot, K, T, r, call_iv, "call"))
            put_price = float(black_scholes_price(spot, K, T, r, put_iv, "put"))
            cg = self._calc.calculate(spot, K, T, r, call_iv, "call")
            pg = self._calc.calculate(spot, K, T, r, put_iv, "put")

            rng = np.random.default_rng(int(K) + int(spot))
            oi_factor = max(1.0 - abs(K - atm_strike) / step * 0.08, 0.1)

            chain.append({
                "strike": float(K),
                "call": {"price": round(call_price, 2), "iv": round(call_iv * 100, 2),
                         "delta": round(float(cg["delta"]), 4), "gamma": round(float(cg["gamma"]), 6),
                         "theta": round(float(cg["theta"]), 4), "vega": round(float(cg["vega"]), 4),
                         "oi": int(rng.integers(50000, 500000) * oi_factor),
                         "volume": int(rng.integers(10000, 200000) * oi_factor)},
                "put": {"price": round(put_price, 2), "iv": round(put_iv * 100, 2),
                        "delta": round(float(pg["delta"]), 4), "gamma": round(float(pg["gamma"]), 6),
                        "theta": round(float(pg["theta"]), 4), "vega": round(float(pg["vega"]), 4),
                        "oi": int(rng.integers(50000, 500000) * oi_factor),
                        "volume": int(rng.integers(10000, 200000) * oi_factor)},
            })

        return {
            "symbol": symbol, "spot": round(spot, 2), "expiry": expiry,
            "expiries_available": expiry_labels, "T": round(T, 6),
            "chain": chain, "timestamp": datetime.now().isoformat(), "provider": "mock",
        }

    def get_status(self) -> Dict:
        return {"provider": "mock", "connected": True, "symbols": list(self.INSTRUMENTS.keys()),
                "message": "Mock data provider — realistic simulated market data"}


class KiteMarketProvider(MarketDataProvider):
    """Live market data via Zerodha Kite Connect API."""

    def __init__(self, api_key: str = "", access_token: str = ""):
        self._kite = None
        if api_key and access_token:
            try:
                from kiteconnect import KiteConnect
                self._kite = KiteConnect(api_key=api_key)
                self._kite.set_access_token(access_token)
                logger.info("Kite Connect initialized")
            except ImportError:
                logger.warning("kiteconnect not installed")
            except Exception as e:
                logger.error(f"Kite init failed: {e}")

    def get_quote(self, symbol: str) -> Dict:
        if not self._kite:
            raise ConnectionError("Kite not initialized")
        data = self._kite.quote(f"NSE:{symbol}")
        q = data[f"NSE:{symbol}"]
        return {"symbol": symbol, "last_price": q["last_price"], "change": q["net_change"],
                "provider": "kite", "timestamp": datetime.now().isoformat()}

    def get_option_chain(self, symbol: str, expiry: Optional[str] = None) -> Dict:
        raise NotImplementedError("Live option chain requires NFO instrument filtering")

    def get_status(self) -> Dict:
        connected = self._kite is not None
        return {"provider": "kite", "connected": connected,
                "message": "Connected" if connected else "Not connected"}


_provider_instance: Optional[MarketDataProvider] = None

def create_provider(mode: str = "mock", **kwargs) -> MarketDataProvider:
    global _provider_instance
    if mode == "mock":
        _provider_instance = MockMarketProvider()
    elif mode == "kite":
        _provider_instance = KiteMarketProvider(**kwargs)
    else:
        raise ValueError(f"mode must be 'mock' or 'kite', got '{mode}'")
    return _provider_instance

def get_provider() -> MarketDataProvider:
    global _provider_instance
    if _provider_instance is None:
        _provider_instance = MockMarketProvider()
    return _provider_instance

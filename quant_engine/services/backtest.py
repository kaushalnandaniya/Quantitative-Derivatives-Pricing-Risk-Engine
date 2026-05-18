"""
Backtesting Service
====================
Backtest option strategies against historical spot data.
Uses yfinance for historical prices and Black-Scholes for
reconstructed option pricing at each historical entry point.
"""

import logging
import numpy as np
from typing import Dict, List
from datetime import datetime, timedelta

from pricing.black_scholes import black_scholes_price
from services.strategies import STRATEGY_TEMPLATES, build_strategy_legs

logger = logging.getLogger(__name__)


def _get_historical_spots(symbol: str, lookback_weeks: int = 12) -> List[Dict]:
    """Fetch historical weekly closing prices."""
    try:
        import yfinance as yf
        yf_map = {"NIFTY": "^NSEI", "BANKNIFTY": "^NSEBANK"}
        yf_sym = yf_map.get(symbol, f"{symbol}.NS")
        ticker = yf.Ticker(yf_sym)
        hist = ticker.history(period=f"{lookback_weeks * 7 + 30}d")
        
        if hist.empty:
            return _generate_mock_spots(symbol, lookback_weeks)
        
        # Sample weekly (every 5 trading days)
        weekly = hist.iloc[::5]
        results = []
        for i in range(len(weekly) - 1):
            results.append({
                "entry_date": weekly.index[i].strftime("%Y-%m-%d"),
                "entry_spot": round(float(weekly.iloc[i]["Close"]), 2),
                "expiry_date": weekly.index[i + 1].strftime("%Y-%m-%d"),
                "expiry_spot": round(float(weekly.iloc[i + 1]["Close"]), 2),
            })
        return results[-lookback_weeks:]
    except Exception as e:
        logger.warning(f"yfinance failed: {e}, using mock data")
        return _generate_mock_spots(symbol, lookback_weeks)


def _generate_mock_spots(symbol: str, weeks: int) -> List[Dict]:
    """Generate realistic mock weekly spot data."""
    base = {"NIFTY": 24000, "BANKNIFTY": 51000, "RELIANCE": 2800}.get(symbol, 20000)
    rng = np.random.default_rng(42)
    spots = [base]
    for _ in range(weeks):
        change = rng.normal(0.001, 0.025)
        spots.append(round(spots[-1] * (1 + change), 2))
    
    today = datetime.now()
    results = []
    for i in range(weeks):
        entry = today - timedelta(weeks=weeks - i)
        expiry = entry + timedelta(days=7)
        results.append({
            "entry_date": entry.strftime("%Y-%m-%d"),
            "entry_spot": spots[i],
            "expiry_date": expiry.strftime("%Y-%m-%d"),
            "expiry_spot": spots[i + 1],
        })
    return results


def run_backtest(
    strategy_id: str,
    symbol: str = "NIFTY",
    lookback_weeks: int = 12,
    sigma: float = 0.15,
    r: float = 0.069,
    lot_size: int = 1,
) -> Dict:
    """
    Backtest a strategy over historical weekly expiries.
    
    For each week:
      1. Get entry spot and expiry spot from historical data
      2. Build strategy legs at ATM using BS pricing with estimated IV
      3. Compute entry premium
      4. Compute P&L at expiry using intrinsic value
    """
    if strategy_id not in STRATEGY_TEMPLATES:
        raise ValueError(f"Unknown strategy: {strategy_id}")
    
    template = STRATEGY_TEMPLATES[strategy_id]
    historical = _get_historical_spots(symbol, lookback_weeks)
    
    if not historical:
        return {"error": "No historical data available"}
    
    results = []
    T = 7 / 365  # Weekly expiry
    
    for week in historical:
        S = week["entry_spot"]
        K = S  # ATM
        expiry_S = week["expiry_spot"]
        
        # Build legs and compute entry premium
        positions = build_strategy_legs(strategy_id, S, K, T, r, sigma, lot_size)
        entry_premium = 0.0
        for pos in positions:
            price = float(black_scholes_price(pos["S"], pos["K"], T, r, sigma, pos["type"]))
            entry_premium += price * pos["qty"]
        
        # Compute P&L at expiry (intrinsic value)
        payoff = 0.0
        for pos in positions:
            if pos["type"] == "call":
                intrinsic = max(expiry_S - pos["K"], 0)
            else:
                intrinsic = max(pos["K"] - expiry_S, 0)
            payoff += intrinsic * pos["qty"]
        
        pnl = round(payoff - entry_premium, 2)
        spot_change = round(((expiry_S - S) / S) * 100, 2)
        
        results.append({
            "entry_date": week["entry_date"],
            "expiry_date": week["expiry_date"],
            "entry_spot": S,
            "expiry_spot": expiry_S,
            "spot_change_pct": spot_change,
            "entry_premium": round(entry_premium, 2),
            "pnl": pnl,
            "win": pnl > 0,
        })
    
    # Summary stats
    pnls = [r["pnl"] for r in results]
    wins = sum(1 for p in pnls if p > 0)
    total = len(pnls)
    avg_pnl = round(np.mean(pnls), 2) if pnls else 0
    total_pnl = round(sum(pnls), 2)
    max_drawdown = round(min(pnls), 2) if pnls else 0
    best_trade = round(max(pnls), 2) if pnls else 0
    
    # Equity curve
    equity = [0.0]
    for p in pnls:
        equity.append(round(equity[-1] + p, 2))
    
    return {
        "strategy": {"id": strategy_id, "name": template["name"]},
        "symbol": symbol,
        "lookback_weeks": lookback_weeks,
        "results": results,
        "summary": {
            "total_trades": total,
            "wins": wins,
            "losses": total - wins,
            "win_rate": round(wins / max(total, 1) * 100, 1),
            "avg_pnl": avg_pnl,
            "total_pnl": total_pnl,
            "best_trade": best_trade,
            "worst_trade": max_drawdown,
        },
        "equity_curve": equity,
    }

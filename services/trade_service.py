"""
Trade Service
==============
Trade booking, position tracking, and P&L calculation.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from db.models import Trade, TradeStatus, TradeSide, OptionType
from pricing.black_scholes import black_scholes_price

logger = logging.getLogger(__name__)


def book_trade(
    db: Session, user_id: str, side: str, option_type: str,
    spot: float, strike: float, T: float, r: float, sigma: float,
    quantity: int, portfolio_id: str = None, notes: str = None,
) -> Trade:
    """Book a new trade — auto-calculates entry premium via Black-Scholes."""
    premium = float(black_scholes_price(spot, strike, T, r, sigma, option_type))

    trade = Trade(
        user_id=user_id,
        portfolio_id=portfolio_id,
        side=TradeSide(side),
        option_type=OptionType(option_type),
        spot_at_entry=spot,
        strike=strike,
        premium=premium,
        quantity=quantity,
        sigma_at_entry=sigma,
        T_at_entry=T,
        r_at_entry=r,
        status=TradeStatus.open,
        notes=notes,
    )
    db.add(trade)
    db.commit()
    db.refresh(trade)
    logger.info(f"Trade booked: {side} {quantity}x {option_type} K={strike} @ {premium:.4f}")
    return trade


def get_user_trades(
    db: Session, user_id: str, status: str = None, limit: int = 100,
) -> List[Trade]:
    """Get user's trades with optional status filter."""
    q = db.query(Trade).filter(Trade.user_id == user_id)
    if status:
        q = q.filter(Trade.status == TradeStatus(status))
    return q.order_by(Trade.traded_at.desc()).limit(limit).all()


def get_trade(db: Session, trade_id: str, user_id: str) -> Optional[Trade]:
    return db.query(Trade).filter(Trade.id == trade_id, Trade.user_id == user_id).first()


def close_trade(db: Session, trade: Trade, close_premium: float = None) -> Trade:
    """Close an open trade. Auto-calculates closing premium if not provided."""
    if trade.status != TradeStatus.open:
        raise ValueError(f"Trade {trade.id} is already {trade.status.value}")

    if close_premium is None:
        # Auto-calculate current premium (approximate — T reduced by elapsed time)
        elapsed_years = (datetime.now(timezone.utc) - trade.traded_at).total_seconds() / (365.25 * 86400)
        remaining_T = max(trade.T_at_entry - elapsed_years, 1e-6)
        close_premium = float(black_scholes_price(
            trade.spot_at_entry, trade.strike, remaining_T,
            trade.r_at_entry, trade.sigma_at_entry, trade.option_type.value,
        ))

    trade.close_premium = close_premium
    trade.closed_at = datetime.now(timezone.utc)
    trade.status = TradeStatus.closed
    db.commit()
    db.refresh(trade)
    logger.info(f"Trade closed: {trade.id} @ {close_premium:.4f}")
    return trade


def get_position_summary(db: Session, user_id: str) -> dict:
    """Aggregate open positions by option type and strike."""
    open_trades = get_user_trades(db, user_id, status="open")

    positions = {}
    total_notional = 0.0

    for t in open_trades:
        key = f"{t.option_type.value}_{t.strike}"
        if key not in positions:
            positions[key] = {
                "option_type": t.option_type.value,
                "strike": t.strike,
                "net_qty": 0,
                "avg_premium": 0.0,
                "total_premium": 0.0,
            }

        direction = 1 if t.side == TradeSide.buy else -1
        positions[key]["net_qty"] += t.quantity * direction
        positions[key]["total_premium"] += t.premium * t.quantity * direction
        total_notional += t.premium * t.quantity

    # Compute average premium
    for p in positions.values():
        if p["net_qty"] != 0:
            p["avg_premium"] = round(abs(p["total_premium"] / p["net_qty"]), 4)
        p["total_premium"] = round(p["total_premium"], 4)

    return {
        "n_open_trades": len(open_trades),
        "n_positions": len(positions),
        "total_notional": round(total_notional, 4),
        "positions": list(positions.values()),
    }

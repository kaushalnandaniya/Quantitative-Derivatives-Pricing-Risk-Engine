"""
Trade API Routes
==================
Trade booking, blotter, and position summary.
"""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from api.middleware.auth import get_current_user
from schemas.portfolio import TradeBookRequest, TradeCloseRequest
from services.trade_service import book_trade, get_user_trades, get_trade, close_trade, get_position_summary

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/trades", tags=["Trades"])


@router.post("", status_code=201, summary="Book Trade")
def create_trade(data: TradeBookRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    trade = book_trade(
        db, user.id, data.side, data.option_type,
        data.spot, data.strike, data.T, data.r, data.sigma,
        data.quantity, data.portfolio_id, data.notes,
    )
    return _to_response(trade)


@router.get("", summary="Trade Blotter")
def list_trades(
    status: Optional[str] = Query(None, description="Filter: open, closed, expired"),
    limit: int = Query(100, ge=1, le=1000),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    trades = get_user_trades(db, user.id, status, limit)
    return {"trades": [_to_response(t) for t in trades], "count": len(trades)}


@router.get("/positions", summary="Position Summary")
def positions(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return get_position_summary(db, user.id)


@router.get("/{trade_id}", summary="Trade Detail")
def get_one(trade_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    trade = get_trade(db, trade_id, user.id)
    if not trade:
        raise HTTPException(404, "Trade not found")
    return _to_response(trade)


@router.put("/{trade_id}/close", summary="Close Trade")
def close(trade_id: str, data: TradeCloseRequest = None, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    trade = get_trade(db, trade_id, user.id)
    if not trade:
        raise HTTPException(404, "Trade not found")
    try:
        close_premium = data.close_premium if data else None
        trade = close_trade(db, trade, close_premium)
        return _to_response(trade)
    except ValueError as e:
        raise HTTPException(400, str(e))


def _to_response(t) -> dict:
    return {
        "id": t.id, "side": t.side.value, "option_type": t.option_type.value,
        "spot_at_entry": t.spot_at_entry, "strike": t.strike,
        "premium": round(t.premium, 4), "quantity": t.quantity,
        "sigma_at_entry": t.sigma_at_entry, "T_at_entry": t.T_at_entry,
        "status": t.status.value, "traded_at": t.traded_at.isoformat(),
        "closed_at": t.closed_at.isoformat() if t.closed_at else None,
        "close_premium": round(t.close_premium, 4) if t.close_premium else None,
        "notes": t.notes, "portfolio_id": t.portfolio_id,
        "notional": round(t.premium * t.quantity, 4),
    }

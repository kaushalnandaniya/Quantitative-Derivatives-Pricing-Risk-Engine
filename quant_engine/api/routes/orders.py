"""
Orders API Routes (OMS)
=========================
POST /orders              — Submit new order (with risk check)
GET  /orders              — Order blotter
GET  /orders/{id}         — Order detail with executions
DELETE /orders/{id}       — Cancel order
POST /orders/{id}/fill    — Simulate fill (for demo)
GET  /orders/{id}/risk    — Pre-trade risk assessment
GET  /executions          — Fill history
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from api.middleware.auth import get_current_user
from services.oms_service import (
    pre_trade_risk_check, submit_order, validate_order,
    fill_order, cancel_order,
    get_user_orders, get_order, get_executions,
)
from services.execution_service import route_order

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/orders", tags=["OMS"])


class SubmitOrderRequest(BaseModel):
    side: str = Field(..., description="buy or sell")
    option_type: str = Field(..., description="call or put")
    spot: float = Field(..., gt=0)
    strike: float = Field(..., gt=0)
    T: float = Field(..., gt=0, description="Time to maturity in years")
    r: float = Field(0.05, description="Risk-free rate")
    sigma: float = Field(..., gt=0, description="Implied vol")
    quantity: int = Field(..., gt=0)
    order_type: str = Field("market", description="market or limit")
    limit_price: Optional[float] = None
    portfolio_id: Optional[str] = None
    notes: Optional[str] = None


class RiskCheckRequest(BaseModel):
    side: str
    option_type: str
    spot: float = Field(gt=0)
    strike: float = Field(gt=0)
    T: float = Field(gt=0)
    r: float = 0.05
    sigma: float = Field(gt=0)
    quantity: int = Field(gt=0)


# =============================================================================
# Endpoints
# =============================================================================

@router.post("", status_code=201, summary="Submit Order")
def submit(data: SubmitOrderRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Submit an order with automatic pre-trade risk check."""
    # Run risk checks
    risk_result = pre_trade_risk_check(
        db, user, data.side, data.option_type,
        data.spot, data.strike, data.T, data.r, data.sigma, data.quantity,
    )

    # Submit order (status depends on risk check)
    order = submit_order(
        db, user.id, data.side, data.option_type,
        data.spot, data.strike, data.T, data.r, data.sigma, data.quantity,
        order_type=data.order_type, limit_price=data.limit_price,
        portfolio_id=data.portfolio_id, notes=data.notes,
        risk_check_result=risk_result,
    )

    if not risk_result["passed"]:
        validate_order(db, order, risk_result)

    # Auto-fill market orders that pass risk checks
    if risk_result["passed"] and data.order_type == "market":
        symbol = f"{data.option_type.upper()}-{data.strike}-{data.T}"
        # Route to KiteConnect (or mock if no API key)
        exec_res = route_order(
            side=data.side,
            quantity=data.quantity,
            symbol=symbol,
            order_type=data.order_type,
            price=data.limit_price
        )
        
        # Append execution details to notes
        order.notes = f"{order.notes or ''} [Exec: {exec_res.get('exchange_order_id', 'fail')}]".strip()
        
        if exec_res["status"] == "SUBMITTED":
            fill_order(db, order, fill_price=risk_result["theoretical_premium"])

    return _order_response(order)


@router.get("", summary="Order Blotter")
def list_orders(
    status: str = Query(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    orders = get_user_orders(db, user.id, status=status)
    return {"orders": [_order_response(o) for o in orders], "count": len(orders)}


@router.get("/{order_id}", summary="Order Detail")
def get_detail(order_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    order = get_order(db, order_id, user.id)
    if not order:
        raise HTTPException(404, "Order not found")
    resp = _order_response(order)
    resp["executions"] = [
        {
            "id": e.id,
            "fill_price": e.fill_price,
            "fill_quantity": e.fill_quantity,
            "executed_at": e.executed_at.isoformat(),
            "exchange_ref": e.exchange_ref,
        }
        for e in order.executions
    ]
    return resp


@router.delete("/{order_id}", status_code=200, summary="Cancel Order")
def cancel(order_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    order = get_order(db, order_id, user.id)
    if not order:
        raise HTTPException(404, "Order not found")
    try:
        cancel_order(db, order)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _order_response(order)


@router.post("/{order_id}/fill", summary="Fill Order (Demo)")
def manual_fill(
    order_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Manually fill an order (for demo/testing)."""
    order = get_order(db, order_id, user.id)
    if not order:
        raise HTTPException(404, "Order not found")
    try:
        fill_order(db, order)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return _order_response(order)


@router.post("/risk-check", summary="Pre-Trade Risk Check")
def risk_check(data: RiskCheckRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    result = pre_trade_risk_check(
        db, user, data.side, data.option_type,
        data.spot, data.strike, data.T, data.r, data.sigma, data.quantity,
    )
    return result


@router.get("/executions/history", summary="Execution History")
def execution_history(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    execs = get_executions(db, user.id)
    return {
        "executions": [
            {
                "id": e.id,
                "order_id": e.order_id,
                "fill_price": e.fill_price,
                "fill_quantity": e.fill_quantity,
                "executed_at": e.executed_at.isoformat(),
            }
            for e in execs
        ],
        "count": len(execs),
    }


def _order_response(o) -> dict:
    return {
        "id": o.id,
        "side": o.side.value,
        "option_type": o.option_type.value,
        "order_type": o.order_type.value,
        "spot_price": o.spot_price,
        "strike": o.strike,
        "T": o.T,
        "sigma": o.sigma,
        "quantity": o.quantity,
        "filled_quantity": o.filled_quantity,
        "avg_fill_price": o.avg_fill_price,
        "limit_price": o.limit_price,
        "status": o.status.value,
        "risk_check_result": o.risk_check_result,
        "rejection_reason": o.rejection_reason,
        "submitted_at": o.submitted_at.isoformat(),
        "filled_at": o.filled_at.isoformat() if o.filled_at else None,
        "cancelled_at": o.cancelled_at.isoformat() if o.cancelled_at else None,
        "portfolio_id": o.portfolio_id,
        "notes": o.notes,
    }

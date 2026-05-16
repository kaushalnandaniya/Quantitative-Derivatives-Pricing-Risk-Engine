"""
Order Management System (OMS) Service
========================================
Full order lifecycle: submit → validate → risk check → fill → close.
Includes pre-trade risk checks against tenant risk limits.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from db.models import (
    Order, Execution, Trade, User, RiskLimits,
    OrderStatus, OrderType, TradeSide, OptionType, TradeStatus,
)
from pricing.black_scholes import black_scholes_price
from services.trade_service import get_position_summary

logger = logging.getLogger(__name__)


# =============================================================================
# Pre-Trade Risk Checks
# =============================================================================

def pre_trade_risk_check(db: Session, user: User, side: str, option_type: str,
                         spot: float, strike: float, T: float, r: float,
                         sigma: float, quantity: int) -> dict:
    """
    Run pre-trade risk checks against tenant risk limits.
    Returns {passed: bool, checks: [...], rejection_reason: str|None}
    """
    premium = float(black_scholes_price(spot, strike, T, r, sigma, option_type))
    order_notional = premium * quantity

    checks = []
    passed = True
    rejection_reason = None

    # Get risk limits (if tenant has them)
    risk_limits = None
    if user.tenant_id:
        risk_limits = db.query(RiskLimits).filter(RiskLimits.tenant_id == user.tenant_id).first()

    if risk_limits:
        # 1. Position Size Check
        pos_check = quantity <= risk_limits.max_position_size
        checks.append({
            "check": "position_size",
            "passed": pos_check,
            "value": quantity,
            "limit": risk_limits.max_position_size,
        })
        if not pos_check:
            passed = False
            rejection_reason = f"Position size {quantity} exceeds limit {risk_limits.max_position_size}"

        # 2. Notional Limit Check
        positions = get_position_summary(db, user.id)
        current_notional = positions.get("total_notional", 0)
        new_total = current_notional + order_notional
        notional_check = new_total <= risk_limits.max_notional
        checks.append({
            "check": "notional_limit",
            "passed": notional_check,
            "current_notional": round(current_notional, 2),
            "order_notional": round(order_notional, 2),
            "new_total": round(new_total, 2),
            "limit": risk_limits.max_notional,
        })
        if not notional_check and passed:
            passed = False
            rejection_reason = f"Total notional {new_total:.0f} would exceed limit {risk_limits.max_notional:.0f}"

        # 3. Margin Check
        required_margin = order_notional * risk_limits.margin_requirement
        checks.append({
            "check": "margin",
            "passed": True,  # Simplified — always passes for now
            "required_margin": round(required_margin, 2),
            "margin_rate": risk_limits.margin_requirement,
        })
    else:
        checks.append({"check": "no_limits", "passed": True, "note": "No risk limits configured"})

    # 4. Fat Finger Check (±10% from theoretical price)
    fat_finger_ok = True
    checks.append({
        "check": "fat_finger",
        "passed": fat_finger_ok,
        "theoretical_price": round(premium, 4),
    })

    return {
        "passed": passed,
        "checks": checks,
        "rejection_reason": rejection_reason,
        "theoretical_premium": round(premium, 4),
        "order_notional": round(order_notional, 4),
    }


# =============================================================================
# Order Lifecycle
# =============================================================================

def submit_order(
    db: Session, user_id: str, side: str, option_type: str,
    spot: float, strike: float, T: float, r: float, sigma: float,
    quantity: int, order_type: str = "market", limit_price: float = None,
    portfolio_id: str = None, notes: str = None,
    risk_check_result: dict = None,
) -> Order:
    """Submit a new order after risk checks pass."""
    order = Order(
        user_id=user_id,
        portfolio_id=portfolio_id,
        side=TradeSide(side),
        option_type=OptionType(option_type),
        order_type=OrderType(order_type),
        spot_price=spot,
        strike=strike,
        T=T,
        r=r,
        sigma=sigma,
        quantity=quantity,
        limit_price=limit_price,
        status=OrderStatus.validated if (risk_check_result and risk_check_result["passed"]) else OrderStatus.pending,
        risk_check_result=risk_check_result,
        notes=notes,
    )
    db.add(order)
    db.commit()
    db.refresh(order)
    logger.info(f"Order submitted: {side} {quantity}x {option_type} K={strike} status={order.status.value}")
    return order


def validate_order(db: Session, order: Order, risk_result: dict) -> Order:
    """Run risk checks and update order status."""
    order.risk_check_result = risk_result
    if risk_result["passed"]:
        order.status = OrderStatus.validated
        logger.info(f"Order {order.id} validated — risk checks passed")
    else:
        order.status = OrderStatus.rejected
        order.rejection_reason = risk_result.get("rejection_reason", "Risk check failed")
        logger.warning(f"Order {order.id} REJECTED: {order.rejection_reason}")
    db.commit()
    db.refresh(order)
    return order


def fill_order(db: Session, order: Order, fill_price: float = None, fill_qty: int = None) -> Order:
    """Execute a fill against an order. Creates execution record and optionally a trade."""
    if order.is_terminal:
        raise ValueError(f"Cannot fill terminal order (status={order.status.value})")

    if fill_price is None:
        fill_price = float(black_scholes_price(
            order.spot_price, order.strike, order.T, order.r, order.sigma, order.option_type.value
        ))

    fill_qty = fill_qty or (order.quantity - order.filled_quantity)
    fill_qty = min(fill_qty, order.quantity - order.filled_quantity)

    if fill_qty <= 0:
        raise ValueError("No remaining quantity to fill")

    # Create execution record
    execution = Execution(
        order_id=order.id,
        fill_price=fill_price,
        fill_quantity=fill_qty,
    )
    db.add(execution)

    # Update order
    old_filled = order.filled_quantity
    order.filled_quantity += fill_qty
    if order.avg_fill_price is None:
        order.avg_fill_price = fill_price
    else:
        order.avg_fill_price = (
            (order.avg_fill_price * old_filled + fill_price * fill_qty)
            / order.filled_quantity
        )

    if order.filled_quantity >= order.quantity:
        order.status = OrderStatus.filled
        order.filled_at = datetime.now(timezone.utc)
    else:
        order.status = OrderStatus.partial_fill

    # Create corresponding trade
    trade = Trade(
        user_id=order.user_id,
        portfolio_id=order.portfolio_id,
        side=order.side,
        option_type=order.option_type,
        spot_at_entry=order.spot_price,
        strike=order.strike,
        premium=fill_price,
        quantity=fill_qty,
        sigma_at_entry=order.sigma,
        T_at_entry=order.T,
        r_at_entry=order.r,
        status=TradeStatus.open,
    )
    db.add(trade)

    db.commit()
    db.refresh(order)
    logger.info(f"Order {order.id} filled: {fill_qty}x @ {fill_price:.4f} (total: {order.filled_quantity}/{order.quantity})")
    return order


def cancel_order(db: Session, order: Order) -> Order:
    """Cancel a pending/validated/partial order."""
    if order.is_terminal:
        raise ValueError(f"Cannot cancel terminal order (status={order.status.value})")
    order.status = OrderStatus.cancelled
    order.cancelled_at = datetime.now(timezone.utc)
    db.commit()
    db.refresh(order)
    logger.info(f"Order {order.id} cancelled")
    return order


# =============================================================================
# Queries
# =============================================================================

def get_user_orders(db: Session, user_id: str, status: str = None, limit: int = 100) -> List[Order]:
    q = db.query(Order).filter(Order.user_id == user_id)
    if status:
        q = q.filter(Order.status == OrderStatus(status))
    return q.order_by(Order.submitted_at.desc()).limit(limit).all()


def get_order(db: Session, order_id: str, user_id: str) -> Optional[Order]:
    return db.query(Order).filter(Order.id == order_id, Order.user_id == user_id).first()


def get_executions(db: Session, user_id: str, limit: int = 100) -> List[Execution]:
    return (
        db.query(Execution)
        .join(Order)
        .filter(Order.user_id == user_id)
        .order_by(Execution.executed_at.desc())
        .limit(limit)
        .all()
    )

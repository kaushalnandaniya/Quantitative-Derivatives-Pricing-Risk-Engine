"""
Alert Service
===============
Alert rules engine for VaR breach, price trigger, expiry warning, and margin call detection.
"""

import logging
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy.orm import Session

from db.models import Alert, AlertType

logger = logging.getLogger(__name__)


# =============================================================================
# CRUD
# =============================================================================

def create_alert(
    db: Session, user_id: str, alert_type: str,
    condition: dict, portfolio_id: str = None, message: str = None,
) -> Alert:
    """Create a new alert rule."""
    alert = Alert(
        user_id=user_id,
        portfolio_id=portfolio_id,
        alert_type=AlertType(alert_type),
        condition=condition,
        message=message,
    )
    db.add(alert)
    db.commit()
    db.refresh(alert)
    logger.info(f"Alert created: {alert_type} for user={user_id}")
    return alert


def get_user_alerts(db: Session, user_id: str, active_only: bool = True) -> List[Alert]:
    q = db.query(Alert).filter(Alert.user_id == user_id)
    if active_only:
        q = q.filter(Alert.is_active == True)
    return q.order_by(Alert.created_at.desc()).all()


def get_alert(db: Session, alert_id: str, user_id: str) -> Optional[Alert]:
    return db.query(Alert).filter(
        Alert.id == alert_id, Alert.user_id == user_id
    ).first()


def delete_alert(db: Session, alert: Alert):
    alert.is_active = False
    db.commit()


def trigger_alert(db: Session, alert: Alert, message: str = None):
    """Mark an alert as triggered."""
    alert.triggered = True
    alert.triggered_at = datetime.now(timezone.utc)
    if message:
        alert.message = message
    db.commit()
    db.refresh(alert)
    logger.warning(f"Alert TRIGGERED: {alert.alert_type.value} id={alert.id}")
    return alert


# =============================================================================
# Evaluation Engine
# =============================================================================

def evaluate_var_alert(alert: Alert, current_var: float) -> bool:
    """Check if VaR exceeds the alert threshold."""
    threshold = alert.condition.get("threshold", 0)
    return abs(current_var) >= abs(threshold)


def evaluate_price_alert(alert: Alert, current_price: float) -> bool:
    """Check if price crossed the trigger level."""
    target = alert.condition.get("target_price", 0)
    direction = alert.condition.get("direction", "above")  # "above" or "below"
    if direction == "above":
        return current_price >= target
    return current_price <= target


def evaluate_expiry_alert(alert: Alert, days_to_expiry: float) -> bool:
    """Check if position is approaching expiry."""
    warning_days = alert.condition.get("warning_days", 3)
    return days_to_expiry <= warning_days


def check_alerts(
    db: Session, user_id: str,
    current_var: float = None,
    current_prices: dict = None,
    days_to_expiry: dict = None,
) -> List[dict]:
    """
    Evaluate all active alerts for a user and trigger any that match.
    Returns list of newly triggered alerts.
    """
    alerts = get_user_alerts(db, user_id, active_only=True)
    triggered = []

    for alert in alerts:
        if alert.triggered:
            continue

        should_trigger = False
        msg = ""

        if alert.alert_type == AlertType.var_breach and current_var is not None:
            should_trigger = evaluate_var_alert(alert, current_var)
            msg = f"VaR breach: |{current_var:.2f}| exceeded threshold |{alert.condition.get('threshold', 0):.2f}|"

        elif alert.alert_type == AlertType.price_trigger and current_prices:
            symbol = alert.condition.get("symbol", "")
            price = current_prices.get(symbol)
            if price is not None:
                should_trigger = evaluate_price_alert(alert, price)
                msg = f"Price trigger: {symbol} at {price:.2f} (target: {alert.condition.get('target_price'):.2f})"

        elif alert.alert_type == AlertType.expiry_warning and days_to_expiry:
            position_id = alert.condition.get("position_id", "")
            dte = days_to_expiry.get(position_id)
            if dte is not None:
                should_trigger = evaluate_expiry_alert(alert, dte)
                msg = f"Expiry warning: Position {position_id} expires in {dte:.1f} days"

        if should_trigger:
            trigger_alert(db, alert, msg)
            triggered.append({
                "id": alert.id,
                "type": alert.alert_type.value,
                "message": msg,
                "triggered_at": alert.triggered_at.isoformat(),
            })

    return triggered

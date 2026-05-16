"""
Alerts API Routes
==================
POST /alerts           — Create alert rule
GET  /alerts           — List user's alerts
GET  /alerts/{id}      — Get alert detail
DELETE /alerts/{id}    — Deactivate alert
POST /alerts/evaluate  — Evaluate all alerts against current data
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from api.middleware.auth import get_current_user
from services.alert_service import (
    create_alert, get_user_alerts, get_alert,
    delete_alert, check_alerts,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/alerts", tags=["Alerts"])


class CreateAlertRequest(BaseModel):
    alert_type: str = Field(..., description="var_breach | price_trigger | expiry_warning | margin_call")
    condition: dict = Field(..., description="Alert conditions (threshold, target_price, direction, etc.)")
    portfolio_id: Optional[str] = None
    message: Optional[str] = None


class EvaluateRequest(BaseModel):
    current_var: Optional[float] = None
    current_prices: Optional[dict] = None
    days_to_expiry: Optional[dict] = None


@router.post("", status_code=201, summary="Create Alert")
def create(data: CreateAlertRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    alert = create_alert(db, user.id, data.alert_type, data.condition, data.portfolio_id, data.message)
    return _to_response(alert)


@router.get("", summary="List Alerts")
def list_alerts(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    alerts = get_user_alerts(db, user.id)
    return {"alerts": [_to_response(a) for a in alerts], "count": len(alerts)}


@router.get("/{alert_id}", summary="Get Alert")
def get_one(alert_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    alert = get_alert(db, alert_id, user.id)
    if not alert:
        raise HTTPException(404, "Alert not found")
    return _to_response(alert)


@router.delete("/{alert_id}", status_code=204, summary="Delete Alert")
def delete(alert_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    alert = get_alert(db, alert_id, user.id)
    if not alert:
        raise HTTPException(404, "Alert not found")
    delete_alert(db, alert)


@router.post("/evaluate", summary="Evaluate Alerts")
def evaluate(data: EvaluateRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Evaluate all active alerts against current market data."""
    triggered = check_alerts(
        db, user.id,
        current_var=data.current_var,
        current_prices=data.current_prices,
        days_to_expiry=data.days_to_expiry,
    )
    return {"triggered": triggered, "count": len(triggered)}


def _to_response(a) -> dict:
    return {
        "id": a.id,
        "alert_type": a.alert_type.value,
        "condition": a.condition,
        "message": a.message,
        "is_active": a.is_active,
        "triggered": a.triggered,
        "triggered_at": a.triggered_at.isoformat() if a.triggered_at else None,
        "portfolio_id": a.portfolio_id,
        "created_at": a.created_at.isoformat(),
    }

"""
Portfolio API Routes
=====================
CRUD endpoints for saved portfolios + risk calculation on saved portfolios.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from api.middleware.auth import get_current_user
from schemas.portfolio import PortfolioCreateRequest, PortfolioUpdateRequest, PortfolioResponse
from services.portfolio_db_service import (
    create_portfolio, get_user_portfolios, get_portfolio,
    update_portfolio, delete_portfolio,
)
from services.risk_service import compute_portfolio_risk

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/portfolios", tags=["Portfolios"])


@router.get("", summary="List Portfolios")
def list_portfolios(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    portfolios = get_user_portfolios(db, user.id)
    return {"portfolios": [_to_response(p) for p in portfolios], "count": len(portfolios)}


@router.post("", status_code=201, summary="Create Portfolio")
def create(data: PortfolioCreateRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    positions = [pos.model_dump() for pos in data.positions]
    p = create_portfolio(db, user.id, data.name, data.description, positions)
    return _to_response(p)


@router.get("/{portfolio_id}", summary="Get Portfolio")
def get_one(portfolio_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    p = get_portfolio(db, portfolio_id, user.id)
    if not p:
        raise HTTPException(404, "Portfolio not found")
    return _to_response(p)


@router.put("/{portfolio_id}", summary="Update Portfolio")
def update(portfolio_id: str, data: PortfolioUpdateRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    p = get_portfolio(db, portfolio_id, user.id)
    if not p:
        raise HTTPException(404, "Portfolio not found")
    positions = [pos.model_dump() for pos in data.positions] if data.positions is not None else None
    p = update_portfolio(db, p, data.name, data.description, positions)
    return _to_response(p)


@router.delete("/{portfolio_id}", status_code=204, summary="Delete Portfolio")
def delete(portfolio_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    p = get_portfolio(db, portfolio_id, user.id)
    if not p:
        raise HTTPException(404, "Portfolio not found")
    delete_portfolio(db, p)


@router.post("/{portfolio_id}/calculate-risk", summary="Calculate Risk on Saved Portfolio")
def calculate_risk(portfolio_id: str, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    p = get_portfolio(db, portfolio_id, user.id)
    if not p:
        raise HTTPException(404, "Portfolio not found")
    if not p.positions:
        raise HTTPException(400, "Portfolio has no positions")
    return compute_portfolio_risk(p.positions)


def _to_response(p) -> dict:
    return {
        "id": p.id, "name": p.name, "description": p.description,
        "positions": p.positions, "is_active": p.is_active,
        "created_at": p.created_at.isoformat(), "updated_at": p.updated_at.isoformat(),
    }

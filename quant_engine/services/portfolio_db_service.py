"""
Portfolio DB Service
=====================
CRUD operations for saved portfolios.
"""

import logging
from typing import List, Optional
from sqlalchemy.orm import Session

from db.models import Portfolio

logger = logging.getLogger(__name__)


def create_portfolio(db: Session, user_id: str, name: str, description: str = None, positions: list = None) -> Portfolio:
    portfolio = Portfolio(user_id=user_id, name=name, description=description, positions=positions or [])
    db.add(portfolio)
    db.commit()
    db.refresh(portfolio)
    logger.info(f"Portfolio created: '{name}' for user={user_id}")
    return portfolio


def get_user_portfolios(db: Session, user_id: str, active_only: bool = True) -> List[Portfolio]:
    q = db.query(Portfolio).filter(Portfolio.user_id == user_id)
    if active_only:
        q = q.filter(Portfolio.is_active == True)
    return q.order_by(Portfolio.updated_at.desc()).all()


def get_portfolio(db: Session, portfolio_id: str, user_id: str) -> Optional[Portfolio]:
    return db.query(Portfolio).filter(
        Portfolio.id == portfolio_id, Portfolio.user_id == user_id
    ).first()


def update_portfolio(db: Session, portfolio: Portfolio, name: str = None, description: str = None, positions: list = None) -> Portfolio:
    if name is not None:
        portfolio.name = name
    if description is not None:
        portfolio.description = description
    if positions is not None:
        portfolio.positions = positions
    db.commit()
    db.refresh(portfolio)
    return portfolio


def delete_portfolio(db: Session, portfolio: Portfolio):
    portfolio.is_active = False  # Soft delete
    db.commit()

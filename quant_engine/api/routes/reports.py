"""
Reports API Routes
====================
GET  /reports/trades        — Download CSV trade report
GET  /reports/portfolios    — Download CSV portfolio report
POST /reports/risk          — Generate risk report for open positions
"""

import logging
from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import io

from db.database import get_db
from db.models import User
from api.middleware.auth import get_current_user
from services.report_service import (
    generate_trade_report_csv,
    generate_portfolio_report_csv,
    generate_risk_report,
)
from services.risk_service import compute_portfolio_risk
from services.trade_service import get_user_trades

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/reports", tags=["Reports"])


@router.get("/trades", summary="Download Trade Report (CSV)")
def trade_report(
    status: str = Query(None, description="Filter: open, closed"),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    csv_data = generate_trade_report_csv(db, user.id, status)
    return StreamingResponse(
        io.BytesIO(csv_data.encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=trade_report.csv"},
    )


@router.get("/portfolios", summary="Download Portfolio Report (CSV)")
def portfolio_report(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    csv_data = generate_portfolio_report_csv(db, user.id)
    return StreamingResponse(
        io.BytesIO(csv_data.encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=portfolio_report.csv"},
    )


@router.post("/risk", summary="Generate Risk Report")
def risk_report(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate a comprehensive risk report for all open positions."""
    trades = get_user_trades(db, user.id, status="open")
    if not trades:
        return {"error": "No open trades to analyze"}

    # Build portfolio from open trades
    portfolio = []
    for t in trades:
        direction = 1 if t.side.value == "buy" else -1
        portfolio.append({
            "type": t.option_type.value,
            "S": t.spot_at_entry,
            "K": t.strike,
            "T": t.T_at_entry,
            "r": t.r_at_entry,
            "sigma": t.sigma_at_entry,
            "qty": t.quantity * direction,
        })

    risk_results = compute_portfolio_risk(portfolio)
    report = generate_risk_report(db, user.id, risk_results)
    return report

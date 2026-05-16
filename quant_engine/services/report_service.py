"""
Report Service
================
Generate PDF and CSV reports for portfolios and risk analytics.
Uses built-in libraries (no external PDF dependency required).
"""

import csv
import io
import json
import logging
from datetime import datetime, timezone
from typing import List

from sqlalchemy.orm import Session

from db.models import Trade, Portfolio, TradeStatus

logger = logging.getLogger(__name__)


# =============================================================================
# CSV Reports
# =============================================================================

def generate_trade_report_csv(db: Session, user_id: str, status: str = None) -> str:
    """Generate a CSV report of user's trades."""
    q = db.query(Trade).filter(Trade.user_id == user_id)
    if status:
        q = q.filter(Trade.status == TradeStatus(status))
    trades = q.order_by(Trade.traded_at.desc()).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Trade ID", "Date", "Side", "Type", "Strike", "Spot at Entry",
        "Premium", "Quantity", "Notional", "Sigma", "T", "Status",
        "Closed At", "Close Premium", "P&L", "Notes",
    ])

    for t in trades:
        pnl = ""
        if t.status == TradeStatus.closed and t.close_premium is not None:
            direction = 1 if t.side.value == "buy" else -1
            pnl = round((t.close_premium - t.premium) * t.quantity * direction, 4)

        writer.writerow([
            t.id,
            t.traded_at.strftime("%Y-%m-%d %H:%M"),
            t.side.value,
            t.option_type.value,
            t.strike,
            t.spot_at_entry,
            round(t.premium, 4),
            t.quantity,
            round(t.premium * t.quantity, 4),
            t.sigma_at_entry,
            t.T_at_entry,
            t.status.value,
            t.closed_at.strftime("%Y-%m-%d %H:%M") if t.closed_at else "",
            round(t.close_premium, 4) if t.close_premium else "",
            pnl,
            t.notes or "",
        ])

    logger.info(f"CSV trade report generated: {len(trades)} trades")
    return output.getvalue()


def generate_portfolio_report_csv(db: Session, user_id: str) -> str:
    """Generate a CSV report of user's portfolios."""
    portfolios = db.query(Portfolio).filter(
        Portfolio.user_id == user_id, Portfolio.is_active == True
    ).all()

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Portfolio ID", "Name", "Description", "# Positions",
        "Created", "Updated", "Positions (JSON)",
    ])

    for p in portfolios:
        writer.writerow([
            p.id, p.name, p.description or "",
            len(p.positions or []),
            p.created_at.strftime("%Y-%m-%d"),
            p.updated_at.strftime("%Y-%m-%d"),
            json.dumps(p.positions),
        ])

    logger.info(f"CSV portfolio report generated: {len(portfolios)} portfolios")
    return output.getvalue()


# =============================================================================
# Risk Report (JSON summary)
# =============================================================================

def generate_risk_report(db: Session, user_id: str, risk_results: dict) -> dict:
    """Generate a structured risk report combining positions and analytics."""
    trades = db.query(Trade).filter(
        Trade.user_id == user_id, Trade.status == TradeStatus.open
    ).all()

    report = {
        "report_type": "portfolio_risk",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "user_id": user_id,
        "summary": {
            "n_open_trades": len(trades),
            "total_notional": round(sum(t.premium * t.quantity for t in trades), 2),
        },
        "risk_metrics": {
            "VaR": risk_results.get("VaR"),
            "CVaR": risk_results.get("CVaR"),
            "method": risk_results.get("method"),
            "confidence": risk_results.get("confidence"),
            "portfolio_value": risk_results.get("portfolio_value"),
        },
        "positions": [
            {
                "id": t.id,
                "side": t.side.value,
                "type": t.option_type.value,
                "strike": t.strike,
                "premium": round(t.premium, 4),
                "quantity": t.quantity,
                "sigma": t.sigma_at_entry,
            }
            for t in trades
        ],
        "pnl_statistics": risk_results.get("pnl_statistics", {}),
    }

    logger.info(f"Risk report generated for user={user_id}")
    return report

"""
Regulatory Reporting API Routes
==================================
POST /regulatory/var               — 10-day 99% Regulatory VaR
POST /regulatory/stressed-var      — Stressed VaR (crisis scenario)
POST /regulatory/capital-charge    — Basel capital charge
POST /regulatory/leverage          — Leverage ratio
POST /regulatory/concentration     — Concentration risk analysis
POST /regulatory/full-report       — Complete Basel III/IV report
"""

import logging
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from typing import List, Optional
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import User
from api.middleware.auth import get_current_user, require_role
from services.regulatory_service import (
    regulatory_var, stressed_var, capital_charge,
    leverage_ratio, concentration_risk, full_regulatory_report,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/regulatory", tags=["Regulatory"])


class PositionItem(BaseModel):
    type: str = "call"
    S: float = Field(gt=0)
    K: float = Field(gt=0)
    T: float = Field(gt=0)
    r: float = 0.05
    sigma: float = Field(gt=0)
    qty: int = 1


class RegulatoryRequest(BaseModel):
    portfolio: List[PositionItem]
    confidence: float = 0.99
    stress_factor: float = 2.0
    avg_var_60d: Optional[float] = None
    tier1_capital: Optional[float] = None


@router.post("/var", summary="Regulatory VaR (10-day 99%)")
def reg_var(data: RegulatoryRequest, user: User = Depends(get_current_user)):
    portfolio = [p.model_dump() for p in data.portfolio]
    return regulatory_var(portfolio, data.confidence)


@router.post("/stressed-var", summary="Stressed VaR")
def stress_var(data: RegulatoryRequest, user: User = Depends(get_current_user)):
    portfolio = [p.model_dump() for p in data.portfolio]
    return stressed_var(portfolio, data.stress_factor)


@router.post("/capital-charge", summary="Basel Capital Charge")
def cap_charge(data: RegulatoryRequest, user: User = Depends(get_current_user)):
    portfolio = [p.model_dump() for p in data.portfolio]
    return capital_charge(portfolio, data.avg_var_60d)


@router.post("/leverage", summary="Leverage Ratio")
def lev_ratio(data: RegulatoryRequest, user: User = Depends(get_current_user)):
    portfolio = [p.model_dump() for p in data.portfolio]
    return leverage_ratio(portfolio, data.tier1_capital)


@router.post("/concentration", summary="Concentration Risk")
def conc_risk(data: RegulatoryRequest, user: User = Depends(get_current_user)):
    portfolio = [p.model_dump() for p in data.portfolio]
    return concentration_risk(portfolio)


@router.post("/full-report", summary="Full Basel III/IV Report")
def full_report(data: RegulatoryRequest, user: User = Depends(require_role("admin", "risk_manager"))):
    portfolio = [p.model_dump() for p in data.portfolio]
    return full_regulatory_report(portfolio)

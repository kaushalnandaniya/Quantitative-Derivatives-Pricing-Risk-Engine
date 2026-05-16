"""
Portfolio & Trade Schemas
===========================
Pydantic models for portfolio CRUD and trade booking.
"""

from typing import Optional, List, Literal
from pydantic import BaseModel, Field


# =============================================================================
# Portfolio Schemas
# =============================================================================

class PositionItem(BaseModel):
    type: Literal["call", "put"]
    S: float = Field(..., gt=0)
    K: float = Field(..., gt=0)
    T: float = Field(..., gt=0, le=30)
    r: float = Field(0.05, ge=-0.1, le=1.0)
    sigma: float = Field(0.2, gt=0, le=5.0)
    qty: int = Field(1)


class PortfolioCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: Optional[str] = None
    positions: List[PositionItem] = Field(default_factory=list)


class PortfolioUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    description: Optional[str] = None
    positions: Optional[List[PositionItem]] = None


class PortfolioResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    positions: list
    is_active: bool
    created_at: str
    updated_at: str
    model_config = {"from_attributes": True}


# =============================================================================
# Trade Schemas
# =============================================================================

class TradeBookRequest(BaseModel):
    portfolio_id: Optional[str] = None
    side: Literal["buy", "sell"]
    option_type: Literal["call", "put"]
    spot: float = Field(..., gt=0, description="Spot price at entry")
    strike: float = Field(..., gt=0, description="Strike price")
    T: float = Field(..., gt=0, le=30, description="Time to maturity (years)")
    r: float = Field(0.05, description="Risk-free rate")
    sigma: float = Field(0.2, gt=0, le=5.0, description="Volatility")
    quantity: int = Field(..., ge=1, description="Number of contracts")
    notes: Optional[str] = None


class TradeCloseRequest(BaseModel):
    close_premium: Optional[float] = Field(None, description="Premium at close (auto-calculated if omitted)")


class TradeResponse(BaseModel):
    id: str
    side: str
    option_type: str
    spot_at_entry: float
    strike: float
    premium: float
    quantity: int
    sigma_at_entry: float
    T_at_entry: float
    status: str
    traded_at: str
    closed_at: Optional[str]
    close_premium: Optional[float]
    notes: Optional[str]
    portfolio_id: Optional[str]
    model_config = {"from_attributes": True}

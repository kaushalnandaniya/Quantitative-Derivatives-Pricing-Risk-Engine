"""
Market Data Schemas
====================
Pydantic models for market data API endpoints.
"""

from typing import Optional
from pydantic import BaseModel, Field


class QuoteRequest(BaseModel):
    symbol: str = Field(..., description="Instrument symbol (e.g., NIFTY, BANKNIFTY, RELIANCE)")


class OptionChainRequest(BaseModel):
    symbol: str = Field(..., description="Instrument symbol")
    expiry: Optional[str] = Field(None, description="Expiry date (YYYY-MM-DD). Defaults to nearest.")

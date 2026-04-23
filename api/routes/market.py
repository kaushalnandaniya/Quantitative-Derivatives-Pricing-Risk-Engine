"""
Market Data Routes
===================
API endpoints for market data:
    GET /market/status
    GET /market/quote/{symbol}
    GET /market/option-chain/{symbol}
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query

from services.market_data import get_provider

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market", tags=["Market Data"])


@router.get("/status", summary="Market Data Provider Status")
def market_status():
    """Check the market data provider status and available symbols."""
    return get_provider().get_status()


@router.get("/quote/{symbol}", summary="Get Quote")
def get_quote(symbol: str):
    """Get current quote for a symbol (NIFTY, BANKNIFTY, RELIANCE)."""
    return get_provider().get_quote(symbol)


@router.get("/option-chain/{symbol}", summary="Get Option Chain")
def get_option_chain(symbol: str, expiry: Optional[str] = Query(None)):
    """Get full option chain with strikes, prices, IV, and Greeks."""
    return get_provider().get_option_chain(symbol, expiry)

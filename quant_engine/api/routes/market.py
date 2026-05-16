"""
Market Data Routes
===================
API endpoints for market data:
    GET  /market/status
    GET  /market/quote/{symbol}
    GET  /market/option-chain/{symbol}
    GET  /market/history/{symbol}
    GET  /market/search
    POST /market/kite/connect
    POST /market/kite/disconnect
    GET  /market/kite/status
    POST /market/kite/order
    GET  /market/kite/orders
    GET  /market/kite/positions
    GET  /market/kite/holdings
"""

import logging
from typing import Optional

from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel

from services.market_data import get_provider
from services import kite_broker
from db.models import User
from api.middleware.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/market", tags=["Market Data"])


# ==========================================================================
# Core Market Data
# ==========================================================================

@router.get("/status", summary="Market Data Provider Status")
def market_status():
    """Check the market data provider status and available symbols."""
    return get_provider().get_status()


@router.get("/quote/{symbol}", summary="Get Quote")
def get_quote(symbol: str):
    """Get current quote for a symbol (NIFTY, BANKNIFTY, RELIANCE, etc.)."""
    try:
        return get_provider().get_quote(symbol)
    except ValueError as e:
        # If not in mock list, try yfinance live quote
        try:
            import yfinance as yf
            yf_sym = symbol.upper()
            if not yf_sym.endswith(".NS") and not yf_sym.endswith(".BO"):
                yf_sym += ".NS"
            ticker = yf.Ticker(yf_sym)
            info = ticker.fast_info
            hist = ticker.history(period="2d")
            if hist.empty:
                raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")
            last = hist.iloc[-1]
            prev_close = hist.iloc[-2]["Close"] if len(hist) > 1 else last["Close"]
            change = last["Close"] - prev_close
            return {
                "symbol": symbol.upper(),
                "name": symbol.upper(),
                "last_price": round(last["Close"], 2),
                "change": round(change, 2),
                "change_pct": round((change / prev_close) * 100, 2) if prev_close else 0,
                "open": round(last["Open"], 2),
                "high": round(last["High"], 2),
                "low": round(last["Low"], 2),
                "volume": int(last["Volume"]),
                "timestamp": last.name.strftime("%Y-%m-%dT%H:%M:%S"),
                "provider": "yfinance",
            }
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=404, detail=f"Symbol '{symbol}' not found")


@router.get("/option-chain/{symbol}", summary="Get Option Chain")
def get_option_chain(symbol: str, expiry: Optional[str] = Query(None)):
    """Get full option chain with strikes, prices, IV, and Greeks."""
    return get_provider().get_option_chain(symbol, expiry)


# ==========================================================================
# Historical Data (free via yfinance)
# ==========================================================================

@router.get("/history/{symbol}", summary="Historical Price Data")
def get_history(
    symbol: str,
    period: str = Query("1mo", description="1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max"),
):
    """Get historical OHLCV data for charting. Uses Yahoo Finance (free)."""
    try:
        return get_provider().get_historical_data(symbol, period)
    except Exception as e:
        logger.error(f"History fetch failed for {symbol}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==========================================================================
# Symbol Search
# ==========================================================================

@router.get("/search", summary="Search Symbols")
def search_symbols(q: str = Query(..., min_length=1, description="Search query")):
    """Search for NSE/BSE stock symbols."""
    return {"results": get_provider().search_symbols(q)}


# ==========================================================================
# Kite Connect Integration (optional — requires user's API key)
# ==========================================================================

class KiteConnectRequest(BaseModel):
    api_key: str
    api_secret: str
    request_token: str

class KiteOrderRequest(BaseModel):
    tradingsymbol: str
    exchange: str = "NSE"
    transaction_type: str  # BUY or SELL
    order_type: str = "MARKET"  # MARKET, LIMIT, SL, SL-M
    quantity: int
    product: str = "CNC"  # CNC (delivery), MIS (intraday), NRML
    price: Optional[float] = None
    trigger_price: Optional[float] = None


@router.post("/kite/connect", summary="Connect to Kite")
def kite_connect(data: KiteConnectRequest, user: User = Depends(get_current_user)):
    """Authenticate with Zerodha Kite Connect using your API credentials."""
    result = kite_broker.connect_kite(
        user_id=str(user.id),
        api_key=data.api_key,
        api_secret=data.api_secret,
        request_token=data.request_token,
    )
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/kite/disconnect", summary="Disconnect from Kite")
def kite_disconnect(user: User = Depends(get_current_user)):
    """Disconnect your Kite session."""
    return kite_broker.disconnect_kite(str(user.id))


@router.get("/kite/status", summary="Kite Connection Status")
def kite_status(user: User = Depends(get_current_user)):
    """Check if you have an active Kite connection."""
    connected = kite_broker.is_connected(str(user.id))
    return {"connected": connected}


@router.post("/kite/order", summary="Place Order via Kite")
def kite_place_order(data: KiteOrderRequest, user: User = Depends(get_current_user)):
    """Place a buy/sell order through your connected Kite account."""
    result = kite_broker.place_order(str(user.id), data.model_dump())
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/kite/orders", summary="Kite Order Book")
def kite_orders(user: User = Depends(get_current_user)):
    """Get today's orders from your Kite account."""
    result = kite_broker.get_orders(str(user.id))
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/kite/positions", summary="Kite Positions")
def kite_positions(user: User = Depends(get_current_user)):
    """Get your current positions from Kite."""
    result = kite_broker.get_positions(str(user.id))
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.get("/kite/holdings", summary="Kite Holdings")
def kite_holdings(user: User = Depends(get_current_user)):
    """Get your portfolio holdings from Kite."""
    result = kite_broker.get_holdings(str(user.id))
    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])
    return result

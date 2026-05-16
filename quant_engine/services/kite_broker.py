"""
Kite Broker Service
=====================
Handles Zerodha Kite Connect authentication and order placement.
Users who have a Kite Connect API subscription can:
    1. Authenticate via API Key + Request Token
    2. Place buy/sell orders
    3. View order book and positions

This module is entirely optional — the platform works without it.
"""

import os
import logging
from typing import Dict, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# In-memory session store (per-user Kite sessions)
# In production, store encrypted tokens in the database.
_kite_sessions: Dict[str, object] = {}


def connect_kite(user_id: str, api_key: str, api_secret: str, request_token: str) -> Dict:
    """
    Complete the Kite Connect login flow:
      1. Exchange the request_token for an access_token
      2. Store the session for subsequent API calls
    """
    try:
        from kiteconnect import KiteConnect
    except ImportError:
        return {"success": False, "error": "kiteconnect package is not installed on the server."}

    try:
        kite = KiteConnect(api_key=api_key)
        session = kite.generate_session(request_token, api_secret=api_secret)
        kite.set_access_token(session["access_token"])

        _kite_sessions[user_id] = kite

        profile = kite.profile()
        logger.info(f"Kite connected for user {user_id}: {profile.get('user_name', 'N/A')}")

        return {
            "success": True,
            "user_name": profile.get("user_name", ""),
            "email": profile.get("email", ""),
            "broker": profile.get("broker", "ZERODHA"),
            "exchanges": profile.get("exchanges", []),
            "connected_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        logger.error(f"Kite login failed for user {user_id}: {e}")
        return {"success": False, "error": str(e)}


def disconnect_kite(user_id: str) -> Dict:
    """Remove a user's Kite session."""
    if user_id in _kite_sessions:
        del _kite_sessions[user_id]
        return {"success": True, "message": "Disconnected from Kite."}
    return {"success": False, "error": "No active Kite session."}


def get_kite(user_id: str):
    """Get the KiteConnect instance for a user, or None."""
    return _kite_sessions.get(user_id)


def is_connected(user_id: str) -> bool:
    return user_id in _kite_sessions


def place_order(user_id: str, params: Dict) -> Dict:
    """
    Place an order via Kite Connect.

    Expected params:
        tradingsymbol: str   — e.g. "INFY", "RELIANCE"
        exchange: str        — "NSE" or "BSE"
        transaction_type: str — "BUY" or "SELL"
        order_type: str      — "MARKET", "LIMIT", "SL", "SL-M"
        quantity: int
        product: str         — "CNC" (delivery), "MIS" (intraday), "NRML"
        price: float         — required for LIMIT / SL orders
        trigger_price: float — required for SL / SL-M orders
    """
    kite = get_kite(user_id)
    if not kite:
        return {"success": False, "error": "Kite not connected. Please authenticate first."}

    try:
        order_id = kite.place_order(
            variety=kite.VARIETY_REGULAR,
            tradingsymbol=params["tradingsymbol"],
            exchange=params.get("exchange", "NSE"),
            transaction_type=params["transaction_type"],
            order_type=params["order_type"],
            quantity=int(params["quantity"]),
            product=params.get("product", "CNC"),
            price=params.get("price"),
            trigger_price=params.get("trigger_price"),
        )
        logger.info(f"Order placed for user {user_id}: {order_id}")
        return {"success": True, "order_id": order_id}
    except Exception as e:
        logger.error(f"Order failed for user {user_id}: {e}")
        return {"success": False, "error": str(e)}


def get_orders(user_id: str) -> Dict:
    """Get today's order book from Kite."""
    kite = get_kite(user_id)
    if not kite:
        return {"success": False, "error": "Kite not connected."}
    try:
        orders = kite.orders()
        return {"success": True, "orders": orders}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_positions(user_id: str) -> Dict:
    """Get current positions from Kite."""
    kite = get_kite(user_id)
    if not kite:
        return {"success": False, "error": "Kite not connected."}
    try:
        positions = kite.positions()
        return {"success": True, "positions": positions}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_holdings(user_id: str) -> Dict:
    """Get portfolio holdings from Kite."""
    kite = get_kite(user_id)
    if not kite:
        return {"success": False, "error": "Kite not connected."}
    try:
        holdings = kite.holdings()
        return {"success": True, "holdings": holdings}
    except Exception as e:
        return {"success": False, "error": str(e)}

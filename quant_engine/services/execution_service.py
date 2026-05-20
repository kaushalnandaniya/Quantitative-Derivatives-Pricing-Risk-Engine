"""
Execution Service — Broker Integration
========================================
Handles order routing to the primary reference broker (Zerodha Kite).
Falls back to simulated mock execution if API keys are missing.
"""

import os
import logging
import uuid
from typing import Dict, Any

logger = logging.getLogger(__name__)

# Try to initialize KiteConnect
KITE_API_KEY = os.getenv("KITE_API_KEY")
KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")

kite = None
try:
    if KITE_API_KEY:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=KITE_API_KEY)
        if KITE_ACCESS_TOKEN:
            kite.set_access_token(KITE_ACCESS_TOKEN)
        logger.info("KiteConnect initialized successfully.")
    else:
        logger.warning("KITE_API_KEY not found. Using mock execution service.")
except ImportError:
    logger.warning("kiteconnect package not installed. Using mock execution service.")


def route_order(
    side: str, 
    quantity: int, 
    symbol: str, 
    order_type: str = "MARKET", 
    price: float = None
) -> Dict[str, Any]:
    """
    Route an order to the broker.
    Returns a dict with 'status' and 'exchange_order_id'.
    """
    if kite and KITE_ACCESS_TOKEN:
        try:
            # Map our terminology to Kite terminology
            transaction_type = kite.TRANSACTION_TYPE_BUY if side.lower() == "buy" else kite.TRANSACTION_TYPE_SELL
            order_type_kite = kite.ORDER_TYPE_MARKET if order_type.upper() == "MARKET" else kite.ORDER_TYPE_LIMIT
            
            order_id = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=kite.EXCHANGE_NFO,
                tradingsymbol=symbol,
                transaction_type=transaction_type,
                quantity=quantity,
                product=kite.PRODUCT_NRML,
                order_type=order_type_kite,
                price=price
            )
            return {"status": "SUBMITTED", "exchange_order_id": order_id}
        except Exception as e:
            logger.error(f"Kite order placement failed: {e}")
            return {"status": "FAILED", "error": str(e)}
    else:
        # Mock execution
        logger.info(f"[MOCK] Routing {side} {quantity} of {symbol} at {order_type}")
        return {"status": "SUBMITTED", "exchange_order_id": f"mock_{uuid.uuid4().hex[:8]}"}

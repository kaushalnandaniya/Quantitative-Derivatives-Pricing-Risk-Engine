"""
WebSocket Market Feed
======================
Real-time market data and portfolio updates via WebSocket.

Channels:
    - market:{symbol}    → price ticks every 1s
    - portfolio:{id}     → P&L updates every 5s
    - alerts:{user_id}   → triggered alerts
"""

import asyncio
import json
import logging
import time
from typing import Dict, Set

from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages active WebSocket connections and channel subscriptions."""

    def __init__(self):
        self.connections: Dict[str, Set[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, channel: str):
        await websocket.accept()
        if channel not in self.connections:
            self.connections[channel] = set()
        self.connections[channel].add(websocket)
        logger.info(f"WS connected: {channel} (total={len(self.connections[channel])})")

    def disconnect(self, websocket: WebSocket, channel: str):
        if channel in self.connections:
            self.connections[channel].discard(websocket)
            if not self.connections[channel]:
                del self.connections[channel]
        logger.info(f"WS disconnected: {channel}")

    async def broadcast(self, channel: str, data: dict):
        if channel not in self.connections:
            return
        dead = set()
        for ws in self.connections[channel]:
            try:
                await ws.send_json(data)
            except Exception:
                dead.add(ws)
        for ws in dead:
            self.connections[channel].discard(ws)

    @property
    def active_connections(self) -> int:
        return sum(len(conns) for conns in self.connections.values())


manager = ConnectionManager()


async def market_feed_handler(websocket: WebSocket, symbol: str):
    """
    WebSocket handler for market price ticks.
    Pushes simulated price updates every second.
    """
    channel = f"market:{symbol.upper()}"
    await manager.connect(websocket, channel)

    try:
        # Import here to avoid circular imports
        from services.market_data import get_provider
        provider = get_provider()

        while True:
            try:
                quote = provider.get_quote(symbol)
                await websocket.send_json({
                    "channel": channel,
                    "type": "tick",
                    "data": {
                        "symbol": quote["symbol"],
                        "last_price": quote["last_price"],
                        "change": quote["change"],
                        "change_pct": quote["change_pct"],
                        "timestamp": quote["timestamp"],
                    },
                })
            except Exception as e:
                logger.error(f"WS tick error: {e}")

            await asyncio.sleep(1.0)

    except WebSocketDisconnect:
        manager.disconnect(websocket, channel)
    except Exception as e:
        logger.error(f"WS error: {e}")
        manager.disconnect(websocket, channel)

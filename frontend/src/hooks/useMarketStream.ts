"use client";

import { useState, useEffect, useRef } from "react";

interface MarketTick {
  symbol: string;
  last_price: number;
  change: number;
  change_pct: number;
  timestamp: string;
}

/**
 * React hook for real-time market data via WebSocket.
 * Auto-reconnects on disconnect with exponential backoff.
 */
export function useMarketStream(symbol: string, enabled = true) {
  const [tick, setTick] = useState<MarketTick | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef(0);

  useEffect(() => {
    if (!enabled || !symbol) return;

    let unmounted = false;
    let retryTimer: ReturnType<typeof setTimeout>;

    function connect() {
      if (unmounted) return;

      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
      const wsUrl = baseUrl.replace(/^http/, "ws") + `/ws/market/${symbol}`;

      try {
        const ws = new WebSocket(wsUrl);
        wsRef.current = ws;

        ws.onopen = () => {
          if (!unmounted) {
            setConnected(true);
            retryRef.current = 0;
          }
        };

        ws.onmessage = (event) => {
          if (unmounted) return;
          try {
            const msg = JSON.parse(event.data);
            if (msg.type === "tick" && msg.data) {
              setTick(msg.data);
            }
          } catch { /* ignore */ }
        };

        ws.onclose = () => {
          if (unmounted) return;
          setConnected(false);
          wsRef.current = null;
          const delay = Math.min(1000 * Math.pow(2, retryRef.current), 30000);
          retryRef.current++;
          retryTimer = setTimeout(connect, delay);
        };

        ws.onerror = () => { ws.close(); };
      } catch { /* retry via onclose */ }
    }

    connect();

    return () => {
      unmounted = true;
      clearTimeout(retryTimer);
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [symbol, enabled]);

  return { tick, connected };
}

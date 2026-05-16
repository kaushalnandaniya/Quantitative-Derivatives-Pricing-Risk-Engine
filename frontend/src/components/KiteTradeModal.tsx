"use client";

import { useState } from "react";
import { marketApi } from "@/lib/api";

interface Props {
  symbol: string;
  lastPrice: number;
  token: string;
  onClose: () => void;
}

export default function KiteTradeModal({ symbol, lastPrice, token, onClose }: Props) {
  const [side, setSide] = useState<"BUY" | "SELL">("BUY");
  const [orderType, setOrderType] = useState("MARKET");
  const [qty, setQty] = useState(1);
  const [price, setPrice] = useState(lastPrice);
  const [product, setProduct] = useState("CNC");
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [error, setError] = useState("");

  const submit = async () => {
    setSubmitting(true);
    setError("");
    try {
      const res = await marketApi.kiteOrder(
        { tradingsymbol: symbol, exchange: "NSE", transaction_type: side, order_type: orderType, quantity: qty, product, price: orderType === "LIMIT" ? price : undefined },
        token
      );
      setResult(res);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Order failed");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div style={{ position: "fixed", inset: 0, zIndex: 100, display: "flex", alignItems: "center", justifyContent: "center", background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)" }} onClick={onClose}>
      <div onClick={e => e.stopPropagation()} style={{ background: "var(--color-bg-card)", border: "1px solid var(--color-border-subtle)", borderRadius: 16, padding: 28, width: 420, maxWidth: "90vw" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 20 }}>
          <h3 style={{ color: "var(--color-text-primary)", fontSize: 18, fontWeight: 700, margin: 0 }}>Trade {symbol}</h3>
          <button onClick={onClose} style={{ background: "none", border: "none", color: "var(--color-text-muted)", fontSize: 20, cursor: "pointer" }}>✕</button>
        </div>

        <div style={{ color: "var(--color-text-secondary)", fontSize: 13, marginBottom: 16 }}>
          LTP: <span style={{ color: "var(--color-accent-blue)", fontFamily: "var(--font-mono)", fontWeight: 600 }}>₹{lastPrice.toLocaleString()}</span>
        </div>

        {/* Side */}
        <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
          {(["BUY", "SELL"] as const).map(s => (
            <button key={s} onClick={() => setSide(s)} style={{ flex: 1, padding: "10px 0", borderRadius: 8, border: "none", fontWeight: 700, fontSize: 14, cursor: "pointer", background: side === s ? (s === "BUY" ? "var(--color-accent-green)" : "var(--color-accent-red)") : "var(--color-bg-secondary)", color: side === s ? "#fff" : "var(--color-text-secondary)", transition: "all 0.15s" }}>
              {s}
            </button>
          ))}
        </div>

        {/* Order Type + Product */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 14 }}>
          <div>
            <label style={{ fontSize: 11, color: "var(--color-text-muted)", fontWeight: 600, textTransform: "uppercase", display: "block", marginBottom: 4 }}>Order Type</label>
            <select value={orderType} onChange={e => setOrderType(e.target.value)} style={{ width: "100%", padding: 8, borderRadius: 8, border: "1px solid var(--color-border-subtle)", background: "var(--color-bg-secondary)", color: "var(--color-text-primary)", fontSize: 13 }}>
              <option value="MARKET">Market</option>
              <option value="LIMIT">Limit</option>
              <option value="SL">Stop Loss</option>
              <option value="SL-M">SL-Market</option>
            </select>
          </div>
          <div>
            <label style={{ fontSize: 11, color: "var(--color-text-muted)", fontWeight: 600, textTransform: "uppercase", display: "block", marginBottom: 4 }}>Product</label>
            <select value={product} onChange={e => setProduct(e.target.value)} style={{ width: "100%", padding: 8, borderRadius: 8, border: "1px solid var(--color-border-subtle)", background: "var(--color-bg-secondary)", color: "var(--color-text-primary)", fontSize: 13 }}>
              <option value="CNC">CNC (Delivery)</option>
              <option value="MIS">MIS (Intraday)</option>
              <option value="NRML">NRML</option>
            </select>
          </div>
        </div>

        {/* Qty + Price */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 10, marginBottom: 20 }}>
          <div>
            <label style={{ fontSize: 11, color: "var(--color-text-muted)", fontWeight: 600, textTransform: "uppercase", display: "block", marginBottom: 4 }}>Quantity</label>
            <input type="number" value={qty} min={1} onChange={e => setQty(Number(e.target.value))} style={{ width: "100%", padding: 8, borderRadius: 8, border: "1px solid var(--color-border-subtle)", background: "var(--color-bg-secondary)", color: "var(--color-text-primary)", fontSize: 13 }} />
          </div>
          {orderType !== "MARKET" && (
            <div>
              <label style={{ fontSize: 11, color: "var(--color-text-muted)", fontWeight: 600, textTransform: "uppercase", display: "block", marginBottom: 4 }}>Price</label>
              <input type="number" value={price} step={0.05} onChange={e => setPrice(Number(e.target.value))} style={{ width: "100%", padding: 8, borderRadius: 8, border: "1px solid var(--color-border-subtle)", background: "var(--color-bg-secondary)", color: "var(--color-text-primary)", fontSize: 13 }} />
            </div>
          )}
        </div>

        {error && <div style={{ color: "var(--color-accent-red)", fontSize: 12, marginBottom: 12, padding: "8px 12px", borderRadius: 8, background: "rgba(255,69,58,0.1)" }}>{error}</div>}
        {result && <div style={{ color: "var(--color-accent-green)", fontSize: 12, marginBottom: 12, padding: "8px 12px", borderRadius: 8, background: "rgba(48,209,88,0.1)" }}>✓ Order placed! ID: {String(result.order_id)}</div>}

        <button onClick={submit} disabled={submitting} style={{ width: "100%", padding: 12, borderRadius: 10, border: "none", fontWeight: 700, fontSize: 14, cursor: submitting ? "wait" : "pointer", background: side === "BUY" ? "var(--color-accent-green)" : "var(--color-accent-red)", color: "#fff", opacity: submitting ? 0.6 : 1, transition: "all 0.15s" }}>
          {submitting ? "Placing..." : `${side} ${qty} × ${symbol}`}
        </button>
      </div>
    </div>
  );
}

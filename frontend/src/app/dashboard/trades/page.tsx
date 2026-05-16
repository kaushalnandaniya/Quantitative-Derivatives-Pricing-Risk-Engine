"use client";

import { useState, useEffect } from "react";
import { useAuth } from "@/lib/auth";
import { tradesApi, type TradeData } from "@/lib/api";

export default function TradesPage() {
  const { accessToken } = useAuth();
  const [trades, setTrades] = useState<TradeData[]>([]);
  const [positions, setPositions] = useState<Record<string, unknown> | null>(null);
  const [showBookForm, setShowBookForm] = useState(false);
  const [filter, setFilter] = useState<string>("");
  const [form, setForm] = useState({ side: "buy", option_type: "call", spot: "24000", strike: "24000", T: "0.08", r: "0.069", sigma: "0.14", quantity: "10" });
  const [loading, setLoading] = useState(false);

  const loadData = () => {
    if (!accessToken) return;
    tradesApi.list(accessToken, filter || undefined).then((d: any) => setTrades(d.trades));
    tradesApi.positions(accessToken).then(setPositions);
  };

  useEffect(() => { loadData(); }, [accessToken, filter]);

  const bookTrade = async () => {
    if (!accessToken) return;
    setLoading(true);
    try {
      await tradesApi.book({
        side: form.side, option_type: form.option_type,
        spot: +form.spot, strike: +form.strike, T: +form.T, r: +form.r, sigma: +form.sigma, quantity: +form.quantity,
      }, accessToken);
      setShowBookForm(false);
      loadData();
    } catch {} finally { setLoading(false); }
  };

  const closeTrade = async (id: string) => {
    if (!accessToken) return;
    await tradesApi.close(id, accessToken);
    loadData();
  };

  return (
    <div>
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold" style={{ color: "var(--color-text-primary)" }}>Trade Blotter</h1>
        <button onClick={() => setShowBookForm(!showBookForm)} className="btn-primary">
          {showBookForm ? "Cancel" : "+ Book Trade"}
        </button>
      </div>

      {/* Position Summary */}
      {positions && (
        <div className="grid grid-cols-4 gap-4 mb-6">
          <div className="card text-center py-4">
            <div className="metric-label mb-1">Open Trades</div>
            <div className="metric-value" style={{ color: "var(--color-accent-blue)" }}>{String((positions as Record<string, unknown>).n_open_trades)}</div>
          </div>
          <div className="card text-center py-4">
            <div className="metric-label mb-1">Positions</div>
            <div className="metric-value" style={{ color: "var(--color-accent-purple)" }}>{String((positions as Record<string, unknown>).n_positions)}</div>
          </div>
          <div className="card text-center py-4">
            <div className="metric-label mb-1">Total Notional</div>
            <div className="metric-value" style={{ color: "var(--color-accent-green)", fontFamily: "var(--font-mono)" }}>
              ₹{Number((positions as Record<string, unknown>).total_notional).toLocaleString()}
            </div>
          </div>
          <div className="card text-center py-4">
            <div className="metric-label mb-1">Filter</div>
            <select className="input-field text-xs" value={filter} onChange={e => setFilter(e.target.value)}>
              <option value="">All</option><option value="open">Open</option><option value="closed">Closed</option>
            </select>
          </div>
        </div>
      )}

      {/* Book Trade Form */}
      {showBookForm && (
        <div className="card mb-6">
          <h3 className="font-semibold text-sm mb-4" style={{ color: "var(--color-text-primary)" }}>New Trade</h3>
          <div className="grid grid-cols-4 gap-4">
            <div><label className="label">Side</label>
              <select className="input-field" value={form.side} onChange={e => setForm({ ...form, side: e.target.value })}>
                <option value="buy">Buy</option><option value="sell">Sell</option></select></div>
            <div><label className="label">Type</label>
              <select className="input-field" value={form.option_type} onChange={e => setForm({ ...form, option_type: e.target.value })}>
                <option value="call">Call</option><option value="put">Put</option></select></div>
            {[{ k: "spot", l: "Spot" }, { k: "strike", l: "Strike" }, { k: "T", l: "Maturity" }, { k: "sigma", l: "Vol" }, { k: "quantity", l: "Qty" }].map(f => (
              <div key={f.k}><label className="label">{f.l}</label>
                <input className="input-field" type="number" step="any" value={form[f.k as keyof typeof form]}
                  onChange={e => setForm({ ...form, [f.k]: e.target.value })} /></div>
            ))}
            <div className="flex items-end">
              <button onClick={bookTrade} className="btn-primary w-full" disabled={loading}>
                {loading ? "..." : "Book"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Trade Table */}
      <div className="card overflow-x-auto">
        <table className="data-table">
          <thead>
            <tr>
              <th>Date</th><th>Side</th><th>Type</th><th>Strike</th>
              <th>Premium</th><th>Qty</th><th>Notional</th><th>Status</th><th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {trades.map(t => (
              <tr key={t.id}>
                <td className="text-xs">{new Date(t.traded_at).toLocaleDateString()}</td>
                <td><span className={t.side === "buy" ? "badge badge-green" : "badge badge-red"}>{t.side}</span></td>
                <td>{t.option_type}</td>
                <td>{t.strike}</td>
                <td>₹{t.premium.toFixed(2)}</td>
                <td>{t.quantity}</td>
                <td>₹{t.notional.toFixed(2)}</td>
                <td><span className={`badge ${t.status === "open" ? "badge-blue" : t.status === "closed" ? "badge-green" : "badge-amber"}`}>{t.status}</span></td>
                <td>
                  {t.status === "open" && (
                    <button onClick={() => closeTrade(t.id)} className="text-xs font-semibold" style={{ color: "var(--color-accent-red)" }}>Close</button>
                  )}
                </td>
              </tr>
            ))}
            {trades.length === 0 && (
              <tr><td colSpan={9} className="text-center py-8" style={{ color: "var(--color-text-muted)" }}>No trades yet</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

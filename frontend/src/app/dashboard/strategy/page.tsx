"use client";

import { useState } from "react";
import { useAuth } from "@/lib/auth";
import { strategiesApi, type StrategyResult } from "@/lib/api";

const STRATEGIES = [
  { id: "long_call", name: "Long Call" }, { id: "long_put", name: "Long Put" },
  { id: "bull_call_spread", name: "Bull Call Spread" }, { id: "bear_put_spread", name: "Bear Put Spread" },
  { id: "straddle", name: "Straddle" }, { id: "strangle", name: "Strangle" },
  { id: "iron_condor", name: "Iron Condor" }, { id: "butterfly", name: "Butterfly" },
];

export default function StrategySimulator() {
  const { accessToken } = useAuth();
  const [stratId, setStratId] = useState("straddle");
  const [form, setForm] = useState({ S: "24000", K: "24000", T: "0.08", r: "0.069", sigma: "0.14" });
  const [result, setResult] = useState<StrategyResult | null>(null);
  const [loading, setLoading] = useState(false);

  const simulate = async () => {
    if (!accessToken) return;
    setLoading(true);
    try {
      const res = await strategiesApi.simulate(
        { strategy_id: stratId, S: +form.S, K: +form.K, T: +form.T, r: +form.r, sigma: +form.sigma },
        accessToken
      );
      setResult(res);
    } catch {}
    finally { setLoading(false); }
  };

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6" style={{ color: "var(--color-text-primary)" }}>Strategy Simulator</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card">
          <h3 className="font-semibold text-sm mb-4" style={{ color: "var(--color-text-primary)" }}>Strategy</h3>
          <div className="grid grid-cols-2 gap-2 mb-6">
            {STRATEGIES.map(s => (
              <button key={s.id} onClick={() => setStratId(s.id)}
                className={stratId === s.id ? "btn-primary text-xs py-2" : "btn-secondary text-xs py-2"}>
                {s.name}
              </button>
            ))}
          </div>

          <div className="space-y-4">
            {[{ k: "S", l: "Spot (S)" }, { k: "K", l: "ATM Strike (K)" }, { k: "T", l: "Maturity (T)" }, { k: "r", l: "Rate (r)" }, { k: "sigma", l: "Vol (σ)" }].map(f => (
              <div key={f.k}><label className="label">{f.l}</label>
                <input className="input-field" type="number" step="any" value={form[f.k as keyof typeof form]}
                  onChange={e => setForm({ ...form, [f.k]: e.target.value })} /></div>
            ))}
            <button onClick={simulate} className="btn-primary w-full" disabled={loading}>
              {loading ? "Simulating..." : "Simulate Strategy"}
            </button>
          </div>
        </div>

        <div className="lg:col-span-2">
          {result ? (
            <div className="space-y-4">
              {/* Net Greeks */}
              <div className="grid grid-cols-5 gap-3">
                {Object.entries(result.greeks).map(([k, v]) => (
                  <div key={k} className="card text-center py-4">
                    <div className="metric-label mb-1">Net {k.toUpperCase()}</div>
                    <div className="text-sm font-bold" style={{ fontFamily: "var(--font-mono)", color: "var(--color-accent-blue)" }}>
                      {Number(v).toFixed(4)}
                    </div>
                  </div>
                ))}
              </div>

              {/* P&L Summary */}
              <div className="grid grid-cols-4 gap-3">
                <div className="card text-center py-4">
                  <div className="metric-label mb-1">Max Profit</div>
                  <div className="text-sm font-bold metric-positive" style={{ fontFamily: "var(--font-mono)" }}>
                    {result.max_profit > 1e6 ? "Unlimited" : `₹${result.max_profit.toFixed(2)}`}
                  </div>
                </div>
                <div className="card text-center py-4">
                  <div className="metric-label mb-1">Max Loss</div>
                  <div className="text-sm font-bold metric-negative" style={{ fontFamily: "var(--font-mono)" }}>
                    ₹{result.max_loss.toFixed(2)}
                  </div>
                </div>
                <div className="card text-center py-4">
                  <div className="metric-label mb-1">Entry Premium</div>
                  <div className="text-sm font-bold" style={{ fontFamily: "var(--font-mono)", color: "var(--color-text-primary)" }}>
                    ₹{result.entry_premium.toFixed(2)}
                  </div>
                </div>
                <div className="card text-center py-4">
                  <div className="metric-label mb-1">Breakevens</div>
                  <div className="text-sm font-bold" style={{ fontFamily: "var(--font-mono)", color: "var(--color-accent-amber)" }}>
                    {result.breakevens.map((b: any) => Number(b).toFixed(0)).join(" / ")}
                  </div>
                </div>
              </div>

              {/* Legs */}
              <div className="card">
                <h3 className="font-semibold text-xs mb-3" style={{ color: "var(--color-text-secondary)" }}>LEGS</h3>
                <table className="data-table">
                  <thead><tr><th>Side</th><th>Type</th><th>Strike</th><th>Premium</th></tr></thead>
                  <tbody>
                    {result.legs.map((leg: any, i: any) => (
                      <tr key={i}>
                        <td><span className={leg.side === "buy" ? "badge badge-green" : "badge badge-red"}>{leg.side}</span></td>
                        <td>{leg.type}</td>
                        <td>{leg.strike}</td>
                        <td>₹{leg.premium.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="card flex items-center justify-center h-64">
              <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>Select a strategy and simulate</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

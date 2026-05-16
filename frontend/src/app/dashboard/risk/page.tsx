"use client";

import { useState } from "react";
import { useAuth } from "@/lib/auth";
import { riskApi } from "@/lib/api";

export default function RiskEngine() {
  const { accessToken } = useAuth();
  const [positions, setPositions] = useState([
    { type: "call", S: "24000", K: "24000", T: "0.25", r: "0.05", sigma: "0.2", qty: "10" },
    { type: "put", S: "24000", K: "23500", T: "0.25", r: "0.05", sigma: "0.25", qty: "5" },
  ]);
  const [config, setConfig] = useState({ method: "historical", confidence: "0.95", n_sims: "100000" });
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);

  const addPos = () => setPositions([...positions, { type: "call", S: "24000", K: "24000", T: "0.25", r: "0.05", sigma: "0.2", qty: "1" }]);
  const rmPos = (i: number) => setPositions(positions.filter((_, idx) => idx !== i));

  const run = async () => {
    if (!accessToken) return;
    setLoading(true);
    try {
      const res = await riskApi.portfolio({
        portfolio: positions.map(p => ({ type: p.type, S: +p.S, K: +p.K, T: +p.T, r: +p.r, sigma: +p.sigma, qty: +p.qty })),
        method: config.method, confidence: +config.confidence, n_sims: +config.n_sims, seed: 42,
      }, accessToken);
      setResult(res);
    } catch {} finally { setLoading(false); }
  };

  const stats = result?.pnl_statistics as Record<string, number> | undefined;

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6" style={{ color: "var(--color-text-primary)" }}>Risk Engine</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-sm" style={{ color: "var(--color-text-primary)" }}>Portfolio</h3>
            <button onClick={addPos} className="btn-secondary text-xs">+ Add</button>
          </div>

          {positions.map((p, i) => (
            <div key={i} className="flex gap-2 mb-3 items-end">
              <select className="input-field text-xs w-16" value={p.type}
                onChange={e => { const n = [...positions]; n[i].type = e.target.value; setPositions(n); }}>
                <option value="call">C</option><option value="put">P</option>
              </select>
              <input className="input-field text-xs w-20" placeholder="K" value={p.K}
                onChange={e => { const n = [...positions]; n[i].K = e.target.value; setPositions(n); }} />
              <input className="input-field text-xs w-14" placeholder="Qty" value={p.qty}
                onChange={e => { const n = [...positions]; n[i].qty = e.target.value; setPositions(n); }} />
              <input className="input-field text-xs w-14" placeholder="σ" value={p.sigma}
                onChange={e => { const n = [...positions]; n[i].sigma = e.target.value; setPositions(n); }} />
              <button onClick={() => rmPos(i)} className="text-xs" style={{ color: "var(--color-accent-red)" }}>✕</button>
            </div>
          ))}

          <hr className="my-4" style={{ borderColor: "var(--color-border)" }} />
          <div className="space-y-3">
            <div><label className="label">Method</label>
              <select className="input-field" value={config.method} onChange={e => setConfig({ ...config, method: e.target.value })}>
                <option value="historical">Historical</option><option value="parametric">Parametric</option><option value="monte_carlo">Monte Carlo</option>
              </select></div>
            <div><label className="label">Confidence</label>
              <input className="input-field" value={config.confidence} onChange={e => setConfig({ ...config, confidence: e.target.value })} /></div>
            <button onClick={run} className="btn-primary w-full" disabled={loading}>{loading ? "Computing..." : "Run Risk Analysis"}</button>
          </div>
        </div>

        <div className="lg:col-span-2">
          {result ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="card text-center py-5">
                  <div className="metric-label mb-2">VaR ({config.confidence})</div>
                  <div className="metric-value metric-negative" style={{ fontFamily: "var(--font-mono)" }}>₹{Number(result.VaR).toFixed(2)}</div>
                </div>
                <div className="card text-center py-5">
                  <div className="metric-label mb-2">CVaR / ES</div>
                  <div className="metric-value metric-negative" style={{ fontFamily: "var(--font-mono)" }}>₹{Number(result.CVaR).toFixed(2)}</div>
                </div>
                <div className="card text-center py-5">
                  <div className="metric-label mb-2">Method</div>
                  <div className="text-lg font-semibold" style={{ color: "var(--color-accent-blue)" }}>{String(result.method)}</div>
                </div>
                <div className="card text-center py-5">
                  <div className="metric-label mb-2">Portfolio Value</div>
                  <div className="metric-value metric-positive" style={{ fontFamily: "var(--font-mono)" }}>₹{Number(result.portfolio_value).toFixed(2)}</div>
                </div>
              </div>

              {stats && (
                <div className="card">
                  <h3 className="font-semibold text-xs mb-3" style={{ color: "var(--color-text-secondary)" }}>P&L DISTRIBUTION STATISTICS</h3>
                  <div className="grid grid-cols-4 gap-4">
                    {[
                      { l: "Mean", v: stats.mean },
                      { l: "Std Dev", v: stats.std },
                      { l: "Skewness", v: stats.skewness },
                      { l: "Kurtosis", v: stats.kurtosis },
                    ].map(s => (
                      <div key={s.l} className="text-center">
                        <div className="metric-label mb-1">{s.l}</div>
                        <div className="text-sm font-mono font-semibold" style={{ color: "var(--color-text-primary)" }}>{s.v?.toFixed(4)}</div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          ) : (
            <div className="card flex items-center justify-center h-64">
              <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>Build a portfolio and run risk analysis</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

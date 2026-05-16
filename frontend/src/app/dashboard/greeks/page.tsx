"use client";

import { useState } from "react";
import { useAuth } from "@/lib/auth";
import { greeksApi } from "@/lib/api";

export default function GreeksExplorer() {
  const { accessToken } = useAuth();
  const [form, setForm] = useState({ S: "100", K: "100", T: "1", r: "0.05", sigma: "0.2", option_type: "call", method: "analytical" });
  const [result, setResult] = useState<Record<string, number> | null>(null);
  const [loading, setLoading] = useState(false);

  const handleCompute = async () => {
    if (!accessToken) return;
    setLoading(true);
    try {
      const res = await greeksApi.calculate({
        S: +form.S, K: +form.K, T: +form.T, r: +form.r, sigma: +form.sigma,
        option_type: form.option_type, method: form.method,
      }, accessToken);
      setResult(res.greeks);
    } catch {}
    finally { setLoading(false); }
  };

  const greekCards = [
    { key: "delta", symbol: "Δ", label: "Delta", desc: "Price sensitivity" },
    { key: "gamma", symbol: "Γ", label: "Gamma", desc: "Delta convexity" },
    { key: "vega", symbol: "V", label: "Vega", desc: "Vol sensitivity" },
    { key: "theta", symbol: "Θ", label: "Theta", desc: "Time decay" },
    { key: "rho", symbol: "ρ", label: "Rho", desc: "Rate sensitivity" },
  ];

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6" style={{ color: "var(--color-text-primary)" }}>Greeks Explorer</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card">
          <div className="space-y-4">
            <div><label className="label">Type</label>
              <select className="input-field" value={form.option_type}
                onChange={e => setForm({ ...form, option_type: e.target.value })}>
                <option value="call">Call</option><option value="put">Put</option>
              </select></div>
            {[{ k: "S", l: "Spot (S)" }, { k: "K", l: "Strike (K)" }, { k: "T", l: "Maturity (T)" }, { k: "r", l: "Rate (r)" }, { k: "sigma", l: "Vol (σ)" }].map(f => (
              <div key={f.k}><label className="label">{f.l}</label>
                <input className="input-field" type="number" step="any" value={form[f.k as keyof typeof form]}
                  onChange={e => setForm({ ...form, [f.k]: e.target.value })} /></div>
            ))}
            <div><label className="label">Method</label>
              <select className="input-field" value={form.method}
                onChange={e => setForm({ ...form, method: e.target.value })}>
                <option value="analytical">Analytical</option><option value="numerical">Numerical</option>
              </select></div>
            <button onClick={handleCompute} className="btn-primary w-full" disabled={loading}>
              {loading ? "Computing..." : "Compute Greeks"}
            </button>
          </div>
        </div>

        <div className="lg:col-span-2">
          <div className="grid grid-cols-5 gap-4 mb-6">
            {greekCards.map(g => (
              <div key={g.key} className="card text-center py-5">
                <div className="text-xs font-semibold mb-1" style={{ color: "var(--color-text-secondary)" }}>
                  {g.label} ({g.symbol})
                </div>
                <div className="text-xl font-bold" style={{ fontFamily: "var(--font-mono)", color: "var(--color-accent-blue)" }}>
                  {result ? (result[g.key] >= 0 ? "+" : "") + result[g.key].toFixed(6) : "—"}
                </div>
                <div className="text-[10px] mt-1" style={{ color: "var(--color-text-muted)" }}>{g.desc}</div>
              </div>
            ))}
          </div>

          {result && (
            <div className="card">
              <h3 className="font-semibold text-xs mb-3" style={{ color: "var(--color-text-secondary)" }}>INTERPRETATION</h3>
              <div className="space-y-2 text-sm" style={{ color: "var(--color-text-primary)" }}>
                <p>• <strong>Delta {result.delta >= 0 ? "+" : ""}{result.delta.toFixed(4)}</strong>: A ₹1 move in spot changes the option price by ₹{Math.abs(result.delta).toFixed(4)}</p>
                <p>• <strong>Gamma {result.gamma.toFixed(4)}</strong>: Delta changes by {result.gamma.toFixed(4)} per ₹1 spot move</p>
                <p>• <strong>Theta {result.theta.toFixed(4)}</strong>: The option loses ₹{Math.abs(result.theta / 365).toFixed(4)} per day</p>
                <p>• <strong>Vega {result.vega.toFixed(4)}</strong>: A 1% vol increase changes the price by ₹{(result.vega / 100).toFixed(4)}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

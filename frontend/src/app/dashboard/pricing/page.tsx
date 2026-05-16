"use client";

import { useState } from "react";
import { useAuth } from "@/lib/auth";
import { pricingApi } from "@/lib/api";

export default function PricingLab() {
  const { accessToken } = useAuth();
  const [model, setModel] = useState<"bs" | "mc" | "binomial">("bs");
  const [form, setForm] = useState({ S: "24000", K: "24000", T: "0.08", r: "0.069", sigma: "0.14", option_type: "call", n_sims: "100000", method: "standard", style: "european", N: "200" });
  const [result, setResult] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handlePrice = async () => {
    if (!accessToken) return;
    setLoading(true); setError(""); setResult(null);
    try {
      const base = { S: +form.S, K: +form.K, T: +form.T, r: +form.r, sigma: +form.sigma, option_type: form.option_type };
      let res;
      if (model === "bs") {
        res = await pricingApi.blackScholes(base, accessToken);
      } else if (model === "mc") {
        res = await pricingApi.monteCarlo({ ...base, n_sims: +form.n_sims, method: form.method }, accessToken);
      } else {
        res = await pricingApi.binomial({ ...base, style: form.style, N: +form.N }, accessToken);
      }
      setResult(res);
    } catch (err: unknown) { setError(err instanceof Error ? err.message : "Error"); }
    finally { setLoading(false); }
  };

  const fields = [
    { key: "S", label: "Spot (S)" }, { key: "K", label: "Strike (K)" }, { key: "T", label: "Maturity (T)" },
    { key: "r", label: "Rate (r)" }, { key: "sigma", label: "Vol (σ)" },
  ];

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6" style={{ color: "var(--color-text-primary)" }}>Pricing Lab</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Input Panel */}
        <div className="card">
          <h3 className="font-semibold text-sm mb-4" style={{ color: "var(--color-text-primary)" }}>Model</h3>
          <div className="flex gap-2 mb-6">
            {[{ v: "bs", l: "Black-Scholes" }, { v: "mc", l: "Monte Carlo" }, { v: "binomial", l: "Binomial" }].map(m => (
              <button key={m.v} onClick={() => setModel(m.v as "bs" | "mc" | "binomial")}
                className={model === m.v ? "btn-primary text-xs" : "btn-secondary text-xs"}>
                {m.l}
              </button>
            ))}
          </div>

          <div className="space-y-4">
            <div>
              <label className="label">Type</label>
              <select className="input-field" value={form.option_type}
                onChange={e => setForm({ ...form, option_type: e.target.value })}>
                <option value="call">Call</option>
                <option value="put">Put</option>
              </select>
            </div>

            {fields.map(f => (
              <div key={f.key}>
                <label className="label">{f.label}</label>
                <input className="input-field" type="number" step="any"
                  value={form[f.key as keyof typeof form]}
                  onChange={e => setForm({ ...form, [f.key]: e.target.value })} />
              </div>
            ))}

            {model === "mc" && (
              <>
                <div><label className="label">Simulations</label>
                  <input className="input-field" type="number" value={form.n_sims}
                    onChange={e => setForm({ ...form, n_sims: e.target.value })} /></div>
                <div><label className="label">Method</label>
                  <select className="input-field" value={form.method}
                    onChange={e => setForm({ ...form, method: e.target.value })}>
                    <option value="standard">Standard</option>
                    <option value="antithetic">Antithetic</option>
                    <option value="control">Control Variate</option>
                  </select></div>
              </>
            )}

            {model === "binomial" && (
              <>
                <div><label className="label">Steps (N)</label>
                  <input className="input-field" type="number" value={form.N}
                    onChange={e => setForm({ ...form, N: e.target.value })} /></div>
                <div><label className="label">Style</label>
                  <select className="input-field" value={form.style}
                    onChange={e => setForm({ ...form, style: e.target.value })}>
                    <option value="european">European</option>
                    <option value="american">American</option>
                  </select></div>
              </>
            )}

            <button onClick={handlePrice} className="btn-primary w-full" disabled={loading}>
              {loading ? "Computing..." : "Price Option"}
            </button>

            {error && <p className="text-xs mt-2" style={{ color: "var(--color-accent-red)" }}>{error}</p>}
          </div>
        </div>

        {/* Result Panel */}
        <div className="lg:col-span-2">
          {result ? (
            <div className="space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="card text-center py-6">
                  <div className="metric-label mb-2">Price</div>
                  <div className="metric-value metric-positive" style={{ fontFamily: "var(--font-mono)" }}>
                    ₹{Number(result.price).toFixed(4)}
                  </div>
                </div>
                <div className="card text-center py-6">
                  <div className="metric-label mb-2">Model</div>
                  <div className="text-lg font-semibold" style={{ color: "var(--color-accent-blue)" }}>
                    {String(result.model).toUpperCase()}
                  </div>
                </div>
                <div className="card text-center py-6">
                  <div className="metric-label mb-2">Elapsed</div>
                  <div className="text-lg font-mono font-semibold" style={{ color: "var(--color-text-primary)" }}>
                    {Number(result.elapsed_ms).toFixed(2)}ms
                  </div>
                </div>
                <div className="card text-center py-6">
                  <div className="metric-label mb-2">Type</div>
                  <div className="text-lg font-semibold" style={{ color: form.option_type === "call" ? "var(--color-accent-green)" : "var(--color-accent-red)" }}>
                    {form.option_type.toUpperCase()}
                  </div>
                </div>
              </div>

              {/* Full Result JSON */}
              <div className="card">
                <h3 className="font-semibold text-xs mb-3" style={{ color: "var(--color-text-secondary)" }}>RAW OUTPUT</h3>
                <pre className="text-xs overflow-auto max-h-64 leading-relaxed"
                  style={{ color: "var(--color-text-primary)", fontFamily: "var(--font-mono)" }}>
                  {JSON.stringify(result, null, 2)}
                </pre>
              </div>
            </div>
          ) : (
            <div className="card flex items-center justify-center h-64">
              <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>
                Select parameters and click "Price Option" to see results
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

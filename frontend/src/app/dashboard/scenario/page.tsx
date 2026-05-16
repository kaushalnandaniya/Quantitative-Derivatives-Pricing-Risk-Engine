"use client";

import { useState } from "react";
import { useAuth } from "@/lib/auth";
import { scenarioApi } from "@/lib/api";

export default function ScenarioAnalysis() {
  const { accessToken } = useAuth();
  const [positions, setPositions] = useState([
    { type: "call", S: "24000", K: "24000", T: "0.08", r: "0.069", sigma: "0.14", qty: "10" },
  ]);
  const [heatmapCfg, setHeatmapCfg] = useState({ x_axis: "spot", y_axis: "vol", n_points: "12" });
  const [heatmap, setHeatmap] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(false);

  const addPos = () => setPositions([...positions, { type: "call", S: "24000", K: "24000", T: "0.08", r: "0.069", sigma: "0.14", qty: "1" }]);

  const runHeatmap = async () => {
    if (!accessToken) return;
    setLoading(true);
    try {
      const res = await scenarioApi.heatmap({
        positions: positions.map(p => ({ type: p.type, S: +p.S, K: +p.K, T: +p.T, r: +p.r, sigma: +p.sigma, qty: +p.qty })),
        x_axis: heatmapCfg.x_axis, y_axis: heatmapCfg.y_axis, n_points: +heatmapCfg.n_points,
      }, accessToken);
      setHeatmap(res);
    } catch {} finally { setLoading(false); }
  };

  const zMatrix = heatmap?.z_matrix as number[][] | undefined;
  const xValues = heatmap?.x_values as number[] | undefined;
  const yValues = heatmap?.y_values as number[] | undefined;

  return (
    <div>
      <h1 className="text-2xl font-bold mb-6" style={{ color: "var(--color-text-primary)" }}>Scenario Analysis</h1>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="card">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-sm" style={{ color: "var(--color-text-primary)" }}>Portfolio</h3>
            <button onClick={addPos} className="btn-secondary text-xs">+ Add</button>
          </div>
          {positions.map((p, i) => (
            <div key={i} className="flex gap-2 mb-2">
              <select className="input-field text-xs w-16" value={p.type}
                onChange={e => { const n = [...positions]; n[i].type = e.target.value; setPositions(n); }}>
                <option value="call">C</option><option value="put">P</option>
              </select>
              <input className="input-field text-xs w-20" value={p.K} onChange={e => { const n = [...positions]; n[i].K = e.target.value; setPositions(n); }} />
              <input className="input-field text-xs w-12" value={p.qty} onChange={e => { const n = [...positions]; n[i].qty = e.target.value; setPositions(n); }} />
            </div>
          ))}
          <hr className="my-4" style={{ borderColor: "var(--color-border)" }} />
          <h3 className="font-semibold text-sm mb-3" style={{ color: "var(--color-text-primary)" }}>Heatmap Config</h3>
          <div className="space-y-3">
            <div><label className="label">X Axis</label>
              <select className="input-field" value={heatmapCfg.x_axis} onChange={e => setHeatmapCfg({ ...heatmapCfg, x_axis: e.target.value })}>
                <option value="spot">Spot</option><option value="vol">Vol</option><option value="time">Time</option>
              </select></div>
            <div><label className="label">Y Axis</label>
              <select className="input-field" value={heatmapCfg.y_axis} onChange={e => setHeatmapCfg({ ...heatmapCfg, y_axis: e.target.value })}>
                <option value="vol">Vol</option><option value="spot">Spot</option><option value="time">Time</option>
              </select></div>
            <div><label className="label">Grid Size</label>
              <input className="input-field" type="number" value={heatmapCfg.n_points} onChange={e => setHeatmapCfg({ ...heatmapCfg, n_points: e.target.value })} /></div>
            <button onClick={runHeatmap} className="btn-primary w-full" disabled={loading}>
              {loading ? "Computing..." : "Generate Heatmap"}
            </button>
          </div>
        </div>

        <div className="lg:col-span-2">
          {zMatrix && xValues && yValues ? (
            <div className="card">
              <h3 className="font-semibold text-xs mb-4" style={{ color: "var(--color-text-secondary)" }}>
                P&L HEATMAP — {heatmapCfg.x_axis.toUpperCase()} vs {heatmapCfg.y_axis.toUpperCase()}
              </h3>
              <div className="overflow-auto">
                <table className="w-full text-xs" style={{ fontFamily: "var(--font-mono)" }}>
                  <thead>
                    <tr>
                      <th className="p-1 text-right" style={{ color: "var(--color-text-muted)" }}>{heatmapCfg.y_axis}↓ / {heatmapCfg.x_axis}→</th>
                      {xValues.map((x, i) => <th key={i} className="p-1 text-center" style={{ color: "var(--color-text-secondary)" }}>{x.toFixed(x < 100 ? 3 : 0)}</th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {yValues.map((y, yi) => (
                      <tr key={yi}>
                        <td className="p-1 text-right font-semibold" style={{ color: "var(--color-text-secondary)" }}>{y.toFixed(y < 100 ? 3 : 0)}</td>
                        {zMatrix[yi].map((val, xi) => {
                          const maxAbs = Math.max(...zMatrix.flat().map(Math.abs));
                          const norm = maxAbs > 0 ? val / maxAbs : 0;
                          const bg = val >= 0
                            ? `rgba(0,204,102,${Math.min(Math.abs(norm) * 0.6, 0.6)})`
                            : `rgba(255,59,48,${Math.min(Math.abs(norm) * 0.6, 0.6)})`;
                          return (
                            <td key={xi} className="p-1.5 text-center text-[10px] font-semibold" style={{ background: bg, color: "var(--color-text-primary)" }}>
                              {val.toFixed(1)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="text-[10px] mt-3 text-center" style={{ color: "var(--color-text-muted)" }}>
                Base value: ₹{Number(heatmap?.base_value).toFixed(2)} | Green = profit, Red = loss
              </p>
            </div>
          ) : (
            <div className="card flex items-center justify-center h-64">
              <p className="text-sm" style={{ color: "var(--color-text-muted)" }}>Generate a heatmap to visualize P&L sensitivity</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

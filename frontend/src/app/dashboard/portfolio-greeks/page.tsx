"use client";

import { useState, useEffect, useCallback } from "react";
import { useAuth } from "@/lib/auth";
import { tradesApi } from "@/lib/api";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, ResponsiveContainer,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip
} from "recharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface PositionGreeks {
  index: number; type: string; S: number; K: number; T: number; qty: number;
  greeks_weighted: Record<string, number>;
}

export default function PortfolioGreeksPage() {
  const { accessToken } = useAuth();
  const [positions, setPositions] = useState<PositionGreeks[]>([]);
  const [totals, setTotals] = useState<Record<string, number> | null>(null);
  const [loading, setLoading] = useState(false);
  const [trades, setTrades] = useState<any[]>([]);

  const loadTrades = useCallback(async () => {
    if (!accessToken) return;
    try {
      const data = await tradesApi.list(accessToken, "open");
      setTrades(data.trades || []);
    } catch { /* no trades */ }
  }, [accessToken]);

  useEffect(() => { loadTrades(); }, [loadTrades]);

  const computeGreeks = async () => {
    if (!accessToken || trades.length === 0) return;
    setLoading(true);
    try {
      const portfolio = trades.map(t => ({
        type: t.option_type || "call",
        S: t.spot_price || 24000,
        K: t.strike || 24000,
        T: t.T || 0.02,
        r: t.r || 0.069,
        sigma: t.sigma || 0.15,
        qty: t.side === "sell" ? -(t.quantity || 1) : (t.quantity || 1),
      }));

      const res = await fetch(`${API_URL}/greeks/portfolio`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${accessToken}` },
        body: JSON.stringify({ portfolio }),
      });
      const data = await res.json();
      setPositions(data.positions || []);
      setTotals(data.totals || null);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  };

  const greekNames = ["delta", "gamma", "vega", "theta", "rho"];

  // Radar data (normalized)
  const radarData = totals ? greekNames.map(g => ({
    greek: g.charAt(0).toUpperCase() + g.slice(1),
    value: Math.abs(totals[g]),
  })) : [];

  // Bar data per position
  const barData = positions.map(p => ({
    label: `${p.type.toUpperCase()} ${p.K}`,
    delta: p.greeks_weighted.delta,
    gamma: p.greeks_weighted.gamma * 1000,
    vega: p.greeks_weighted.vega,
    theta: p.greeks_weighted.theta,
  }));

  return (
    <div className="flex flex-col gap-4">
      {/* Controls */}
      <div className="card shadow-sm flex items-center gap-4">
        <h2 className="font-bold text-sm text-[var(--color-text-primary)]">Portfolio Greeks</h2>
        <span className="text-xs text-[var(--color-text-muted)]">{trades.length} open position(s)</span>
        <button onClick={computeGreeks} disabled={loading || trades.length === 0} className="btn-primary px-5 py-2 text-xs font-bold ml-auto">
          {loading ? "Computing..." : "Compute Greeks"}
        </button>
      </div>

      {totals ? (
        <>
          {/* Net Greeks Cards */}
          <div className="card grid grid-cols-5 gap-4 shadow-sm">
            {greekNames.map(g => {
              const val = totals[g];
              const symbol = { delta: "Δ", gamma: "Γ", vega: "ν", theta: "Θ", rho: "ρ" }[g] || g;
              return (
                <div key={g} className="text-center">
                  <div className="text-2xl font-bold text-[var(--color-text-muted)] mb-1">{symbol}</div>
                  <div className={`font-mono font-bold text-lg ${val >= 0 ? "text-[var(--color-accent-green)]" : "text-[var(--color-accent-red)]"}`}>
                    {val >= 0 ? "+" : ""}{val.toFixed(4)}
                  </div>
                  <div className="text-[10px] text-[var(--color-text-secondary)] uppercase font-semibold mt-1">{g}</div>
                </div>
              );
            })}
          </div>

          {/* Charts Row */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Radar */}
            <div className="card shadow-sm min-h-[300px]">
              <h3 className="text-xs font-bold uppercase text-[var(--color-text-secondary)] tracking-wider mb-3">Exposure Profile</h3>
              <ResponsiveContainer width="100%" height={250}>
                <RadarChart data={radarData}>
                  <PolarGrid stroke="var(--color-border-subtle)" />
                  <PolarAngleAxis dataKey="greek" tick={{ fill: "var(--color-text-secondary)", fontSize: 11 }} />
                  <Radar dataKey="value" stroke="var(--color-accent-blue)" fill="var(--color-accent-blue)" fillOpacity={0.2} strokeWidth={2} />
                </RadarChart>
              </ResponsiveContainer>
            </div>

            {/* Greeks by Position */}
            <div className="card shadow-sm min-h-[300px]">
              <h3 className="text-xs font-bold uppercase text-[var(--color-text-secondary)] tracking-wider mb-3">Delta by Position</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={barData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" vertical={false} />
                  <XAxis dataKey="label" stroke="var(--color-text-muted)" fontSize={9} />
                  <YAxis stroke="var(--color-text-muted)" fontSize={10} />
                  <Tooltip contentStyle={{ background: "var(--color-bg-card)", borderColor: "var(--color-border-subtle)", borderRadius: 6, fontSize: "11px" }} />
                  <Bar dataKey="delta" name="Delta" fill="var(--color-accent-blue)" fillOpacity={0.7} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Positions Table */}
          <div className="card shadow-sm">
            <h3 className="text-xs font-bold uppercase text-[var(--color-text-secondary)] tracking-wider mb-3">Per-Position Greeks</h3>
            <div className="overflow-x-auto">
              <table className="data-table text-xs w-full">
                <thead>
                  <tr>
                    <th>#</th><th>Type</th><th>Strike</th><th>Qty</th>
                    <th>Delta</th><th>Gamma</th><th>Vega</th><th>Theta</th><th>Rho</th>
                  </tr>
                </thead>
                <tbody>
                  {positions.map(p => (
                    <tr key={p.index}>
                      <td>{p.index + 1}</td>
                      <td className="uppercase font-semibold">{p.type}</td>
                      <td className="font-mono">{p.K}</td>
                      <td className="font-mono">{p.qty}</td>
                      {greekNames.map(g => (
                        <td key={g} className={`font-mono ${p.greeks_weighted[g] >= 0 ? "text-[var(--color-accent-green)]" : "text-[var(--color-accent-red)]"}`}>
                          {p.greeks_weighted[g].toFixed(4)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      ) : (
        <div className="card shadow-sm text-center py-12 text-[var(--color-text-muted)]">
          {trades.length === 0
            ? "No open trades found. Book trades from the Strategy Builder to see portfolio Greeks."
            : "Click \"Compute Greeks\" to analyze your portfolio exposure"}
        </div>
      )}
    </div>
  );
}

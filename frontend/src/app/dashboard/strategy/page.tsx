"use client";

import { useState, useEffect } from "react";
import { useAuth } from "@/lib/auth";
import { strategiesApi, type StrategyResult } from "@/lib/api";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine
} from "recharts";

const STRATEGIES = [
  { id: "long_call", name: "Buy Call", group: "Bullish" }, 
  { id: "long_put", name: "Buy Put", group: "Bearish" },
  { id: "bull_call_spread", name: "Bull Call Spread", group: "Bullish" }, 
  { id: "bear_put_spread", name: "Bear Put Spread", group: "Bearish" },
  { id: "straddle", name: "Straddle", group: "Neutral" }, 
  { id: "strangle", name: "Strangle", group: "Neutral" },
  { id: "iron_condor", name: "Iron Condor", group: "Neutral" }, 
  { id: "butterfly", name: "Butterfly", group: "Neutral" },
];

export default function StrategyBuilder() {
  const { accessToken } = useAuth();
  const [stratId, setStratId] = useState("straddle");
  const [form, setForm] = useState({ S: "24000", K: "24000", T: "0.08", r: "0.069", sigma: "0.14", lot_size: "1" });
  const [result, setResult] = useState<StrategyResult | null>(null);
  const [loading, setLoading] = useState(false);

  const simulate = async () => {
    if (!accessToken) return;
    setLoading(true);
    try {
      const res = await strategiesApi.simulate(
        { strategy_id: stratId, S: +form.S, K: +form.K, T: +form.T, r: +form.r, sigma: +form.sigma, lot_size: +form.lot_size },
        accessToken
      );
      setResult(res);
    } catch {}
    finally { setLoading(false); }
  };

  // Auto-simulate on mount or when strategy changes if we have a token
  useEffect(() => {
    if (accessToken) simulate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stratId, accessToken]);

  // Prepare chart data
  const chartData = result ? result.spots.map((spot, i) => ({
    spot: Number(spot.toFixed(2)),
    pnl: Number(result.pnl[i].toFixed(2))
  })) : [];

  // Calculate gradient offset for green/red split
  const gradientOffset = () => {
    if (!result) return 0;
    const dataMax = result.max_profit;
    const dataMin = result.max_loss;
    if (dataMax <= 0) return 0;
    if (dataMin >= 0) return 1;
    return dataMax / (dataMax - dataMin);
  };
  const off = gradientOffset();

  return (
    <div className="flex flex-col lg:flex-row gap-6">
      
      {/* LEFT SIDEBAR: BUILDER */}
      <div className="w-full lg:w-[380px] flex-shrink-0 flex flex-col gap-4">
        
        {/* Controls Card */}
        <div className="card shadow-sm h-full flex flex-col">
          <div className="flex justify-between items-center border-b border-[var(--color-border-subtle)] pb-4 mb-4">
            <h2 className="font-bold text-[var(--color-text-primary)]">Strategy Builder</h2>
            <div className="text-xs font-semibold px-2 py-1 bg-[var(--color-bg-elevated)] rounded">Settings</div>
          </div>

          <div className="mb-6">
            <div className="flex justify-between items-center mb-3">
              <span className="text-sm font-bold">Ready-made</span>
              <span className="text-xs text-[var(--color-accent-blue)] cursor-pointer hover:underline">Learn Strategies</span>
            </div>
            
            <div className="grid grid-cols-2 gap-2">
              {STRATEGIES.map(s => (
                <button 
                  key={s.id} 
                  onClick={() => setStratId(s.id)}
                  className={`border rounded flex flex-col items-center justify-center p-3 transition-colors ${
                    stratId === s.id 
                      ? "border-[var(--color-accent-blue)] bg-[rgba(41,98,255,0.05)] text-[var(--color-accent-blue)]" 
                      : "border-[var(--color-border-subtle)] hover:border-[var(--color-border)] hover:bg-[var(--color-bg-hover)] text-[var(--color-text-secondary)]"
                  }`}
                >
                  <span className="text-xs font-semibold text-center leading-tight">{s.name}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="space-y-3 mb-6">
            <h3 className="text-sm font-bold border-b border-[var(--color-border-subtle)] pb-2">Parameters</h3>
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="label">Spot Price</label>
                <input className="input-field" type="number" step="any" value={form.S} onChange={e => setForm({ ...form, S: e.target.value })} />
              </div>
              <div>
                <label className="label">ATM Strike</label>
                <input className="input-field" type="number" step="any" value={form.K} onChange={e => setForm({ ...form, K: e.target.value })} />
              </div>
              <div>
                <label className="label">Expiry (Years)</label>
                <input className="input-field" type="number" step="any" value={form.T} onChange={e => setForm({ ...form, T: e.target.value })} />
              </div>
              <div>
                <label className="label">Volatility (σ)</label>
                <input className="input-field" type="number" step="any" value={form.sigma} onChange={e => setForm({ ...form, sigma: e.target.value })} />
              </div>
              <div>
                <label className="label">Risk-free Rate</label>
                <input className="input-field" type="number" step="any" value={form.r} onChange={e => setForm({ ...form, r: e.target.value })} />
              </div>
              <div>
                <label className="label">Lot Size</label>
                <input className="input-field" type="number" step="1" value={form.lot_size} onChange={e => setForm({ ...form, lot_size: e.target.value })} />
              </div>
            </div>
          </div>

          <button onClick={simulate} className="btn-primary w-full py-3 text-[13px]" disabled={loading}>
            {loading ? "Calculating..." : "Update Payoff Graph"}
          </button>

          {/* Important Info Disclaimer */}
          <div className="mt-auto pt-6">
            <div className="bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] rounded p-4">
              <h4 className="text-xs font-bold mb-1 flex items-center gap-1 text-[var(--color-text-primary)]">
                Important info
                <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" /></svg>
              </h4>
              <p className="text-[11px] text-[var(--color-text-secondary)] leading-relaxed">
                The profit and loss are projections, and they depend on premia, liquidity, IV, etc. While we make the best effort to ensure they are right, the actual numbers may vary.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* RIGHT MAIN AREA: ANALYSIS */}
      <div className="flex-1 flex flex-col gap-4 min-w-0">
        
        {/* Top Metrics Row */}
        {result ? (
          <div className="card grid grid-cols-2 lg:grid-cols-5 gap-4 shadow-sm items-center">
            <div>
              <div className="text-[11px] font-semibold text-[var(--color-text-secondary)] mb-1 uppercase tracking-wide">Max Profit</div>
              <div className="font-mono font-bold text-[15px] positive">
                {result.max_profit > 1e6 ? "Unlimited" : `+₹${result.max_profit.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`}
              </div>
            </div>
            <div className="border-l border-[var(--color-border-subtle)] pl-4">
              <div className="text-[11px] font-semibold text-[var(--color-text-secondary)] mb-1 uppercase tracking-wide">Max Loss</div>
              <div className="font-mono font-bold text-[15px] negative">
                {result.max_loss < -1e6 ? "Unlimited" : `-₹${Math.abs(result.max_loss).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`}
              </div>
            </div>
            <div className="border-l border-[var(--color-border-subtle)] pl-4">
              <div className="text-[11px] font-semibold text-[var(--color-text-secondary)] mb-1 uppercase tracking-wide">Breakeven</div>
              <div className="font-mono font-bold text-[13px] text-[var(--color-text-primary)]">
                {result.breakevens.length > 0 ? result.breakevens.map((b: any) => Number(b).toFixed(1)).join(" / ") : "None"}
              </div>
            </div>
            <div className="border-l border-[var(--color-border-subtle)] pl-4">
              <div className="text-[11px] font-semibold text-[var(--color-text-secondary)] mb-1 uppercase tracking-wide">Net Premium</div>
              <div className="font-mono font-bold text-[15px]" style={{ color: result.entry_premium > 0 ? "var(--color-accent-red)" : "var(--color-accent-green)"}}>
                {result.entry_premium > 0 ? `Pay ₹${result.entry_premium.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}` : `Receive ₹${Math.abs(result.entry_premium).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}`}
              </div>
            </div>
            <div className="border-l border-[var(--color-border-subtle)] pl-4">
              <div className="text-[11px] font-semibold text-[var(--color-text-secondary)] mb-1 uppercase tracking-wide">Risk / Reward</div>
              <div className="font-mono font-bold text-[15px] text-[var(--color-text-primary)]">
                {result.max_loss < -1e6 || result.max_profit > 1e6 ? "N/A" : `1 : ${Math.abs(result.max_profit / result.max_loss).toFixed(2)}`}
              </div>
            </div>
          </div>
        ) : (
          <div className="card h-20 flex items-center justify-center text-sm text-[var(--color-text-muted)]">Loading metrics...</div>
        )}

        {/* Chart Area */}
        <div className="card flex-1 min-h-[400px] flex flex-col shadow-sm">
          <div className="flex justify-between items-center mb-6">
            <div className="flex gap-4 border-b border-[var(--color-border-subtle)] w-full">
              <div className="px-1 py-2 text-sm font-bold border-b-2 border-[var(--color-accent-blue)] text-[var(--color-text-primary)]">Payoff Graph</div>
              <div className="px-1 py-2 text-sm font-semibold text-[var(--color-text-muted)] cursor-not-allowed">P&L Table</div>
              <div className="px-1 py-2 text-sm font-semibold text-[var(--color-text-muted)] cursor-not-allowed">Greeks</div>
            </div>
          </div>

          <div className="flex-1 w-full relative">
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData} margin={{ top: 10, right: 30, left: 20, bottom: 20 }}>
                  <defs>
                    <linearGradient id="splitColor" x1="0" y1="0" x2="0" y2="1">
                      <stop offset={off} stopColor="var(--color-accent-green)" stopOpacity={0.2} />
                      <stop offset={off} stopColor="var(--color-accent-red)" stopOpacity={0.2} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" vertical={false} />
                  <XAxis 
                    dataKey="spot" 
                    stroke="var(--color-text-muted)" 
                    fontSize={11}
                    tickFormatter={(val) => `₹${val.toLocaleString()}`}
                    dy={10}
                  />
                  <YAxis 
                    stroke="var(--color-text-muted)" 
                    fontSize={11}
                    tickFormatter={(val) => `₹${val.toLocaleString()}`}
                    dx={-5}
                  />
                  <Tooltip 
                    contentStyle={{ background: "var(--color-bg-card)", borderColor: "var(--color-border-subtle)", borderRadius: 8, fontSize: '12px' }}
                    labelFormatter={(val) => `Spot: ₹${val}`}
                    formatter={(val: any) => [
                      `${Number(val) >= 0 ? "+" : ""}₹${Number(val).toLocaleString(undefined, {minimumFractionDigits: 2})}`, 
                      "Projected P&L"
                    ]}
                  />
                  {/* Zero Line */}
                  <ReferenceLine y={0} stroke="var(--color-text-muted)" strokeDasharray="3 3" />
                  {/* Spot Price Line */}
                  <ReferenceLine x={+form.S} stroke="var(--color-accent-blue)" strokeDasharray="3 3">
                  </ReferenceLine>
                  
                  <Area 
                    type="monotone" 
                    dataKey="pnl" 
                    stroke="var(--color-text-primary)" 
                    strokeWidth={2}
                    fill="url(#splitColor)" 
                    isAnimationActive={false}
                  />
                </AreaChart>
              </ResponsiveContainer>
            ) : (
              <div className="absolute inset-0 flex items-center justify-center text-sm text-[var(--color-text-muted)]">
                {loading ? "Calculating Payoff..." : "No data available"}
              </div>
            )}
            
            {/* Spot Label overlay */}
            {chartData.length > 0 && (
              <div className="absolute top-0 right-10 bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] px-3 py-1 rounded text-xs font-mono">
                Spot: ₹{Number(form.S).toLocaleString()}
              </div>
            )}
          </div>
        </div>

        {/* Lower Data Tables */}
        {result && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            
            {/* Net Greeks */}
            <div className="card shadow-sm">
              <h3 className="font-semibold text-xs mb-3 text-[var(--color-text-secondary)] uppercase tracking-wider">Net Greeks Position</h3>
              <table className="data-table">
                <tbody>
                  {Object.entries(result.greeks).map(([k, v]) => (
                    <tr key={k}>
                      <td className="capitalize font-semibold text-[var(--color-text-muted)]">{k}</td>
                      <td className="text-right font-bold text-[var(--color-text-primary)]">{Number(v).toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Position Legs */}
            <div className="card shadow-sm">
              <h3 className="font-semibold text-xs mb-3 text-[var(--color-text-secondary)] uppercase tracking-wider">Strategy Legs</h3>
              <table className="data-table text-xs">
                <thead>
                  <tr>
                    <th>B/S</th>
                    <th>Type</th>
                    <th>Strike</th>
                    <th className="text-right">Price</th>
                  </tr>
                </thead>
                <tbody>
                  {result.legs.map((leg: any, i: any) => (
                    <tr key={i}>
                      <td>
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${leg.side === "buy" ? "bg-[rgba(41,98,255,0.1)] text-[var(--color-accent-blue)]" : "bg-[rgba(242,54,69,0.1)] text-[var(--color-accent-red)]"}`}>
                          {leg.side === "buy" ? "B" : "S"}
                        </span>
                      </td>
                      <td className="uppercase font-semibold text-[var(--color-text-secondary)]">{leg.type}</td>
                      <td className="font-mono">{leg.strike}</td>
                      <td className="text-right font-mono text-[var(--color-text-primary)]">₹{leg.premium.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            
          </div>
        )}
      </div>
    </div>
  );
}

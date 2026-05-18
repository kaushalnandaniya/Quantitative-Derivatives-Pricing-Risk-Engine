"use client";

import { useState } from "react";
import { useAuth } from "@/lib/auth";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
  BarChart, Bar
} from "recharts";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const STRATEGIES = [
  { id: "bull_call_spread", name: "Bull Call Spread" },
  { id: "bear_put_spread", name: "Bear Put Spread" },
  { id: "long_straddle", name: "Long Straddle" },
  { id: "long_strangle", name: "Long Strangle" },
  { id: "iron_condor", name: "Iron Condor" },
  { id: "butterfly", name: "Butterfly" },
];

interface BacktestResult {
  entry_date: string; expiry_date: string;
  entry_spot: number; expiry_spot: number;
  spot_change_pct: number; entry_premium: number;
  pnl: number; win: boolean;
}

export default function BacktestPage() {
  const { accessToken } = useAuth();
  const [stratId, setStratId] = useState("bull_call_spread");
  const [symbol, setSymbol] = useState("NIFTY");
  const [weeks, setWeeks] = useState(12);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<BacktestResult[]>([]);
  const [summary, setSummary] = useState<Record<string, number> | null>(null);
  const [equity, setEquity] = useState<number[]>([]);

  const runBacktest = async () => {
    if (!accessToken) return;
    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/backtest/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${accessToken}` },
        body: JSON.stringify({ strategy_id: stratId, symbol, lookback_weeks: weeks }),
      });
      const data = await res.json();
      setResults(data.results || []);
      setSummary(data.summary || null);
      setEquity(data.equity_curve || []);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  };

  const equityData = equity.map((v, i) => ({ week: i, equity: v }));
  const pnlData = results.map(r => ({ date: r.entry_date.slice(5), pnl: r.pnl }));

  return (
    <div className="flex flex-col gap-4">
      {/* Controls */}
      <div className="card shadow-sm">
        <div className="flex flex-wrap items-end gap-4">
          <div>
            <label className="text-[10px] font-semibold text-[var(--color-text-muted)] uppercase block mb-1">Symbol</label>
            <select className="input-field text-sm" value={symbol} onChange={e => setSymbol(e.target.value)}>
              <option value="NIFTY">NIFTY</option>
              <option value="BANKNIFTY">BANKNIFTY</option>
              <option value="RELIANCE">RELIANCE</option>
            </select>
          </div>
          <div>
            <label className="text-[10px] font-semibold text-[var(--color-text-muted)] uppercase block mb-1">Strategy</label>
            <select className="input-field text-sm" value={stratId} onChange={e => setStratId(e.target.value)}>
              {STRATEGIES.map(s => <option key={s.id} value={s.id}>{s.name}</option>)}
            </select>
          </div>
          <div>
            <label className="text-[10px] font-semibold text-[var(--color-text-muted)] uppercase block mb-1">Lookback (weeks)</label>
            <input type="number" className="input-field text-sm w-20" value={weeks} onChange={e => setWeeks(+e.target.value)} min={4} max={52} />
          </div>
          <button onClick={runBacktest} disabled={loading} className="btn-primary px-6 py-2.5 text-sm font-bold">
            {loading ? "Running..." : "Run Backtest"}
          </button>
        </div>
      </div>

      {summary && (
        <>
          {/* Summary Stats */}
          <div className="card grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3 shadow-sm">
            {[
              { label: "Total Trades", val: summary.total_trades, color: "" },
              { label: "Wins", val: summary.wins, color: "var(--color-accent-green)" },
              { label: "Losses", val: summary.losses, color: "var(--color-accent-red)" },
              { label: "Win Rate", val: `${summary.win_rate}%`, color: summary.win_rate >= 50 ? "var(--color-accent-green)" : "var(--color-accent-red)" },
              { label: "Avg P&L", val: `₹${summary.avg_pnl}`, color: summary.avg_pnl >= 0 ? "var(--color-accent-green)" : "var(--color-accent-red)" },
              { label: "Total P&L", val: `₹${summary.total_pnl}`, color: summary.total_pnl >= 0 ? "var(--color-accent-green)" : "var(--color-accent-red)" },
              { label: "Best Trade", val: `₹${summary.best_trade}`, color: "var(--color-accent-green)" },
              { label: "Worst Trade", val: `₹${summary.worst_trade}`, color: "var(--color-accent-red)" },
            ].map((m, i) => (
              <div key={i}>
                <div className="text-[10px] font-semibold text-[var(--color-text-secondary)] uppercase tracking-wide mb-1">{m.label}</div>
                <div className="font-mono font-bold text-sm" style={{ color: m.color || "var(--color-text-primary)" }}>{m.val}</div>
              </div>
            ))}
          </div>

          {/* Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {/* Equity Curve */}
            <div className="card shadow-sm min-h-[300px]">
              <h3 className="text-xs font-bold uppercase text-[var(--color-text-secondary)] tracking-wider mb-3">Equity Curve</h3>
              <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={equityData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" vertical={false} />
                  <XAxis dataKey="week" stroke="var(--color-text-muted)" fontSize={10} label={{ value: "Week", position: "insideBottom", offset: -5, fontSize: 10, fill: "var(--color-text-muted)" }} />
                  <YAxis stroke="var(--color-text-muted)" fontSize={10} tickFormatter={v => `₹${v}`} />
                  <Tooltip contentStyle={{ background: "var(--color-bg-card)", borderColor: "var(--color-border-subtle)", borderRadius: 6, fontSize: "11px" }} />
                  <ReferenceLine y={0} stroke="var(--color-text-muted)" strokeDasharray="3 3" />
                  <Area type="monotone" dataKey="equity" stroke="var(--color-accent-blue)" strokeWidth={2} fill="rgba(41,98,255,0.1)" isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* P&L per Trade */}
            <div className="card shadow-sm min-h-[300px]">
              <h3 className="text-xs font-bold uppercase text-[var(--color-text-secondary)] tracking-wider mb-3">P&L per Trade</h3>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={pnlData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" vertical={false} />
                  <XAxis dataKey="date" stroke="var(--color-text-muted)" fontSize={9} />
                  <YAxis stroke="var(--color-text-muted)" fontSize={10} tickFormatter={v => `₹${v}`} />
                  <Tooltip contentStyle={{ background: "var(--color-bg-card)", borderColor: "var(--color-border-subtle)", borderRadius: 6, fontSize: "11px" }} />
                  <ReferenceLine y={0} stroke="var(--color-text-muted)" strokeDasharray="3 3" />
                  <Bar dataKey="pnl" name="P&L" isAnimationActive={false} fill="var(--color-accent-blue)">
                    {pnlData.map((entry, i) => (
                      <rect key={i} fill={entry.pnl >= 0 ? "var(--color-accent-green)" : "var(--color-accent-red)"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Results Table */}
          <div className="card shadow-sm">
            <h3 className="text-xs font-bold uppercase text-[var(--color-text-secondary)] tracking-wider mb-3">Trade History</h3>
            <div className="overflow-x-auto">
              <table className="data-table text-xs w-full">
                <thead>
                  <tr>
                    <th>Entry Date</th><th>Expiry</th><th>Entry Spot</th><th>Expiry Spot</th>
                    <th>Spot Chg %</th><th>Premium</th><th>P&L</th><th>Result</th>
                  </tr>
                </thead>
                <tbody>
                  {results.map((r, i) => (
                    <tr key={i}>
                      <td className="font-mono">{r.entry_date}</td>
                      <td className="font-mono">{r.expiry_date}</td>
                      <td className="font-mono">{r.entry_spot.toLocaleString()}</td>
                      <td className="font-mono">{r.expiry_spot.toLocaleString()}</td>
                      <td className="font-mono" style={{ color: r.spot_change_pct >= 0 ? "var(--color-accent-green)" : "var(--color-accent-red)" }}>{r.spot_change_pct}%</td>
                      <td className="font-mono">₹{r.entry_premium}</td>
                      <td className="font-mono font-bold" style={{ color: r.pnl >= 0 ? "var(--color-accent-green)" : "var(--color-accent-red)" }}>
                        {r.pnl >= 0 ? "+" : ""}₹{r.pnl}
                      </td>
                      <td>
                        <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${r.win ? "bg-[rgba(8,153,129,0.1)] text-[var(--color-accent-green)]" : "bg-[rgba(242,54,69,0.1)] text-[var(--color-accent-red)]"}`}>
                          {r.win ? "WIN" : "LOSS"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {!summary && !loading && (
        <div className="card shadow-sm text-center py-12 text-[var(--color-text-muted)]">
          Select a strategy and click &quot;Run Backtest&quot; to see historical performance
        </div>
      )}
    </div>
  );
}

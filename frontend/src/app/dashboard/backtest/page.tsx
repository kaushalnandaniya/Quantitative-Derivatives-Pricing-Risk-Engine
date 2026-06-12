"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import { useAuth } from "@/lib/auth";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
  BarChart, Bar, Cell, PieChart, Pie, Legend
} from "recharts";
import TradingViewChart, { TradeMarker } from "@/components/TradingViewChart";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// ────────────────────────────────────────────────────────────────
// Types
// ────────────────────────────────────────────────────────────────
interface SearchResult {
  symbol: string;
  name: string;
  exchange: string;
}

interface SavedStrategy {
  id: string;
  name: string;
  pine_script: string;
  description?: string;
  created_at: string;
  updated_at: string;
}

interface PineBacktestResult {
  trades: Array<{
    entry_bar: number; exit_bar: number;
    entry_date: string; exit_date: string;
    side: string; entry_price: number;
    exit_price: number; pnl: number; pnl_pct: number;
  }>;
  summary: {
    total_trades: number; wins: number; losses: number; breakevens: number;
    win_rate: number; total_pnl: number; avg_pnl: number;
    best_trade: number; worst_trade: number;
    max_drawdown: number; max_drawdown_pct: number; sharpe_ratio: number; profit_factor: number;
    gross_profit: number; gross_loss: number; open_pnl: number; commission: number;
    cagr: number; return_on_initial_capital: number; avg_bars_in_trade: number;
  };
  equity_curve: number[];
  dates: string[];
  ohlcv?: Array<{ time: string, open: number, high: number, low: number, close: number, volume: number }>;
  symbol?: string;
}

const DEFAULT_SCRIPT = `//@version=5
strategy("SMA Crossover", overlay=true)

// Moving Averages
fast = ta.sma(close, 10)
slow = ta.sma(close, 30)

// Long entry on golden cross
if ta.crossover(fast, slow)
    strategy.entry("Long", strategy.long)

// Close on death cross
if ta.crossunder(fast, slow)
    strategy.close("Long")`;

// ────────────────────────────────────────────────────────────────
// Component
// ────────────────────────────────────────────────────────────────
export default function BacktestPage() {
  const { accessToken } = useAuth();

  // Search state
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);
  const [selectedSymbol, setSelectedSymbol] = useState("RELIANCE");
  const searchRef = useRef<HTMLDivElement>(null);
  const searchTimeout = useRef<NodeJS.Timeout | null>(null);

  // Pine editor state
  const [pineScript, setPineScript] = useState(DEFAULT_SCRIPT);
  const [strategyName, setStrategyName] = useState("");
  const [editingStrategyId, setEditingStrategyId] = useState<string | null>(null);

  // Saved strategies
  const [savedStrategies, setSavedStrategies] = useState<SavedStrategy[]>([]);

  // Backtest settings
  const [periodDays, setPeriodDays] = useState(365);
  const [loading, setLoading] = useState(false);

  // Results
  const [results, setResults] = useState<PineBacktestResult | null>(null);

  // ──────────────────────────
  // Search
  // ──────────────────────────
  const searchSymbols = useCallback(async (q: string) => {
    if (q.length < 1) { setSearchResults([]); setShowDropdown(false); return; }
    try {
      const res = await fetch(`${API_URL}/backtest/search?q=${encodeURIComponent(q)}`);
      const data = await res.json();
      setSearchResults(data.results || []);
      setShowDropdown(true);
    } catch (e) { console.error(e); }
  }, []);

  const handleSearchInput = (val: string) => {
    setSearchQuery(val);
    if (searchTimeout.current) clearTimeout(searchTimeout.current);
    searchTimeout.current = setTimeout(() => searchSymbols(val), 250);
  };

  const selectSymbol = (sym: string) => {
    setSelectedSymbol(sym);
    setSearchQuery("");
    setShowDropdown(false);
  };

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (searchRef.current && !searchRef.current.contains(e.target as Node)) {
        setShowDropdown(false);
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, []);

  // ──────────────────────────
  // Saved Strategies CRUD
  // ──────────────────────────
  const loadSavedStrategies = useCallback(async () => {
    if (!accessToken) return;
    try {
      const res = await fetch(`${API_URL}/backtest/strategies`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (!res.ok) return;
      const data = await res.json();
      setSavedStrategies(data.strategies || []);
    } catch (e) { console.error(e); }
  }, [accessToken]);

  useEffect(() => { loadSavedStrategies(); }, [loadSavedStrategies]);

  const saveStrategy = async () => {
    if (!accessToken) return;
    const name = strategyName.trim();
    const script = pineScript.trim();
    if (!name || !script) { alert("Enter a strategy name and code."); return; }

    try {
      if (editingStrategyId) {
        // Update existing
        await fetch(`${API_URL}/backtest/strategies/${editingStrategyId}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${accessToken}` },
          body: JSON.stringify({ name, pine_script: script }),
        });
      } else {
        // Create new
        await fetch(`${API_URL}/backtest/strategies`, {
          method: "POST",
          headers: { "Content-Type": "application/json", Authorization: `Bearer ${accessToken}` },
          body: JSON.stringify({ name, pine_script: script }),
        });
      }
      setEditingStrategyId(null);
      loadSavedStrategies();
    } catch (e) { console.error(e); }
  };

  const loadStrategy = (s: SavedStrategy) => {
    setPineScript(s.pine_script);
    setStrategyName(s.name);
    setEditingStrategyId(s.id);
  };

  const deleteStrategy = async (id: string) => {
    if (!accessToken) return;
    try {
      await fetch(`${API_URL}/backtest/strategies/${id}`, {
        method: "DELETE",
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      if (editingStrategyId === id) {
        setEditingStrategyId(null);
        setStrategyName("");
      }
      loadSavedStrategies();
    } catch (e) { console.error(e); }
  };

  // ──────────────────────────
  // Run Backtest
  // ──────────────────────────
  const runBacktest = async () => {
    const script = pineScript.trim();
    if (!script) { alert("Write a Pine Script strategy first."); return; }

    setLoading(true);
    setResults(null);
    try {
      const res = await fetch(`${API_URL}/backtest/run-pine`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ pine_script: script, symbol: selectedSymbol, period_days: periodDays }),
      });
      const data = await res.json();
      if (res.ok) {
        setResults(data);
      } else {
        alert(data.detail || "Backtest failed");
      }
    } catch (e) {
      alert("Failed to run backtest. Check API connection.");
      console.error(e);
    }
    setLoading(false);
  };

  // ──────────────────────────
  // Chart data
  // ──────────────────────────
  const equityData = results?.equity_curve?.map((v, i) => ({ trade: i, equity: v })) || [];
  const pnlData = results?.trades?.map((t, i) => ({
    idx: i + 1,
    pnl: t.pnl,
    date: t.exit_date?.split(" ")[0] || "",
  })) || [];

  // Trade markers for Candlestick Chart
  const chartMarkers: TradeMarker[] = results?.trades?.flatMap((t) => {
    const m: TradeMarker[] = [];
    if (t.entry_date) {
      m.push({
        time: t.entry_date.split(" ")[0],
        position: t.side === "long" ? "belowBar" : "aboveBar",
        color: t.side === "long" ? "#26a69a" : "#ef5350",
        shape: t.side === "long" ? "arrowUp" : "arrowDown",
      });
    }
    if (t.exit_date) {
      m.push({
        time: t.exit_date.split(" ")[0],
        position: "inBar",
        color: t.pnl >= 0 ? "#26a69a" : "#ef5350",
        shape: "circle",
      });
    }
    return m;
  }) || [];

  // Advanced Chart Data
  const profitStructureData = results?.summary ? [
    { name: 'Total Profit', value: results.summary.gross_profit, fill: '#26a69a' },
    { name: 'Total Loss', value: -results.summary.gross_loss, fill: '#ef5350' },
    { name: 'Open P&L', value: results.summary.open_pnl, fill: '#fb8c00' },
    { name: 'Commission', value: -results.summary.commission, fill: '#42a5f5' },
    { name: 'Total P&L', value: results.summary.total_pnl, fill: '#2962ff' }
  ] : [];

  const tradeDistData = results?.summary ? [
    { name: 'Winners', value: results.summary.wins, fill: '#26a69a' },
    { name: 'Losers', value: results.summary.losses, fill: '#ef5350' },
    { name: 'Breakevens', value: results.summary.breakevens, fill: '#fb8c00' }
  ].filter(d => d.value > 0) : [];

  const roiBins = (() => {
    if (!results?.trades || results.trades.length === 0) return [];
    const pcts = results.trades.map(t => t.pnl_pct);
    const minPct = Math.min(...pcts, 0);
    const maxPct = Math.max(...pcts, 0);
    const binCount = 20;
    const binSize = Math.max((maxPct - minPct) / binCount, 0.1);
    
    const bins = Array(binCount).fill(0).map((_, i) => ({
      range: `${(minPct + i * binSize).toFixed(1)}%`,
      winners: 0,
      losers: 0
    }));

    results.trades.forEach(t => {
      const binIndex = Math.max(0, Math.min(Math.floor((t.pnl_pct - minPct) / binSize), binCount - 1));
      if (t.pnl_pct >= 0) bins[binIndex].winners++;
      else bins[binIndex].losers++; // We map losers as positive count so bar goes UP like in TradingView
    });
    return bins;
  })();

  // ──────────────────────────
  // Render
  // ──────────────────────────
  return (
    <div className="flex flex-col gap-4 page-enter">

      {/* ═══ TOP BAR: Search + Settings + Run ═══ */}
      <div className="card shadow-sm">
        <div className="flex flex-wrap items-end gap-4">
          {/* Stock Search */}
          <div className="relative flex-1 min-w-[200px]" ref={searchRef}>
            <label className="text-[10px] font-semibold text-[var(--color-text-muted)] uppercase block mb-1">Stock Symbol</label>
            <div className="flex items-center gap-2">
              <span className="px-2.5 py-1.5 rounded text-xs font-bold bg-[rgba(41,98,255,0.12)] text-[var(--color-accent-blue)] border border-[rgba(41,98,255,0.25)] tracking-wide">
                {selectedSymbol}
              </span>
              <input
                type="text"
                className="input-field text-sm flex-1"
                placeholder="Search stock (e.g. RELIANCE, NIFTY, TCS...)"
                value={searchQuery}
                onChange={e => handleSearchInput(e.target.value)}
                autoComplete="off"
              />
            </div>
            {/* Dropdown */}
            {showDropdown && searchResults.length > 0 && (
              <div className="absolute top-full left-0 right-0 mt-1 bg-[var(--color-bg-card)] border border-[var(--color-border)] rounded shadow-2xl z-50 max-h-[220px] overflow-y-auto">
                {searchResults.map(s => (
                  <button
                    key={s.symbol}
                    className="w-full flex justify-between items-center px-3 py-2 text-left hover:bg-[var(--color-bg-hover)] transition-colors"
                    onClick={() => selectSymbol(s.symbol)}
                  >
                    <span className="font-mono font-bold text-sm text-[var(--color-text-primary)]">{s.symbol}</span>
                    <span className="text-[11px] text-[var(--color-text-secondary)]">{s.name}</span>
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Period */}
          <div>
            <label className="text-[10px] font-semibold text-[var(--color-text-muted)] uppercase block mb-1">Lookback (days)</label>
            <input
              type="number" className="input-field text-sm w-24"
              value={periodDays} onChange={e => setPeriodDays(+e.target.value)}
              min={30} max={3650}
            />
          </div>

          {/* Run */}
          <button onClick={runBacktest} disabled={loading} className="btn-primary px-6 py-2.5 text-sm font-bold whitespace-nowrap">
            {loading ? (
              <span className="flex items-center gap-2">
                <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24"><circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" /><path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" /></svg>
                Running…
              </span>
            ) : "▶ Run Backtest"}
          </button>
        </div>
      </div>

      {/* ═══ MAIN: Editor + Sidebar ═══ */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-4">

        {/* ── Pine Script Editor ── */}
        <div className="card shadow-sm flex flex-col">
          <div className="flex items-center justify-between mb-3">
            <h3 className="text-xs font-bold uppercase text-[var(--color-text-secondary)] tracking-wider flex items-center gap-2">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent-blue)" strokeWidth="2"><path d="M16 18l6-6-6-6M8 6l-6 6 6 6" /></svg>
              Pine Script Editor
            </h3>
            <div className="flex items-center gap-2">
              <input
                type="text" className="input-field text-xs py-1.5 w-40"
                placeholder="Strategy name…"
                value={strategyName}
                onChange={e => setStrategyName(e.target.value)}
              />
              <button onClick={saveStrategy} disabled={!accessToken} className="btn-secondary text-xs py-1.5 px-3 whitespace-nowrap">
                {editingStrategyId ? "Update" : "Save"}
              </button>
              {editingStrategyId && (
                <button
                  onClick={() => { setEditingStrategyId(null); setStrategyName(""); }}
                  className="text-[10px] text-[var(--color-text-muted)] hover:text-[var(--color-accent-red)] cursor-pointer"
                >✕ cancel</button>
              )}
            </div>
          </div>

          <textarea
            className="w-full min-h-[420px] p-4 rounded bg-[#0d0e12] text-[#c9d1d9] border border-[var(--color-border)] font-mono text-[13px] leading-relaxed resize-y outline-none focus:border-[var(--color-accent-blue)] transition-colors"
            style={{ tabSize: 4 }}
            spellCheck={false}
            value={pineScript}
            onChange={e => setPineScript(e.target.value)}
            placeholder={`// Write your Pine Script strategy here\n//@version=5\nstrategy("My Strategy", overlay=true)\n\nfast = ta.sma(close, 10)\nslow = ta.sma(close, 30)\n\nif ta.crossover(fast, slow)\n    strategy.entry("Long", strategy.long)\n\nif ta.crossunder(fast, slow)\n    strategy.close("Long")`}
          />
        </div>

        {/* ── Saved Strategies Sidebar ── */}
        <div className="card shadow-sm flex flex-col">
          <h3 className="text-xs font-bold uppercase text-[var(--color-text-secondary)] tracking-wider mb-3 flex items-center gap-2">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="var(--color-accent-blue)" strokeWidth="2"><path d="M19 21l-7-5-7 5V5a2 2 0 012-2h10a2 2 0 012 2z" /></svg>
            Saved Strategies
          </h3>

          {!accessToken ? (
            <p className="text-xs text-[var(--color-text-muted)]">Login to save and load strategies</p>
          ) : savedStrategies.length === 0 ? (
            <p className="text-xs text-[var(--color-text-muted)]">No saved strategies yet</p>
          ) : (
            <div className="flex flex-col gap-1.5 overflow-y-auto max-h-[400px]">
              {savedStrategies.map(s => (
                <div key={s.id}
                  className={`flex items-center justify-between px-2.5 py-2 rounded border text-xs cursor-pointer transition-all group ${
                    editingStrategyId === s.id
                      ? "bg-[rgba(41,98,255,0.08)] border-[var(--color-accent-blue)]"
                      : "bg-[var(--color-bg-elevated)] border-[var(--color-border)] hover:border-[var(--color-text-muted)]"
                  }`}
                >
                  <button className="text-left flex-1 font-medium text-[var(--color-text-primary)]" onClick={() => loadStrategy(s)}>
                    {s.name}
                  </button>
                  <button
                    className="ml-2 text-[var(--color-text-muted)] hover:text-[var(--color-accent-red)] opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={(e) => { e.stopPropagation(); deleteStrategy(s.id); }}
                    title="Delete"
                  >✕</button>
                </div>
              ))}
            </div>
          )}

          {/* Quick templates */}
          <div className="mt-4 pt-3 border-t border-[var(--color-border)]">
            <p className="text-[10px] font-semibold text-[var(--color-text-muted)] uppercase mb-2">Quick Templates</p>
            <div className="flex flex-col gap-1">
              {[
                { name: "SMA Crossover", script: DEFAULT_SCRIPT },
                { name: "RSI Mean Reversion", script: `//@version=5
strategy("RSI Mean Reversion", overlay=true)

rsiVal = ta.rsi(close, 14)

if rsiVal < 30
    strategy.entry("Long", strategy.long)

if rsiVal > 70
    strategy.close("Long")` },
                { name: "EMA Trend Follow", script: `//@version=5
strategy("EMA Trend", overlay=true)

fast = ta.ema(close, 12)
slow = ta.ema(close, 26)

if ta.crossover(fast, slow)
    strategy.entry("Long", strategy.long)

if ta.crossunder(fast, slow)
    strategy.close("Long")` },
              ].map((t, i) => (
                <button key={i}
                  className="text-left text-[11px] px-2 py-1.5 rounded bg-[var(--color-bg-elevated)] border border-[var(--color-border)] text-[var(--color-text-secondary)] hover:text-[var(--color-accent-blue)] hover:border-[var(--color-accent-blue)] transition-all cursor-pointer"
                  onClick={() => { setPineScript(t.script); setStrategyName(t.name); setEditingStrategyId(null); }}
                >
                  {t.name}
                </button>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* ═══ RESULTS ═══ */}
      {results && results.summary && (
        <>
          {/* Summary Stats */}
          <div className="card grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-3 shadow-sm">
            {[
              { label: "Total Trades", val: results.summary.total_trades, color: "" },
              { label: "Win Rate", val: `${results.summary.win_rate}%`, color: results.summary.win_rate >= 50 ? "var(--color-accent-green)" : "var(--color-accent-red)" },
              { label: "Total P&L", val: `₹${results.summary.total_pnl?.toLocaleString()}`, color: results.summary.total_pnl >= 0 ? "var(--color-accent-green)" : "var(--color-accent-red)" },
              { label: "CAGR", val: `${results.summary.cagr}%`, color: results.summary.cagr >= 0 ? "var(--color-accent-green)" : "var(--color-accent-red)" },
              { label: "Sharpe Ratio", val: results.summary.sharpe_ratio, color: results.summary.sharpe_ratio >= 1 ? "var(--color-accent-green)" : "" },
              { label: "Max Drawdown", val: `${results.summary.max_drawdown_pct}%`, color: "var(--color-accent-red)" },
              { label: "Profit Factor", val: results.summary.profit_factor >= 9999 ? "∞" : results.summary.profit_factor, color: results.summary.profit_factor > 1 ? "var(--color-accent-green)" : "" },
              { label: "Open P&L", val: `₹${results.summary.open_pnl?.toLocaleString()}`, color: results.summary.open_pnl >= 0 ? "var(--color-accent-green)" : "var(--color-accent-red)" },
            ].map((m, i) => (
              <div key={i}>
                <div className="text-[10px] font-semibold text-[var(--color-text-secondary)] uppercase tracking-wide mb-1">{m.label}</div>
                <div className="font-mono font-bold text-lg" style={{ color: m.color || "var(--color-text-primary)" }}>{m.val}</div>
              </div>
            ))}
          </div>

          {/* TradingView Chart */}
          {results.ohlcv && (
            <div className="card shadow-sm mt-4 min-h-[450px]">
              <h3 className="text-xs font-bold uppercase text-[var(--color-text-secondary)] tracking-wider mb-3">
                {selectedSymbol} — Price Action & Trades
              </h3>
              <TradingViewChart data={results.ohlcv} markers={chartMarkers} />
            </div>
          )}

          {/* Additional Charts */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4">
            <div className="card shadow-sm min-h-[300px]">
              <h3 className="text-xs font-bold uppercase text-[var(--color-text-secondary)] tracking-wider mb-3">
                Equity Curve — {selectedSymbol}
              </h3>
              <ResponsiveContainer width="100%" height={260}>
                <AreaChart data={equityData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" vertical={false} />
                  <XAxis dataKey="trade" stroke="var(--color-text-muted)" fontSize={10} label={{ value: "Trade #", position: "insideBottom", offset: -5, fontSize: 10, fill: "var(--color-text-muted)" }} />
                  <YAxis stroke="var(--color-text-muted)" fontSize={10} tickFormatter={v => `₹${v}`} />
                  <Tooltip contentStyle={{ background: "var(--color-bg-card)", borderColor: "var(--color-border-subtle)", borderRadius: 6, fontSize: "11px" }} />
                  <ReferenceLine y={0} stroke="var(--color-text-muted)" strokeDasharray="3 3" />
                  <Area type="monotone" dataKey="equity" stroke="var(--color-accent-blue)" strokeWidth={2} fill="rgba(41,98,255,0.1)" isAnimationActive={false} />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            <div className="card shadow-sm min-h-[300px]">
              <h3 className="text-xs font-bold uppercase text-[var(--color-text-secondary)] tracking-wider mb-3">P&L per Trade</h3>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={pnlData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" vertical={false} />
                  <XAxis dataKey="idx" stroke="var(--color-text-muted)" fontSize={10} label={{ value: "Trade #", position: "insideBottom", offset: -5, fontSize: 10, fill: "var(--color-text-muted)" }} />
                  <YAxis stroke="var(--color-text-muted)" fontSize={10} tickFormatter={v => `₹${v}`} />
                  <Tooltip contentStyle={{ background: "var(--color-bg-card)", borderColor: "var(--color-border-subtle)", borderRadius: 6, fontSize: "11px" }} />
                  <ReferenceLine y={0} stroke="var(--color-text-muted)" strokeDasharray="3 3" />
                  <Bar dataKey="pnl" name="P&L" isAnimationActive={false}>
                    {pnlData.map((entry, i) => (
                      <Cell key={i} fill={entry.pnl >= 0 ? "var(--color-accent-green)" : "var(--color-accent-red)"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Advanced Analytics */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 mt-4 mb-4">
            {/* Profit Structure */}
            <div className="card shadow-sm min-h-[300px]">
              <h3 className="text-xs font-bold uppercase text-[var(--color-text-secondary)] tracking-wider mb-3">Profit Structure</h3>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={profitStructureData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" vertical={false} />
                  <XAxis dataKey="name" stroke="var(--color-text-muted)" fontSize={10} />
                  <YAxis stroke="var(--color-text-muted)" fontSize={10} tickFormatter={v => `₹${v}`} />
                  <Tooltip contentStyle={{ background: "var(--color-bg-card)", borderColor: "var(--color-border-subtle)", borderRadius: 6, fontSize: "11px" }} cursor={{fill: 'rgba(255, 255, 255, 0.05)'}} />
                  <ReferenceLine y={0} stroke="var(--color-text-muted)" strokeDasharray="3 3" />
                  <Bar dataKey="value" isAnimationActive={false}>
                    {profitStructureData.map((entry, i) => (
                      <Cell key={i} fill={entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Trades Distribution & ROI */}
            <div className="card shadow-sm min-h-[300px]">
              <h3 className="text-xs font-bold uppercase text-[var(--color-text-secondary)] tracking-wider mb-3">Trades Analysis</h3>
              <div className="flex flex-col md:flex-row h-[260px] items-center">
                <div className="w-full md:w-1/2 h-full">
                  <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                      <Pie
                        data={tradeDistData}
                        cx="50%" cy="50%"
                        innerRadius={60} outerRadius={80}
                        paddingAngle={2}
                        dataKey="value"
                        isAnimationActive={false}
                      >
                        {tradeDistData.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={entry.fill} />
                        ))}
                      </Pie>
                      <Tooltip contentStyle={{ background: "var(--color-bg-card)", borderColor: "var(--color-border-subtle)", borderRadius: 6, fontSize: "11px" }} />
                      <Legend verticalAlign="middle" align="right" layout="vertical" wrapperStyle={{ fontSize: "11px" }} />
                    </PieChart>
                  </ResponsiveContainer>
                </div>
                <div className="w-full md:w-1/2 flex flex-col justify-center px-4 space-y-6">
                  <div className="flex justify-between">
                    <div>
                      <div className="text-[10px] text-[var(--color-text-secondary)] uppercase mb-1">Avg P&L per Trade</div>
                      <div className={`font-mono text-lg font-bold ${results.summary.avg_pnl >= 0 ? "text-[var(--color-accent-green)]" : "text-[var(--color-accent-red)]"}`}>
                        {results.summary.avg_pnl >= 0 ? "+" : ""}₹{results.summary.avg_pnl}
                      </div>
                    </div>
                    <div>
                      <div className="text-[10px] text-[var(--color-text-secondary)] uppercase mb-1">Avg Bars in Trade</div>
                      <div className="font-mono text-lg font-bold">{results.summary.avg_bars_in_trade}</div>
                    </div>
                  </div>
                  <div className="flex justify-between">
                    <div>
                      <div className="text-[10px] text-[var(--color-text-secondary)] uppercase mb-1">Largest Profit</div>
                      <div className="font-mono text-md font-bold text-[var(--color-accent-green)]">₹{results.summary.best_trade}</div>
                    </div>
                    <div>
                      <div className="text-[10px] text-[var(--color-text-secondary)] uppercase mb-1">Largest Loss</div>
                      <div className="font-mono text-md font-bold text-[var(--color-accent-red)]">₹{results.summary.worst_trade}</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            {/* ROI Distribution */}
            <div className="card shadow-sm min-h-[300px] lg:col-span-2">
              <h3 className="text-xs font-bold uppercase text-[var(--color-text-secondary)] tracking-wider mb-3">ROI Distribution</h3>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={roiBins} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" vertical={false} />
                  <XAxis dataKey="range" stroke="var(--color-text-muted)" fontSize={10} tick={{ fill: "var(--color-text-muted)" }} />
                  <YAxis stroke="var(--color-text-muted)" fontSize={10} />
                  <Tooltip 
                    contentStyle={{ background: "var(--color-bg-card)", borderColor: "var(--color-border-subtle)", borderRadius: 6, fontSize: "11px" }}
                    cursor={{fill: 'rgba(255, 255, 255, 0.05)'}}
                  />
                  <Legend verticalAlign="top" align="right" wrapperStyle={{ fontSize: "11px", paddingBottom: "10px" }} />
                  <Bar dataKey="losers" name="Losers" fill="#ef5350" isAnimationActive={false} />
                  <Bar dataKey="winners" name="Winners" fill="#26a69a" isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Trade Log */}
          <div className="card shadow-sm">
            <h3 className="text-xs font-bold uppercase text-[var(--color-text-secondary)] tracking-wider mb-3">Trade Log</h3>
            <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
              <table className="data-table text-xs w-full">
                <thead>
                  <tr>
                    <th>#</th><th>Side</th><th>Entry Date</th><th>Exit Date</th>
                    <th>Entry Price</th><th>Exit Price</th><th>P&L</th><th>P&L %</th>
                  </tr>
                </thead>
                <tbody>
                  {results.trades.map((t, i) => (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td>
                        <span className={`badge ${t.side === "long" ? "badge-green" : "badge-red"}`}>
                          {t.side.toUpperCase()}
                        </span>
                      </td>
                      <td className="font-mono">{t.entry_date?.split(" ")[0]}</td>
                      <td className="font-mono">{t.exit_date?.split(" ")[0]}</td>
                      <td className="font-mono">₹{t.entry_price?.toFixed(2)}</td>
                      <td className="font-mono">₹{t.exit_price?.toFixed(2)}</td>
                      <td className={`font-mono font-bold ${t.pnl >= 0 ? "positive" : "negative"}`}>
                        {t.pnl >= 0 ? "+" : ""}₹{t.pnl?.toFixed(2)}
                      </td>
                      <td className={`font-mono ${t.pnl_pct >= 0 ? "positive" : "negative"}`}>
                        {t.pnl_pct >= 0 ? "+" : ""}{t.pnl_pct?.toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}

      {/* Empty state */}
      {!results && !loading && (
        <div className="card shadow-sm text-center py-16">
          <div className="text-4xl mb-3 opacity-30">📊</div>
          <p className="text-[var(--color-text-secondary)] text-sm font-medium">Write a Pine Script strategy and click &quot;Run Backtest&quot;</p>
          <p className="text-[var(--color-text-muted)] text-xs mt-1">Supports SMA, EMA, RSI, MACD, ATR, Bollinger Bands, crossovers, and more</p>
        </div>
      )}
    </div>
  );
}

"use client";

import { useState, useEffect, useCallback } from "react";
import { useAuth } from "@/lib/auth";
import { marketApi } from "@/lib/api";
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine,
  Line, BarChart, Bar, ComposedChart, Legend
} from "recharts";

interface ChainRow {
  strike: number;
  call: { price: number; iv: number; delta: number; oi: number };
  put: { price: number; iv: number; delta: number; oi: number };
}

interface Leg {
  type: "call" | "put";
  strike: number;
  price: number;
  iv: number;
  qty: number; // positive = buy, negative = sell
}

const READY_MADE = [
  { id: "buy_call", name: "Buy Call", cat: "Bullish", build: (atm: number, chain: ChainRow[]) => {
    const row = chain.find(r => r.strike === atm) || chain[Math.floor(chain.length/2)];
    return [{ type: "call" as const, strike: row.strike, price: row.call.price, iv: row.call.iv, qty: 1 }];
  }},
  { id: "sell_put", name: "Sell Put", cat: "Bullish", build: (atm: number, chain: ChainRow[]) => {
    const row = chain.find(r => r.strike === atm) || chain[Math.floor(chain.length/2)];
    return [{ type: "put" as const, strike: row.strike, price: row.put.price, iv: row.put.iv, qty: -1 }];
  }},
  { id: "bull_call_spread", name: "Bull Call Spread", cat: "Bullish", build: (atm: number, chain: ChainRow[]) => {
    const idx = chain.findIndex(r => r.strike === atm);
    const lo = chain[Math.max(idx, 0)];
    const hi = chain[Math.min(idx + 2, chain.length - 1)];
    return [
      { type: "call" as const, strike: lo.strike, price: lo.call.price, iv: lo.call.iv, qty: 1 },
      { type: "call" as const, strike: hi.strike, price: hi.call.price, iv: hi.call.iv, qty: -1 },
    ];
  }},
  { id: "bear_put_spread", name: "Bear Put Spread", cat: "Bearish", build: (atm: number, chain: ChainRow[]) => {
    const idx = chain.findIndex(r => r.strike === atm);
    const hi = chain[Math.max(idx, 0)];
    const lo = chain[Math.max(idx - 2, 0)];
    return [
      { type: "put" as const, strike: hi.strike, price: hi.put.price, iv: hi.put.iv, qty: 1 },
      { type: "put" as const, strike: lo.strike, price: lo.put.price, iv: lo.put.iv, qty: -1 },
    ];
  }},
  { id: "straddle", name: "Long Straddle", cat: "Neutral", build: (atm: number, chain: ChainRow[]) => {
    const row = chain.find(r => r.strike === atm) || chain[Math.floor(chain.length/2)];
    return [
      { type: "call" as const, strike: row.strike, price: row.call.price, iv: row.call.iv, qty: 1 },
      { type: "put" as const, strike: row.strike, price: row.put.price, iv: row.put.iv, qty: 1 },
    ];
  }},
  { id: "strangle", name: "Long Strangle", cat: "Neutral", build: (atm: number, chain: ChainRow[]) => {
    const idx = chain.findIndex(r => r.strike === atm);
    const hiC = chain[Math.min(idx + 2, chain.length - 1)];
    const loP = chain[Math.max(idx - 2, 0)];
    return [
      { type: "call" as const, strike: hiC.strike, price: hiC.call.price, iv: hiC.call.iv, qty: 1 },
      { type: "put" as const, strike: loP.strike, price: loP.put.price, iv: loP.put.iv, qty: 1 },
    ];
  }},
  { id: "iron_condor", name: "Iron Condor", cat: "Neutral", build: (atm: number, chain: ChainRow[]) => {
    const idx = chain.findIndex(r => r.strike === atm);
    const p1 = chain[Math.max(idx - 4, 0)];
    const p2 = chain[Math.max(idx - 2, 0)];
    const c1 = chain[Math.min(idx + 2, chain.length - 1)];
    const c2 = chain[Math.min(idx + 4, chain.length - 1)];
    return [
      { type: "put" as const, strike: p1.strike, price: p1.put.price, iv: p1.put.iv, qty: 1 },
      { type: "put" as const, strike: p2.strike, price: p2.put.price, iv: p2.put.iv, qty: -1 },
      { type: "call" as const, strike: c1.strike, price: c1.call.price, iv: c1.call.iv, qty: -1 },
      { type: "call" as const, strike: c2.strike, price: c2.call.price, iv: c2.call.iv, qty: 1 },
    ];
  }},
  { id: "butterfly", name: "Butterfly", cat: "Neutral", build: (atm: number, chain: ChainRow[]) => {
    const idx = chain.findIndex(r => r.strike === atm);
    const lo = chain[Math.max(idx - 2, 0)];
    const mid = chain[idx] || chain[Math.floor(chain.length/2)];
    const hi = chain[Math.min(idx + 2, chain.length - 1)];
    return [
      { type: "call" as const, strike: lo.strike, price: lo.call.price, iv: lo.call.iv, qty: 1 },
      { type: "call" as const, strike: mid.strike, price: mid.call.price, iv: mid.call.iv, qty: -2 },
      { type: "call" as const, strike: hi.strike, price: hi.call.price, iv: hi.call.iv, qty: 1 },
    ];
  }},
];

// Client-side Black-Scholes for target-date pricing
function cdf(x: number): number {
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = x < 0 ? -1 : 1;
  const t = 1 / (1 + p * Math.abs(x));
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x / 2);
  return 0.5 * (1 + sign * y);
}
function bsPrice(S: number, K: number, T: number, r: number, sigma: number, type: "call" | "put"): number {
  if (T <= 0) return type === "call" ? Math.max(S - K, 0) : Math.max(K - S, 0);
  const d1 = (Math.log(S / K) + (r + sigma * sigma / 2) * T) / (sigma * Math.sqrt(T));
  const d2 = d1 - sigma * Math.sqrt(T);
  if (type === "call") return S * cdf(d1) - K * Math.exp(-r * T) * cdf(d2);
  return K * Math.exp(-r * T) * cdf(-d2) - S * cdf(-d1);
}

function computePayoff(legs: Leg[], spot: number, lotSize: number, targetT?: number, r = 0.069, nPoints = 100) {
  if (legs.length === 0) return { spots: [], pnl: [], pnlTarget: [] as number[], maxProfit: 0, maxLoss: 0, breakevens: [] as number[] };
  const strikes = legs.map(l => l.strike);
  const minK = Math.min(...strikes);
  const maxK = Math.max(...strikes);
  const range = Math.max(maxK - minK, spot * 0.1);
  const lo = Math.max(minK - range * 1.5, 0);
  const hi = maxK + range * 1.5;

  const premium = legs.reduce((sum, l) => sum + l.price * l.qty * lotSize, 0);
  const spots: number[] = [];
  const pnl: number[] = [];
  const pnlTarget: number[] = [];

  for (let i = 0; i < nPoints; i++) {
    const s = lo + (hi - lo) * i / (nPoints - 1);
    spots.push(Math.round(s * 100) / 100);
    // At expiry (intrinsic)
    let payoffExpiry = 0;
    let payoffTarget = 0;
    for (const leg of legs) {
      const intrinsic = leg.type === "call" ? Math.max(s - leg.strike, 0) : Math.max(leg.strike - s, 0);
      payoffExpiry += intrinsic * leg.qty * lotSize;
      // Target date pricing (BS)
      if (targetT !== undefined && targetT > 0) {
        const sigma = leg.iv / 100;
        const bsVal = bsPrice(s, leg.strike, targetT, r, sigma, leg.type);
        payoffTarget += bsVal * leg.qty * lotSize;
      }
    }
    pnl.push(Math.round((payoffExpiry - premium) * 100) / 100);
    if (targetT !== undefined && targetT > 0) {
      pnlTarget.push(Math.round((payoffTarget - premium) * 100) / 100);
    }
  }

  const maxProfit = Math.max(...pnl);
  const maxLoss = Math.min(...pnl);
  const breakevens: number[] = [];
  for (let i = 0; i < pnl.length - 1; i++) {
    if (pnl[i] * pnl[i + 1] < 0) {
      const x = spots[i] + (spots[i + 1] - spots[i]) * Math.abs(pnl[i]) / (Math.abs(pnl[i]) + Math.abs(pnl[i + 1]));
      breakevens.push(Math.round(x * 10) / 10);
    }
  }
  return { spots, pnl, pnlTarget, maxProfit, maxLoss, breakevens };
}

export default function StrategyBuilder() {
  const { accessToken } = useAuth();
  const [symbol, setSymbol] = useState("NIFTY");
  const [spot, setSpot] = useState(0);
  const [lotSize, setLotSize] = useState(25);
  const [chain, setChain] = useState<ChainRow[]>([]);
  const [expiries, setExpiries] = useState<string[]>([]);
  const [selExpiry, setSelExpiry] = useState("");
  const [atm, setAtm] = useState(0);
  const [legs, setLegs] = useState<Leg[]>([]);
  const [stratId, setStratId] = useState("straddle");
  const [loading, setLoading] = useState(false);
  const [lots, setLots] = useState(1);
  const [sidebarTab, setSidebarTab] = useState<"ready" | "chain">("ready");
  const [customMode, setCustomMode] = useState(false);
  const [chartTab, setChartTab] = useState<"payoff" | "oi">("payoff");
  const [targetDays, setTargetDays] = useState(0); // 0 = today

  // Fetch chain
  const fetchChain = useCallback(async (exp?: string) => {
    if (!accessToken) return;
    setLoading(true);
    try {
      const q = await marketApi.quote(symbol, accessToken) as any;
      setSpot(q.last_price);
      setLotSize(q.lot_size || 25);

      const url = exp ? `/market/option-chain/${symbol}?expiry=${exp}` : `/market/option-chain/${symbol}`;
      const res = await fetch(`${process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"}${url}`, {
        headers: { Authorization: `Bearer ${accessToken}` },
      });
      const data = await res.json();
      setChain(data.chain || []);
      setExpiries(data.expiries_available || []);
      if (!exp && data.expiry) setSelExpiry(data.expiry);
      const atmK = data.chain?.length > 0
        ? data.chain.reduce((prev: ChainRow, curr: ChainRow) => Math.abs(curr.strike - q.last_price) < Math.abs(prev.strike - q.last_price) ? curr : prev).strike
        : 0;
      setAtm(atmK);
    } catch (e) { console.error(e); }
    finally { setLoading(false); }
  }, [accessToken, symbol]);

  useEffect(() => { fetchChain(); }, [fetchChain]);

  // Auto-build legs when strategy or chain changes (only in ready-made mode)
  useEffect(() => {
    if (customMode || chain.length === 0 || atm === 0) return;
    const tmpl = READY_MADE.find(s => s.id === stratId);
    if (tmpl) setLegs(tmpl.build(atm, chain));
  }, [stratId, chain, atm, customMode]);

  const addLeg = (row: ChainRow, type: "call" | "put") => {
    setCustomMode(true);
    const price = type === "call" ? row.call.price : row.put.price;
    const iv = type === "call" ? row.call.iv : row.put.iv;
    setLegs(prev => [...prev, { type, strike: row.strike, price, iv, qty: 1 }]);
  };
  const removeLeg = (idx: number) => setLegs(prev => prev.filter((_, i) => i !== idx));
  const toggleLegSide = (idx: number) => setLegs(prev => prev.map((l, i) => i === idx ? { ...l, qty: -l.qty } : l));
  const clearLegs = () => { setLegs([]); setCustomMode(false); };

  const handleExpiryChange = (exp: string) => {
    setSelExpiry(exp);
    fetchChain(exp);
  };

  // Compute days to expiry and target T
  const daysToExpiry = selExpiry ? Math.max(Math.round((new Date(selExpiry).getTime() - Date.now()) / 86400000), 1) : 7;
  const targetT = targetDays > 0 ? (daysToExpiry - targetDays) / 365 : undefined;

  // Compute payoff
  const payoff = computePayoff(legs, spot, lotSize * lots, targetT);
  const chartData = payoff.spots.map((s, i) => ({ spot: s, pnl: payoff.pnl[i], pnlTarget: payoff.pnlTarget[i] ?? undefined }));
  const premium = legs.reduce((sum, l) => sum + l.price * l.qty * lotSize * lots, 0);

  // OI data for bar chart
  const oiData = chain.map(row => ({ strike: row.strike, callOI: row.call.oi, putOI: row.put.oi }));
  const pcr = chain.length > 0 ? (chain.reduce((s, r) => s + r.put.oi, 0) / Math.max(chain.reduce((s, r) => s + r.call.oi, 0), 1)).toFixed(2) : "—";

  // Gradient offset for green/red split
  const off = payoff.maxProfit <= 0 ? 0 : payoff.maxLoss >= 0 ? 1 : payoff.maxProfit / (payoff.maxProfit - payoff.maxLoss);

  return (
    <div className="flex flex-col lg:flex-row gap-4">
      {/* LEFT SIDEBAR */}
      <div className="w-full lg:w-[400px] flex-shrink-0 flex flex-col gap-3">

        {/* Symbol + Spot */}
        <div className="card shadow-sm">
          <div className="flex items-center gap-3 mb-3">
            <select
              className="input-field text-sm font-bold flex-1"
              value={symbol}
              onChange={e => { setSymbol(e.target.value); }}
            >
              <option value="NIFTY">NIFTY</option>
              <option value="BANKNIFTY">BANKNIFTY</option>
              <option value="RELIANCE">RELIANCE</option>
            </select>
            <div className="text-right">
              <div className="font-mono font-bold text-lg text-[var(--color-text-primary)]">₹{spot.toLocaleString()}</div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[11px] text-[var(--color-text-muted)] uppercase font-semibold">Expiry</span>
            <select className="input-field text-xs flex-1" value={selExpiry} onChange={e => handleExpiryChange(e.target.value)}>
              {expiries.map(e => <option key={e} value={e}>{e}</option>)}
            </select>
            <span className="text-[11px] text-[var(--color-text-muted)] uppercase font-semibold ml-2">Lots</span>
            <div className="flex items-center gap-1">
              <button className="px-2 py-1 rounded border border-[var(--color-border-subtle)] text-xs" onClick={() => setLots(Math.max(1, lots - 1))}>−</button>
              <span className="font-mono text-sm w-6 text-center">{lots}</span>
              <button className="px-2 py-1 rounded border border-[var(--color-border-subtle)] text-xs" onClick={() => setLots(lots + 1)}>+</button>
            </div>
          </div>
        </div>

        {/* Sidebar Tabs */}
        <div className="card shadow-sm">
          <div className="flex gap-1 mb-3 border-b border-[var(--color-border-subtle)]">
            <button onClick={() => setSidebarTab("ready")} className={`px-3 py-1.5 text-xs font-bold ${sidebarTab === "ready" ? "border-b-2 border-[var(--color-accent-blue)] text-[var(--color-text-primary)]" : "text-[var(--color-text-muted)]"}`}>Ready-made</button>
            <button onClick={() => setSidebarTab("chain")} className={`px-3 py-1.5 text-xs font-bold ${sidebarTab === "chain" ? "border-b-2 border-[var(--color-accent-blue)] text-[var(--color-text-primary)]" : "text-[var(--color-text-muted)]"}`}>Option Chain</button>
          </div>

          {sidebarTab === "ready" ? (
            <div className="grid grid-cols-2 gap-2">
              {READY_MADE.map(s => (
                <button key={s.id} onClick={() => { setCustomMode(false); setStratId(s.id); }}
                  className={`border rounded p-2.5 text-xs font-semibold text-center transition-colors ${
                    stratId === s.id && !customMode
                      ? "border-[var(--color-accent-blue)] bg-[rgba(41,98,255,0.06)] text-[var(--color-accent-blue)]"
                      : "border-[var(--color-border-subtle)] text-[var(--color-text-secondary)] hover:border-[var(--color-border)]"
                  }`}
                >{s.name}</button>
              ))}
            </div>
          ) : (
            <div className="max-h-[300px] overflow-y-auto">
              <table className="data-table text-[10px] w-full">
                <thead><tr><th>Strike</th><th className="text-center">CE ₹</th><th className="text-center">PE ₹</th></tr></thead>
                <tbody>
                  {chain.map(row => (
                    <tr key={row.strike} className={row.strike === atm ? "bg-[rgba(41,98,255,0.05)]" : ""}>
                      <td className="font-mono font-semibold">{row.strike}</td>
                      <td className="text-center">
                        <button onClick={() => addLeg(row, "call")} className="font-mono hover:text-[var(--color-accent-green)] hover:font-bold transition-colors px-1">{row.call.price.toFixed(1)}</button>
                      </td>
                      <td className="text-center">
                        <button onClick={() => addLeg(row, "put")} className="font-mono hover:text-[var(--color-accent-red)] hover:font-bold transition-colors px-1">{row.put.price.toFixed(1)}</button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>

        {/* Legs Table */}
        <div className="card shadow-sm">
          <div className="flex justify-between items-center mb-2">
            <h3 className="text-xs font-bold uppercase text-[var(--color-text-secondary)] tracking-wider">
              {legs.length} leg{legs.length !== 1 ? "s" : ""} {customMode ? "(Custom)" : `— ${READY_MADE.find(s => s.id === stratId)?.name || ""}`}
            </h3>
            {legs.length > 0 && <button onClick={clearLegs} className="text-[10px] text-[var(--color-accent-red)] hover:underline">Clear All</button>}
          </div>
          <div className="text-[10px] text-[var(--color-text-muted)] mb-1 grid grid-cols-7 gap-1 font-semibold uppercase">
            <span>B/S</span><span>Expiry</span><span>Strike</span><span>Type</span><span>Lots</span><span className="text-right">Price</span><span></span>
          </div>
          {legs.map((leg, i) => (
            <div key={i} className="grid grid-cols-7 gap-1 items-center py-1.5 border-t border-[var(--color-border-subtle)] text-xs">
              <span>
                <button onClick={() => toggleLegSide(i)} className={`px-1.5 py-0.5 rounded text-[10px] font-bold cursor-pointer ${leg.qty > 0 ? "bg-[rgba(8,153,129,0.1)] text-[var(--color-accent-green)]" : "bg-[rgba(242,54,69,0.1)] text-[var(--color-accent-red)]"}`}>
                  {leg.qty > 0 ? "B" : "S"}
                </button>
              </span>
              <span className="text-[var(--color-text-muted)] font-mono">{selExpiry.slice(5)}</span>
              <span className="font-mono font-semibold">{leg.strike}</span>
              <span className="uppercase">{leg.type === "call" ? "CE" : "PE"}</span>
              <span className="font-mono">{Math.abs(leg.qty) * lots}</span>
              <span className="text-right font-mono font-semibold">{leg.price.toFixed(2)}</span>
              <span className="text-right"><button onClick={() => removeLeg(i)} className="text-[var(--color-text-muted)] hover:text-[var(--color-accent-red)] text-xs">✕</button></span>
            </div>
          ))}
          {legs.length === 0 && <div className="py-4 text-center text-xs text-[var(--color-text-muted)]">Click a strategy or add legs from the chain</div>}
          <div className="border-t border-[var(--color-border-subtle)] pt-2 mt-2 flex justify-between text-xs">
            <span className="text-[var(--color-text-muted)]">Net Premium</span>
            <span className={`font-mono font-bold ${premium > 0 ? "text-[var(--color-accent-red)]" : "text-[var(--color-accent-green)]"}`}>
              {premium > 0 ? `Pay ₹${premium.toFixed(2)}` : `Receive ₹${Math.abs(premium).toFixed(2)}`}
            </span>
          </div>
        </div>

        {/* Important Info */}
        <div className="bg-[var(--color-bg-elevated)] border border-[var(--color-border-subtle)] rounded p-3">
          <h4 className="text-[11px] font-bold mb-1 text-[var(--color-text-primary)]">Important info</h4>
          <p className="text-[10px] text-[var(--color-text-secondary)] leading-relaxed">
            The profit and loss are projections, and they depend on premia, liquidity, IV, etc. While we make the best effort to ensure they are right, the actual numbers may vary.
          </p>
        </div>
      </div>

      {/* RIGHT MAIN AREA */}
      <div className="flex-1 flex flex-col gap-3 min-w-0">

        {/* Metrics Row */}
        <div className="card grid grid-cols-2 lg:grid-cols-5 gap-3 shadow-sm">
          <div>
            <div className="text-[10px] font-semibold text-[var(--color-text-secondary)] uppercase tracking-wide mb-1">Max Profit</div>
            <div className="font-mono font-bold text-sm" style={{ color: "var(--color-accent-green)" }}>
              {payoff.maxProfit > 1e8 ? "Unlimited" : `+₹${payoff.maxProfit.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
            </div>
          </div>
          <div className="border-l border-[var(--color-border-subtle)] pl-3">
            <div className="text-[10px] font-semibold text-[var(--color-text-secondary)] uppercase tracking-wide mb-1">Max Loss</div>
            <div className="font-mono font-bold text-sm" style={{ color: "var(--color-accent-red)" }}>
              {payoff.maxLoss < -1e8 ? "Unlimited" : `-₹${Math.abs(payoff.maxLoss).toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
            </div>
          </div>
          <div className="border-l border-[var(--color-border-subtle)] pl-3">
            <div className="text-[10px] font-semibold text-[var(--color-text-secondary)] uppercase tracking-wide mb-1">Breakeven</div>
            <div className="font-mono font-bold text-xs text-[var(--color-text-primary)]">
              {payoff.breakevens.length > 0 ? payoff.breakevens.join(" / ") : "—"}
            </div>
          </div>
          <div className="border-l border-[var(--color-border-subtle)] pl-3">
            <div className="text-[10px] font-semibold text-[var(--color-text-secondary)] uppercase tracking-wide mb-1">Reward / Risk</div>
            <div className="font-mono font-bold text-sm text-[var(--color-text-primary)]">
              {payoff.maxLoss === 0 || payoff.maxProfit > 1e8 ? "N/A" : `1 : ${Math.abs(payoff.maxProfit / payoff.maxLoss).toFixed(2)}`}
            </div>
          </div>
          <div className="border-l border-[var(--color-border-subtle)] pl-3">
            <div className="text-[10px] font-semibold text-[var(--color-text-secondary)] uppercase tracking-wide mb-1">Lot Size</div>
            <div className="font-mono font-bold text-sm text-[var(--color-text-primary)]">{lotSize}</div>
          </div>
        </div>

        {/* Chart Area */}
        <div className="card flex-1 min-h-[380px] flex flex-col shadow-sm">
          <div className="flex gap-4 border-b border-[var(--color-border-subtle)] mb-4">
            <button onClick={() => setChartTab("payoff")} className={`px-1 py-2 text-sm font-bold ${chartTab === "payoff" ? "border-b-2 border-[var(--color-accent-blue)] text-[var(--color-text-primary)]" : "text-[var(--color-text-muted)]"}`}>Payoff Graph</button>
            <button onClick={() => setChartTab("oi")} className={`px-1 py-2 text-sm font-bold ${chartTab === "oi" ? "border-b-2 border-[var(--color-accent-blue)] text-[var(--color-text-primary)]" : "text-[var(--color-text-muted)]"}`}>Open Interest</button>
            <div className="ml-auto flex items-center gap-2 text-[10px] text-[var(--color-text-muted)]">
              <span>PCR: <strong className="text-[var(--color-text-primary)]">{pcr}</strong></span>
            </div>
          </div>

          {chartTab === "payoff" ? (
            <>
              <div className="flex-1 w-full relative">
                {chartData.length > 0 && !loading ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={chartData} margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                      <defs>
                        <linearGradient id="splitColor" x1="0" y1="0" x2="0" y2="1">
                          <stop offset={off} stopColor="var(--color-accent-green)" stopOpacity={0.2} />
                          <stop offset={off} stopColor="var(--color-accent-red)" stopOpacity={0.2} />
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" vertical={false} />
                      <XAxis dataKey="spot" stroke="var(--color-text-muted)" fontSize={10} tickFormatter={v => `₹${Number(v).toLocaleString()}`} dy={8} />
                      <YAxis stroke="var(--color-text-muted)" fontSize={10} tickFormatter={v => `₹${Number(v).toLocaleString()}`} dx={-4} />
                      <Tooltip
                        contentStyle={{ background: "var(--color-bg-card)", borderColor: "var(--color-border-subtle)", borderRadius: 6, fontSize: "11px" }}
                        labelFormatter={v => `Spot: ₹${v}`}
                        formatter={(v: any, name: any) => [`${Number(v) >= 0 ? "+" : ""}₹${Number(v).toLocaleString()}`, name === "pnl" ? "On Expiry" : "On Target"]}
                      />
                      <ReferenceLine y={0} stroke="var(--color-text-muted)" strokeDasharray="3 3" />
                      <ReferenceLine x={spot} stroke="var(--color-accent-blue)" strokeDasharray="3 3" label={{ value: "Spot", fill: "var(--color-accent-blue)", fontSize: 10 }} />
                      <Area type="monotone" dataKey="pnl" stroke="var(--color-text-primary)" strokeWidth={2} fill="url(#splitColor)" isAnimationActive={false} name="pnl" />
                      {targetDays > 0 && <Line type="monotone" dataKey="pnlTarget" stroke="var(--color-accent-blue)" strokeWidth={2} strokeDasharray="6 3" dot={false} isAnimationActive={false} name="pnlTarget" />}
                    </ComposedChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="absolute inset-0 flex items-center justify-center text-sm text-[var(--color-text-muted)]">
                    {loading ? "Loading option chain..." : "Select a strategy"}
                  </div>
                )}
              </div>
              {/* Target Date Slider */}
              <div className="flex items-center gap-3 mt-3 pt-3 border-t border-[var(--color-border-subtle)]">
                <span className="text-[10px] font-semibold text-[var(--color-text-muted)] uppercase whitespace-nowrap">Target Date</span>
                <input type="range" min={0} max={daysToExpiry} value={targetDays} onChange={e => setTargetDays(+e.target.value)} className="flex-1 accent-[var(--color-accent-blue)]" />
                <span className="text-[11px] font-mono text-[var(--color-text-primary)] whitespace-nowrap w-24 text-right">
                  {targetDays === 0 ? "Today" : `+${targetDays}d`} → Exp {daysToExpiry}d
                </span>
              </div>
            </>
          ) : (
            /* OI Bar Chart */
            <div className="flex-1 w-full">
              {oiData.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={oiData} margin={{ top: 10, right: 20, left: 10, bottom: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" vertical={false} />
                    <XAxis dataKey="strike" stroke="var(--color-text-muted)" fontSize={9} />
                    <YAxis stroke="var(--color-text-muted)" fontSize={9} tickFormatter={v => `${(Number(v)/1000).toFixed(0)}K`} />
                    <Tooltip contentStyle={{ background: "var(--color-bg-card)", borderColor: "var(--color-border-subtle)", borderRadius: 6, fontSize: "11px" }} />
                    <Legend wrapperStyle={{ fontSize: "11px" }} />
                    <Bar dataKey="callOI" name="Call OI" fill="var(--color-accent-green)" fillOpacity={0.6} />
                    <Bar dataKey="putOI" name="Put OI" fill="var(--color-accent-red)" fillOpacity={0.6} />
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <div className="flex items-center justify-center h-full text-sm text-[var(--color-text-muted)]">No OI data</div>
              )}
            </div>
          )}
        </div>

        {/* Strikewise IVs */}
        {legs.length > 0 && (
          <div className="card shadow-sm">
            <h3 className="text-xs font-bold uppercase text-[var(--color-text-secondary)] tracking-wider mb-3">Strikewise IVs &amp; Greeks</h3>
            <table className="data-table text-xs">
              <thead>
                <tr>
                  <th>Strike</th><th>Expiry</th><th>Type</th><th>IV</th><th>Price</th><th>B/S</th>
                </tr>
              </thead>
              <tbody>
                {legs.map((leg, i) => (
                  <tr key={i}>
                    <td className="font-mono font-semibold">{leg.strike}</td>
                    <td className="text-[var(--color-text-muted)]">{selExpiry}</td>
                    <td className="uppercase">{leg.type === "call" ? "CE" : "PE"}</td>
                    <td className="font-mono">{leg.iv.toFixed(1)}%</td>
                    <td className="font-mono font-semibold">{leg.price.toFixed(2)}</td>
                    <td>
                      <span className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${leg.qty > 0 ? "bg-[rgba(8,153,129,0.1)] text-[var(--color-accent-green)]" : "bg-[rgba(242,54,69,0.1)] text-[var(--color-accent-red)]"}`}>
                        {leg.qty > 0 ? "BUY" : "SELL"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

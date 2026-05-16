"use client";

import { useEffect, useState } from "react";
import { useAuth } from "@/lib/auth";
import { healthApi, portfoliosApi, tradesApi } from "@/lib/api";

export default function DashboardOverview() {
  const { accessToken, user } = useAuth();
  const [health, setHealth] = useState<Record<string, unknown> | null>(null);
  const [portfolioCount, setPortfolioCount] = useState(0);
  const [tradeCount, setTradeCount] = useState(0);
  const [positions, setPositions] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    if (!accessToken) return;
    healthApi.check().then(setHealth).catch(() => {});
    portfoliosApi.list(accessToken).then(d => setPortfolioCount(d.count)).catch(() => {});
    tradesApi.list(accessToken).then(d => setTradeCount(d.count)).catch(() => {});
    tradesApi.positions(accessToken).then(setPositions).catch(() => {});
  }, [accessToken]);

  const metrics = [
    { label: "API Status", value: health ? "Online" : "...", color: "var(--color-accent-green)" },
    { label: "Version", value: (health?.version as string) || "...", color: "var(--color-accent-blue)" },
    { label: "Portfolios", value: String(portfolioCount), color: "var(--color-accent-purple)" },
    { label: "Open Trades", value: String(tradeCount), color: "var(--color-accent-amber)" },
    { label: "Positions", value: String((positions as Record<string, unknown>)?.n_positions || 0), color: "var(--color-accent-blue)" },
    { label: "Total Notional", value: positions ? `₹${Number((positions as Record<string, unknown>).total_notional || 0).toLocaleString()}` : "...", color: "var(--color-accent-green)" },
  ];

  const capabilities = [
    { title: "Pricing Lab", desc: "Black-Scholes, Monte Carlo (3 methods), Binomial Tree (European/American)", icon: "💹" },
    { title: "Greeks Explorer", desc: "Delta, Gamma, Vega, Theta, Rho — analytical & numerical", icon: "Δ" },
    { title: "Risk Engine", desc: "Portfolio VaR/CVaR via Historical, Parametric, Monte Carlo methods", icon: "🛡" },
    { title: "Strategy Simulator", desc: "8 strategies: Straddle, Strangle, Spreads, Iron Condor, Butterfly", icon: "♟" },
    { title: "Scenario Analysis", desc: "Multi-dimension stress testing with 2D P&L heatmaps", icon: "🔥" },
    { title: "Market Data", desc: "NIFTY/BANKNIFTY option chains with IV smile, OI, and Greeks", icon: "📈" },
    { title: "Trade Capture", desc: "Book trades, track positions, compute P&L in real-time", icon: "📋" },
    { title: "Portfolio Persistence", desc: "Save/load portfolios to database, run risk on saved positions", icon: "💼" },
  ];

  return (
    <div>
      <div className="mb-8">
        <h1 className="text-2xl font-bold" style={{ color: "var(--color-text-primary)" }}>
          Welcome, {user?.full_name?.split(" ")[0]}
        </h1>
        <p className="text-sm mt-1" style={{ color: "var(--color-text-secondary)" }}>
          Quant Engine Platform — Institutional-grade derivatives analytics
        </p>
      </div>

      {/* KPI Metrics */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4 mb-8">
        {metrics.map((m) => (
          <div key={m.label} className="card text-center py-5">
            <div className="metric-value" style={{ color: m.color, fontFamily: "var(--font-mono)" }}>
              {m.value}
            </div>
            <div className="metric-label mt-2">{m.label}</div>
          </div>
        ))}
      </div>

      {/* Capabilities Grid */}
      <h2 className="text-lg font-semibold mb-4" style={{ color: "var(--color-text-primary)" }}>
        Platform Capabilities
      </h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {capabilities.map((cap) => (
          <div key={cap.title} className="card hover:border-[var(--color-accent-blue)] transition-all duration-200 cursor-default">
            <div className="text-2xl mb-3">{cap.icon}</div>
            <h3 className="font-semibold text-sm mb-1" style={{ color: "var(--color-text-primary)" }}>
              {cap.title}
            </h3>
            <p className="text-xs leading-relaxed" style={{ color: "var(--color-text-secondary)" }}>
              {cap.desc}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}

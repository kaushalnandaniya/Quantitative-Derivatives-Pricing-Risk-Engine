"use client";

import { useState } from "react";
import { useAuth } from "@/lib/auth";
import { regulatoryApi, portfoliosApi, PortfolioData } from "@/lib/api";
import { useEffect } from "react";

export default function RegulatoryPage() {
  const { accessToken, user } = useAuth();
  const [portfolios, setPortfolios] = useState<PortfolioData[]>([]);
  const [selectedPortfolio, setSelectedPortfolio] = useState<string>("");
  const [report, setReport] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!accessToken) return;
    portfoliosApi.list(accessToken).then((res: any) => {
      setPortfolios(res.portfolios);
      if (res.portfolios.length > 0) setSelectedPortfolio(res.portfolios[0].id);
    });
  }, [accessToken]);

  const generateReport = async () => {
    if (!accessToken || !selectedPortfolio) return;
    try {
      setLoading(true);
      setError("");
      const port = portfolios.find(p => p.id === selectedPortfolio);
      if (!port) return;

      const res = await regulatoryApi.fullReport({ portfolio: port.positions }, accessToken);
      setReport(res);
    } catch (err: any) {
      setError(err.message || "Failed to generate report");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold" style={{ color: "var(--color-text-primary)" }}>Basel III/IV Regulatory</h1>
        <p className="text-sm mt-1" style={{ color: "var(--color-text-secondary)" }}>
          Regulatory risk metrics, capital charges, and concentration analysis.
        </p>
      </div>

      <div className="card p-5 flex items-end gap-4">
        <div className="flex-1">
          <label className="block text-xs font-medium mb-1 text-gray-400">Select Portfolio</label>
          <select 
            className="input-field"
            value={selectedPortfolio}
            onChange={(e) => setSelectedPortfolio(e.target.value)}
          >
            <option value="">-- Choose Portfolio --</option>
            {portfolios.map(p => (
              <option key={p.id} value={p.id}>{p.name} ({p.positions.length} pos)</option>
            ))}
          </select>
        </div>
        <button 
          onClick={generateReport}
          disabled={loading || !selectedPortfolio}
          className="btn-primary"
        >
          {loading ? "Computing..." : "Generate Report"}
        </button>
      </div>

      {error && <div className="p-4 bg-red-500/10 text-red-500 border border-red-500/30 rounded">{error}</div>}

      {report && (
        <div className="space-y-6 animate-in fade-in">
          {/* Summary Cards */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="card p-4">
              <div className="text-xs text-gray-400">Regulatory VaR (10d, 99%)</div>
              <div className="text-xl font-mono text-red-400 mt-1">
                ${report.summary.var_10d_99.toLocaleString(undefined, {minimumFractionDigits: 2})}
              </div>
            </div>
            <div className="card p-4">
              <div className="text-xs text-gray-400">Stressed VaR</div>
              <div className="text-xl font-mono text-red-500 mt-1">
                ${report.summary.stressed_var_10d.toLocaleString(undefined, {minimumFractionDigits: 2})}
              </div>
            </div>
            <div className="card p-4">
              <div className="text-xs text-gray-400">Capital Charge</div>
              <div className="text-xl font-mono text-amber-400 mt-1">
                ${report.summary.total_capital_charge.toLocaleString(undefined, {minimumFractionDigits: 2})}
              </div>
            </div>
            <div className="card p-4">
              <div className="text-xs text-gray-400">Leverage Check</div>
              <div className={`text-xl font-bold mt-1 ${report.summary.leverage_adequate ? 'text-green-500' : 'text-red-500'}`}>
                {report.summary.leverage_adequate ? "PASS" : "FAIL"}
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="card p-5">
              <h3 className="font-bold text-white border-b border-white/10 pb-2 mb-3">Concentration Risk</h3>
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Total Notional</span>
                  <span className="font-mono text-white">${report.concentration_risk.total_notional.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Breaches ({'>'}25%)</span>
                  <span className={`font-mono font-bold ${report.concentration_risk.breach_count > 0 ? 'text-red-500' : 'text-green-500'}`}>
                    {report.concentration_risk.breach_count}
                  </span>
                </div>
                
                <div className="mt-4 pt-4 border-t border-white/10">
                  <div className="text-xs font-semibold text-gray-500 uppercase mb-2">Top Exposures by Strike</div>
                  {report.concentration_risk.concentrations.slice(0, 3).map((c: any) => (
                    <div key={c.strike} className="flex items-center justify-between text-sm mb-1">
                      <span className="text-gray-300">Strike {c.strike}</span>
                      <div className="flex items-center gap-2">
                        <span className={`font-mono ${c.breach ? 'text-red-400' : 'text-gray-400'}`}>{c.percentage}%</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="card p-5">
              <h3 className="font-bold text-white border-b border-white/10 pb-2 mb-3">Capital Requirements</h3>
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Base VaR Component</span>
                  <span className="font-mono text-white">${report.capital_charge.var_component.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Stressed VaR Component</span>
                  <span className="font-mono text-white">${report.capital_charge.stressed_var_10d.toLocaleString()}</span>
                </div>
                <div className="flex justify-between text-sm font-bold border-t border-white/10 pt-2 mt-2">
                  <span className="text-gray-200">Total Capital Charge</span>
                  <span className="font-mono text-amber-400">${report.capital_charge.total_capital_charge.toLocaleString()}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

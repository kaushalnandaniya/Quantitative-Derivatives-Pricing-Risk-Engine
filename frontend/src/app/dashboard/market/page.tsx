"use client";

import { useState, useEffect } from "react";
import { useAuth } from "@/lib/auth";
import { marketApi } from "@/lib/api";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip, ResponsiveContainer, BarChart, Bar, ComposedChart
} from "recharts";
import KiteTradeModal from "@/components/KiteTradeModal";

export default function MarketData() {
  const { accessToken } = useAuth();
  
  // State for active symbol
  const [symbol, setSymbol] = useState("NIFTY");
  
  // Market data state
  const [quote, setQuote] = useState<Record<string, unknown> | null>(null);
  const [chain, setChain] = useState<Record<string, unknown> | null>(null);
  const [history, setHistory] = useState<any[]>([]);
  const [loading, setLoading] = useState(false);
  const [historyPeriod, setHistoryPeriod] = useState("1mo");
  
  // Search state
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<any[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  
  // Kite state
  const [kiteConnected, setKiteConnected] = useState(false);
  const [showKiteConnect, setShowKiteConnect] = useState(false);
  const [showTradeModal, setShowTradeModal] = useState(false);
  
  // Kite connect form state
  const [kiteKey, setKiteKey] = useState("");
  const [kiteSecret, setKiteSecret] = useState("");
  const [kiteRequestToken, setKiteRequestToken] = useState("");
  const [kiteConnecting, setKiteConnecting] = useState(false);

  // Fetch market data
  useEffect(() => {
    if (!accessToken || !symbol) return;
    
    let isMounted = true;
    setLoading(true);
    
    Promise.all([
      marketApi.quote(symbol, accessToken).catch(() => null),
      marketApi.optionChain(symbol, accessToken).catch(() => null),
      marketApi.history(symbol, historyPeriod, accessToken).catch(() => null),
    ]).then(([qRes, cRes, hRes]) => {
      if (!isMounted) return;
      if (qRes) setQuote(qRes);
      if (cRes) setChain(cRes);
      if (hRes && hRes.data) setHistory(hRes.data);
      setLoading(false);
    });
    
    return () => { isMounted = false; };
  }, [symbol, historyPeriod, accessToken]);

  // Check Kite status
  useEffect(() => {
    if (!accessToken) return;
    marketApi.kiteStatus(accessToken)
      .then((res: any) => setKiteConnected(res.connected))
      .catch(() => setKiteConnected(false));
  }, [accessToken]);

  // Handle search
  useEffect(() => {
    if (!searchQuery || searchQuery.length < 2 || !accessToken) {
      setSearchResults([]);
      return;
    }
    
    const timeout = setTimeout(() => {
      setIsSearching(true);
      marketApi.search(searchQuery, accessToken)
        .then((res: any) => setSearchResults(res.results || []))
        .catch(() => setSearchResults([]))
        .finally(() => setIsSearching(false));
    }, 500);
    
    return () => clearTimeout(timeout);
  }, [searchQuery, accessToken]);

  const handleKiteConnect = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!accessToken) return;
    
    setKiteConnecting(true);
    try {
      await marketApi.kiteConnect({
        api_key: kiteKey,
        api_secret: kiteSecret,
        request_token: kiteRequestToken
      }, accessToken);
      setKiteConnected(true);
      setShowKiteConnect(false);
    } catch (err: any) {
      alert(err.message || "Failed to connect to Kite");
    } finally {
      setKiteConnecting(false);
    }
  };

  const handleKiteDisconnect = async () => {
    if (!accessToken) return;
    try {
      await marketApi.kiteDisconnect(accessToken);
      setKiteConnected(false);
    } catch (err: any) {
      alert(err.message || "Failed to disconnect");
    }
  };

  const options = (chain?.chain || []) as Array<Record<string, unknown>>;
  const spotPrice = quote?.last_price as number || 0;
  const quoteChange = Number(quote?.change || 0);

  return (
    <div className="pb-10">
      <div className="flex justify-between items-center mb-6">
        <h1 className="text-2xl font-bold" style={{ color: "var(--color-text-primary)" }}>Market Data & Trading</h1>
        
        {/* Kite Connection Status */}
        <div>
          {kiteConnected ? (
            <div className="flex items-center gap-3">
              <div className="px-3 py-1.5 rounded-full text-xs font-semibold flex items-center gap-2" style={{ background: "rgba(48,209,88,0.1)", color: "var(--color-accent-green)", border: "1px solid rgba(48,209,88,0.2)" }}>
                <div className="w-2 h-2 rounded-full bg-[var(--color-accent-green)]"></div>
                Kite Connected
              </div>
              <button onClick={handleKiteDisconnect} className="text-xs text-[var(--color-text-muted)] hover:text-[var(--color-accent-red)]">Disconnect</button>
            </div>
          ) : (
            <button 
              onClick={() => setShowKiteConnect(!showKiteConnect)}
              className="px-4 py-2 rounded-lg text-sm font-semibold transition-colors"
              style={{ background: "var(--color-accent-blue)", color: "white" }}
            >
              Connect Kite Broker
            </button>
          )}
        </div>
      </div>

      {/* Kite Connect Modal/Dropdown */}
      {showKiteConnect && !kiteConnected && (
        <div className="card mb-6" style={{ background: "rgba(41,98,255,0.05)", border: "1px solid rgba(41,98,255,0.2)" }}>
          <h3 className="text-lg font-bold mb-2" style={{ color: "var(--color-accent-blue)" }}>Connect Zerodha Kite</h3>
          <p className="text-sm mb-4" style={{ color: "var(--color-text-secondary)" }}>Enter your Kite API credentials to enable live trading.</p>
          <form onSubmit={handleKiteConnect} className="grid grid-cols-1 md:grid-cols-4 gap-4 items-end">
            <div>
              <label className="text-xs font-semibold uppercase mb-1 block text-[var(--color-text-muted)]">API Key</label>
              <input type="text" value={kiteKey} onChange={e => setKiteKey(e.target.value)} required className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border-subtle)] text-[var(--color-text-primary)]" />
            </div>
            <div>
              <label className="text-xs font-semibold uppercase mb-1 block text-[var(--color-text-muted)]">API Secret</label>
              <input type="password" value={kiteSecret} onChange={e => setKiteSecret(e.target.value)} required className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border-subtle)] text-[var(--color-text-primary)]" />
            </div>
            <div>
              <label className="text-xs font-semibold uppercase mb-1 block text-[var(--color-text-muted)]">Request Token</label>
              <input type="text" value={kiteRequestToken} onChange={e => setKiteRequestToken(e.target.value)} required className="w-full px-3 py-2 rounded-lg bg-[var(--color-bg-primary)] border border-[var(--color-border-subtle)] text-[var(--color-text-primary)]" />
            </div>
            <div>
              <button type="submit" disabled={kiteConnecting} className="w-full py-2 rounded-lg font-bold text-white bg-[var(--color-accent-blue)] disabled:opacity-50">
                {kiteConnecting ? "Connecting..." : "Connect"}
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Search Bar */}
      <div className="relative mb-6 z-20">
        <div className="flex items-center bg-[var(--color-bg-card)] border border-[var(--color-border-subtle)] rounded-xl px-4 py-3 focus-within:border-[var(--color-accent-blue)] transition-colors">
          <span className="text-[var(--color-text-muted)] mr-3">🔍</span>
          <input 
            type="text" 
            placeholder="Search for NSE/BSE stocks (e.g., INFY, TCS, RELIANCE)..." 
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-transparent border-none outline-none text-[var(--color-text-primary)] text-lg placeholder:text-[var(--color-text-muted)]"
          />
          {isSearching && <div className="w-4 h-4 border-2 border-[var(--color-accent-blue)] border-t-transparent rounded-full animate-spin"></div>}
        </div>
        
        {/* Search Results Dropdown */}
        {searchQuery.length >= 2 && searchResults.length > 0 && (
          <div className="absolute top-full left-0 right-0 mt-2 bg-[var(--color-bg-card)] border border-[var(--color-border-subtle)] rounded-xl shadow-xl overflow-hidden">
            {searchResults.map((res, i) => (
              <button 
                key={i}
                onClick={() => {
                  setSymbol(res.symbol);
                  setSearchQuery("");
                  setSearchResults([]);
                }}
                className="w-full text-left px-4 py-3 hover:bg-[var(--color-bg-secondary)] border-b border-[var(--color-border-subtle)] last:border-none transition-colors"
              >
                <div className="font-bold text-[var(--color-text-primary)]">{res.symbol}</div>
                <div className="text-xs text-[var(--color-text-muted)]">{res.name} • {res.exchange}</div>
              </button>
            ))}
          </div>
        )}
      </div>

      {loading && !quote && (
        <div className="flex justify-center py-10">
          <div className="w-8 h-8 border-2 border-[var(--color-accent-blue)] border-t-transparent rounded-full animate-spin"></div>
        </div>
      )}

      {quote && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          {/* Quote Card */}
          <div className="card lg:col-span-1 flex flex-col justify-between relative overflow-hidden">
            <div className="absolute top-0 right-0 p-4 opacity-10 font-bold text-6xl pointer-events-none select-none">
              {quote.symbol as string}
            </div>
            
            <div>
              <div className="flex justify-between items-start mb-4">
                <div>
                  <h2 className="text-2xl font-bold text-[var(--color-text-primary)]">{quote.symbol as string}</h2>
                  <div className="text-sm text-[var(--color-text-secondary)]">{quote.name as string} • {quote.provider as string}</div>
                </div>
                <div className={`px-2 py-1 rounded text-xs font-bold ${quoteChange >= 0 ? "bg-[rgba(48,209,88,0.1)] text-[var(--color-accent-green)]" : "bg-[rgba(255,69,58,0.1)] text-[var(--color-accent-red)]"}`}>
                  {quoteChange >= 0 ? "+" : ""}{quoteChange.toFixed(2)} ({Number(quote.change_pct).toFixed(2)}%)
                </div>
              </div>

              <div className="text-5xl font-mono font-bold text-[var(--color-text-primary)] mb-6">
                ₹{spotPrice.toLocaleString(undefined, { minimumFractionDigits: 2 })}
              </div>

              <div className="grid grid-cols-2 gap-4 mb-6">
                <div>
                  <div className="text-xs font-semibold text-[var(--color-text-muted)] uppercase mb-1">Open</div>
                  <div className="font-mono text-[var(--color-text-primary)]">{Number(quote.open).toLocaleString()}</div>
                </div>
                <div>
                  <div className="text-xs font-semibold text-[var(--color-text-muted)] uppercase mb-1">Volume</div>
                  <div className="font-mono text-[var(--color-text-primary)]">{Number(quote.volume).toLocaleString()}</div>
                </div>
                <div>
                  <div className="text-xs font-semibold text-[var(--color-text-muted)] uppercase mb-1">High</div>
                  <div className="font-mono text-[var(--color-text-primary)]">{Number(quote.high).toLocaleString()}</div>
                </div>
                <div>
                  <div className="text-xs font-semibold text-[var(--color-text-muted)] uppercase mb-1">Low</div>
                  <div className="font-mono text-[var(--color-text-primary)]">{Number(quote.low).toLocaleString()}</div>
                </div>
              </div>
            </div>

            <div className="mt-auto pt-4 border-t border-[var(--color-border-subtle)]">
              <button 
                onClick={() => {
                  if (kiteConnected) {
                    setShowTradeModal(true);
                  } else {
                    alert("Please connect to Kite broker first to execute trades.");
                    setShowKiteConnect(true);
                  }
                }}
                className="w-full py-3 rounded-lg font-bold text-white transition-opacity hover:opacity-90"
                style={{ background: "var(--color-accent-blue)" }}
              >
                Trade {quote.symbol as string}
              </button>
            </div>
          </div>

          {/* Chart */}
          <div className="card lg:col-span-2">
            <div className="flex justify-between items-center mb-4">
              <h3 className="font-bold text-[var(--color-text-primary)]">Price History</h3>
              <div className="flex bg-[var(--color-bg-secondary)] rounded-lg p-1">
                {["5d", "1mo", "3mo", "6mo", "1y"].map(p => (
                  <button 
                    key={p} 
                    onClick={() => setHistoryPeriod(p)}
                    className={`px-3 py-1 text-xs font-semibold rounded-md transition-colors ${historyPeriod === p ? "bg-[var(--color-bg-card)] text-[var(--color-text-primary)] shadow-sm" : "text-[var(--color-text-muted)] hover:text-[var(--color-text-secondary)]"}`}
                  >
                    {p.toUpperCase()}
                  </button>
                ))}
              </div>
            </div>
            
            <div className="h-72 w-full">
              {history.length > 0 ? (
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={history} margin={{ top: 5, right: 0, left: 0, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-subtle)" vertical={false} />
                    <XAxis 
                      dataKey="date" 
                      stroke="var(--color-text-muted)" 
                      fontSize={10} 
                      tickLine={false} 
                      axisLine={false}
                      tickFormatter={(val) => new Date(val).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}
                    />
                    <YAxis 
                      yAxisId="price"
                      domain={['auto', 'auto']} 
                      stroke="var(--color-text-muted)" 
                      fontSize={10} 
                      tickLine={false} 
                      axisLine={false}
                      tickFormatter={(val) => `₹${val.toLocaleString()}`}
                    />
                    <RechartsTooltip 
                      contentStyle={{ background: "var(--color-bg-card)", borderColor: "var(--color-border-subtle)", borderRadius: 8, color: "var(--color-text-primary)" }}
                      labelStyle={{ color: "var(--color-text-muted)", marginBottom: 4 }}
                      formatter={(val: any, name: any) => [name === "close" ? `₹${Number(val).toLocaleString()}` : Number(val).toLocaleString(), name === "close" ? "Price" : "Volume"]}
                    />
                    <Line yAxisId="price" type="monotone" dataKey="close" stroke="var(--color-accent-blue)" strokeWidth={2} dot={false} activeDot={{ r: 6 }} />
                  </ComposedChart>
                </ResponsiveContainer>
              ) : (
                <div className="w-full h-full flex items-center justify-center text-[var(--color-text-muted)]">
                  {loading ? "Loading chart data..." : "No historical data available"}
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Option Chain */}
      {options.length > 0 && (
        <div className="card overflow-x-auto">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-semibold text-sm" style={{ color: "var(--color-text-secondary)" }}>
              OPTION CHAIN (Simulated)
            </h3>
            <div className="text-xs font-mono bg-[var(--color-bg-secondary)] px-2 py-1 rounded text-[var(--color-text-muted)]">
              ATM: {spotPrice.toFixed(0)}
            </div>
          </div>
          
          <table className="data-table text-xs">
            <thead>
              <tr>
                <th colSpan={4} className="text-center" style={{ color: "var(--color-accent-green)", borderBottom: "2px solid var(--color-accent-green)" }}>CALLS</th>
                <th className="text-center bg-[var(--color-bg-secondary)] border-b-2 border-[var(--color-border-subtle)]">STRIKE</th>
                <th colSpan={4} className="text-center" style={{ color: "var(--color-accent-red)", borderBottom: "2px solid var(--color-accent-red)" }}>PUTS</th>
              </tr>
              <tr>
                <th>OI</th><th>IV</th><th>Delta</th><th>Price</th>
                <th className="text-center bg-[var(--color-bg-secondary)]">K</th>
                <th>Price</th><th>Delta</th><th>IV</th><th>OI</th>
              </tr>
            </thead>
            <tbody>
              {options.map((opt, i) => {
                const call = opt.call as Record<string, unknown> | undefined;
                const put = opt.put as Record<string, unknown> | undefined;
                const strike = opt.strike as number;
                const isATM = Math.abs(strike - spotPrice) < (spotPrice * 0.01);
                return (
                  <tr key={i} style={isATM ? { background: "rgba(41,98,255,0.05)" } : {}}>
                    <td>{call ? Number(call.oi).toLocaleString() : "—"}</td>
                    <td>{call ? `${(Number(call.iv)).toFixed(1)}%` : "—"}</td>
                    <td>{call ? Number(call.delta).toFixed(3) : "—"}</td>
                    <td className="font-bold" style={{ color: "var(--color-accent-green)" }}>{call ? Number(call.price).toFixed(2) : "—"}</td>
                    <td className="text-center font-bold bg-[var(--color-bg-secondary)]" style={{ color: isATM ? "var(--color-accent-blue)" : "var(--color-text-primary)" }}>
                      {strike}
                    </td>
                    <td className="font-bold" style={{ color: "var(--color-accent-red)" }}>{put ? Number(put.price).toFixed(2) : "—"}</td>
                    <td>{put ? Number(put.delta).toFixed(3) : "—"}</td>
                    <td>{put ? `${(Number(put.iv)).toFixed(1)}%` : "—"}</td>
                    <td>{put ? Number(put.oi).toLocaleString() : "—"}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Trade Modal */}
      {showTradeModal && accessToken && quote && (
        <KiteTradeModal 
          symbol={quote.symbol as string} 
          lastPrice={Number(quote.last_price)} 
          token={accessToken}
          onClose={() => setShowTradeModal(false)} 
        />
      )}
    </div>
  );
}

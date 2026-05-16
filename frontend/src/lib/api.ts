/**
 * API Client — Typed fetch wrapper with JWT auth
 */

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

interface ApiOptions {
  method?: string;
  body?: unknown;
  token?: string | null;
}

class ApiError extends Error {
  status: number;
  data: unknown;
  constructor(message: string, status: number, data?: unknown) {
    super(message);
    this.status = status;
    this.data = data;
  }
}

async function apiFetch<T>(endpoint: string, opts: ApiOptions = {}): Promise<T> {
  const { method = "GET", body, token } = opts;

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
  };
  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(`${API_URL}${endpoint}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    const data = await res.json().catch(() => null);
    throw new ApiError(
      data?.detail || `API Error ${res.status}`,
      res.status,
      data
    );
  }

  if (res.status === 204) return null as T;
  return res.json();
}

// ============================================================
// Auth
// ============================================================

export interface UserData {
  id: string;
  email: string;
  full_name: string;
  role: string;
  is_active: boolean;
  created_at: string;
}

export interface LoginResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  user: UserData;
}

export const authApi = {
  sendOtp: (email: string) =>
    apiFetch<{ message: string; email: string }>("/auth/send-otp", { method: "POST", body: { email } }),

  register: (data: { email: string; password: string; full_name: string; otp: string; role?: string }) =>
    apiFetch<UserData>("/auth/register", { method: "POST", body: data }),

  login: (email: string, password: string) =>
    apiFetch<LoginResponse>("/auth/login", { method: "POST", body: { email, password } }),

  refresh: (refreshToken: string) =>
    apiFetch<{ access_token: string }>("/auth/refresh", { method: "POST", body: { refresh_token: refreshToken } }),

  me: (token: string) =>
    apiFetch<UserData>("/auth/me", { token }),
};

// ============================================================
// Pricing
// ============================================================

export interface PriceResult {
  model: string;
  price: number;
  elapsed_ms: number;
  [key: string]: unknown;
}

export const pricingApi = {
  blackScholes: (data: Record<string, unknown>, token: string) =>
    apiFetch<PriceResult>("/price/black-scholes", { method: "POST", body: data, token }),

  monteCarlo: (data: Record<string, unknown>, token: string) =>
    apiFetch<PriceResult>("/price/monte-carlo", { method: "POST", body: data, token }),

  binomial: (data: Record<string, unknown>, token: string) =>
    apiFetch<PriceResult>("/price/binomial", { method: "POST", body: data, token }),
};

// ============================================================
// Greeks
// ============================================================

export interface GreeksResult {
  greeks: { delta: number; gamma: number; vega: number; theta: number; rho: number };
  method: string;
  elapsed_ms: number;
}

export const greeksApi = {
  calculate: (data: Record<string, unknown>, token: string) =>
    apiFetch<GreeksResult>("/greeks/calculate", { method: "POST", body: data, token }),
};

// ============================================================
// Strategies
// ============================================================

export interface StrategyResult {
  strategy: { id: string; name: string };
  spots: number[];
  pnl: number[];
  max_profit: number;
  max_loss: number;
  entry_premium: number;
  breakevens: number[];
  greeks: Record<string, number>;
  legs: Array<{ type: string; side: string; strike: number; premium: number }>;
  [key: string]: unknown;
}

export const strategiesApi = {
  list: (token: string) => apiFetch<{ strategies: Array<{ id: string; name: string; n_legs: number }> }>("/strategies/list", { token }),
  simulate: (data: Record<string, unknown>, token: string) =>
    apiFetch<StrategyResult>("/strategies/simulate", { method: "POST", body: data, token }),
};

// ============================================================
// Scenario
// ============================================================

export const scenarioApi = {
  stressTest: (data: Record<string, unknown>, token: string) =>
    apiFetch<Record<string, unknown>>("/scenario/stress-test", { method: "POST", body: data, token }),
  heatmap: (data: Record<string, unknown>, token: string) =>
    apiFetch<Record<string, unknown>>("/scenario/heatmap", { method: "POST", body: data, token }),
};

// ============================================================
// Risk
// ============================================================

export const riskApi = {
  portfolio: (data: Record<string, unknown>, token: string) =>
    apiFetch<Record<string, unknown>>("/risk/portfolio", { method: "POST", body: data, token }),
};

// ============================================================
// Market Data
// ============================================================

export const marketApi = {
  quote: (symbol: string, token: string) =>
    apiFetch<Record<string, unknown>>(`/market/quote/${symbol}`, { token }),
  optionChain: (symbol: string, token: string) =>
    apiFetch<Record<string, unknown>>(`/market/option-chain/${symbol}`, { token }),
  history: (symbol: string, period: string, token: string) =>
    apiFetch<{ symbol: string; period: string; data: Array<{ date: string; open: number; high: number; low: number; close: number; volume: number }> }>(`/market/history/${symbol}?period=${period}`, { token }),
  search: (query: string, token: string) =>
    apiFetch<{ results: Array<{ symbol: string; name: string; exchange: string }> }>(`/market/search?q=${encodeURIComponent(query)}`, { token }),
  kiteConnect: (data: { api_key: string; api_secret: string; request_token: string }, token: string) =>
    apiFetch<Record<string, unknown>>("/market/kite/connect", { method: "POST", body: data, token }),
  kiteDisconnect: (token: string) =>
    apiFetch<Record<string, unknown>>("/market/kite/disconnect", { method: "POST", token }),
  kiteStatus: (token: string) =>
    apiFetch<{ connected: boolean }>("/market/kite/status", { token }),
  kiteOrder: (data: Record<string, unknown>, token: string) =>
    apiFetch<Record<string, unknown>>("/market/kite/order", { method: "POST", body: data, token }),
  kiteOrders: (token: string) =>
    apiFetch<Record<string, unknown>>("/market/kite/orders", { token }),
  kitePositions: (token: string) =>
    apiFetch<Record<string, unknown>>("/market/kite/positions", { token }),
  kiteHoldings: (token: string) =>
    apiFetch<Record<string, unknown>>("/market/kite/holdings", { token }),
};

// ============================================================
// Portfolios (persisted)
// ============================================================

export interface PortfolioData {
  id: string;
  name: string;
  description: string | null;
  positions: Array<Record<string, unknown>>;
  is_active: boolean;
  created_at: string;
  updated_at: string;
}

export const portfoliosApi = {
  list: (token: string) =>
    apiFetch<{ portfolios: PortfolioData[]; count: number }>("/portfolios", { token }),
  create: (data: Record<string, unknown>, token: string) =>
    apiFetch<PortfolioData>("/portfolios", { method: "POST", body: data, token }),
  get: (id: string, token: string) =>
    apiFetch<PortfolioData>(`/portfolios/${id}`, { token }),
  update: (id: string, data: Record<string, unknown>, token: string) =>
    apiFetch<PortfolioData>(`/portfolios/${id}`, { method: "PUT", body: data, token }),
  delete: (id: string, token: string) =>
    apiFetch<null>(`/portfolios/${id}`, { method: "DELETE", token }),
  calculateRisk: (id: string, token: string) =>
    apiFetch<Record<string, unknown>>(`/portfolios/${id}/calculate-risk`, { method: "POST", token }),
};

// ============================================================
// Trades
// ============================================================

export interface TradeData {
  id: string;
  side: string;
  option_type: string;
  spot_at_entry: number;
  strike: number;
  premium: number;
  quantity: number;
  sigma_at_entry: number;
  T_at_entry: number;
  status: string;
  traded_at: string;
  closed_at: string | null;
  close_premium: number | null;
  notes: string | null;
  portfolio_id: string | null;
  notional: number;
}

export const tradesApi = {
  book: (data: Record<string, unknown>, token: string) =>
    apiFetch<TradeData>("/trades", { method: "POST", body: data, token }),
  list: (token: string, status?: string) =>
    apiFetch<{ trades: TradeData[]; count: number }>(`/trades${status ? `?status=${status}` : ""}`, { token }),
  get: (id: string, token: string) =>
    apiFetch<TradeData>(`/trades/${id}`, { token }),
  close: (id: string, token: string, closePremium?: number) =>
    apiFetch<TradeData>(`/trades/${id}/close`, { method: "PUT", body: closePremium ? { close_premium: closePremium } : {}, token }),
  positions: (token: string) =>
    apiFetch<Record<string, unknown>>("/trades/positions", { token }),
};

// ============================================================
// Orders (OMS)
// ============================================================

export interface OrderData {
  id: string;
  side: string;
  option_type: string;
  order_type: string;
  spot_price: number;
  strike: number;
  T: number;
  sigma: number;
  quantity: number;
  filled_quantity: number;
  avg_fill_price: number | null;
  limit_price: number | null;
  status: string;
  risk_check_result: Record<string, unknown> | null;
  rejection_reason: string | null;
  submitted_at: string;
  filled_at: string | null;
  cancelled_at: string | null;
  portfolio_id: string | null;
  notes: string | null;
}

export const ordersApi = {
  submit: (data: Record<string, unknown>, token: string) =>
    apiFetch<OrderData>("/orders", { method: "POST", body: data, token }),
  list: (token: string, status?: string) =>
    apiFetch<{ orders: OrderData[]; count: number }>(`/orders${status ? `?status=${status}` : ""}`, { token }),
  get: (id: string, token: string) =>
    apiFetch<OrderData & { executions: unknown[] }>(`/orders/${id}`, { token }),
  cancel: (id: string, token: string) =>
    apiFetch<OrderData>(`/orders/${id}`, { method: "DELETE", token }),
  manualFill: (id: string, token: string) =>
    apiFetch<OrderData>(`/orders/${id}/fill`, { method: "POST", token }),
  riskCheck: (data: Record<string, unknown>, token: string) =>
    apiFetch<Record<string, unknown>>("/orders/risk-check", { method: "POST", body: data, token }),
  executions: (token: string) =>
    apiFetch<{ executions: unknown[]; count: number }>("/orders/executions/history", { token }),
};

// ============================================================
// Regulatory (Basel III/IV)
// ============================================================

export const regulatoryApi = {
  var: (data: Record<string, unknown>, token: string) =>
    apiFetch<Record<string, unknown>>("/regulatory/var", { method: "POST", body: data, token }),
  stressedVar: (data: Record<string, unknown>, token: string) =>
    apiFetch<Record<string, unknown>>("/regulatory/stressed-var", { method: "POST", body: data, token }),
  capitalCharge: (data: Record<string, unknown>, token: string) =>
    apiFetch<Record<string, unknown>>("/regulatory/capital-charge", { method: "POST", body: data, token }),
  leverage: (data: Record<string, unknown>, token: string) =>
    apiFetch<Record<string, unknown>>("/regulatory/leverage", { method: "POST", body: data, token }),
  concentration: (data: Record<string, unknown>, token: string) =>
    apiFetch<Record<string, unknown>>("/regulatory/concentration", { method: "POST", body: data, token }),
  fullReport: (data: Record<string, unknown>, token: string) =>
    apiFetch<Record<string, unknown>>("/regulatory/full-report", { method: "POST", body: data, token }),
};

// ============================================================
// Admin
// ============================================================

export const adminApi = {
  listUsers: (token: string) =>
    apiFetch<{ users: UserData[]; count: number }>("/admin/users", { token }),
  updateRole: (id: string, role: string, token: string) =>
    apiFetch<UserData>(`/admin/users/${id}/role`, { method: "PUT", body: { role }, token }),
  auditLog: (token: string, limit = 100) =>
    apiFetch<{ audit_logs: unknown[]; count: number }>(`/admin/audit-log?limit=${limit}`, { token }),
  systemMetrics: (token: string) =>
    apiFetch<Record<string, unknown>>("/admin/system", { token }),
  tenantsList: (token: string) =>
    apiFetch<Record<string, unknown>>("/admin/tenants", { token }),
  createTenant: (data: Record<string, unknown>, token: string) =>
    apiFetch<Record<string, unknown>>("/admin/tenants", { method: "POST", body: data, token }),
};

// ============================================================
// Health
// ============================================================

export const healthApi = {
  check: () => apiFetch<Record<string, unknown>>("/health"),
  deepCheck: () => apiFetch<Record<string, unknown>>("/health/deep"),
};

export { ApiError };

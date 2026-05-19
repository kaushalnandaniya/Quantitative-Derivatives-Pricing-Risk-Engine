"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";

/* ─── Feature data ─── */
const FEATURES = [
  {
    tag: "STRATEGY BUILDER",
    title: "Build Strategies Without Spreadsheets",
    desc: "Choose from 8+ pre-built strategies or create your own custom options strategies. Visualize P&L curves, analyze Greeks, and understand your risk before placing a trade.",
    items: ["Bull Call Spread", "Iron Condor", "Straddle", "Butterfly", "Custom Multi-Leg"],
    accent: "var(--color-accent-blue)",
  },
  {
    tag: "RISK ENGINE",
    title: "Institutional-Grade Risk Analytics",
    desc: "Full Basel III/IV regulatory engine with VaR, Stressed VaR, CVaR, capital charges, and concentration risk. Monitor your portfolio exposure in real-time.",
    items: ["Value at Risk (VaR)", "Conditional VaR", "Stress Testing", "Capital Charges", "Portfolio Greeks"],
    accent: "var(--color-accent-green)",
  },
  {
    tag: "PRICING MODELS",
    title: "Three Pricing Engines, One Platform",
    desc: "Black-Scholes closed-form, Monte Carlo with variance reduction, and Binomial Tree models. Full Greeks suite with real-time IV surface generation.",
    items: ["Black-Scholes-Merton", "Monte Carlo (100K paths)", "Binomial Trees", "Volatility Surface", "Real-time Greeks"],
    accent: "var(--color-accent-purple)",
  },
  {
    tag: "BACKTESTING",
    title: "Test Before You Trade",
    desc: "Backtest any strategy against historical weekly expiries. See equity curves, win rates, and per-trade P&L breakdowns before risking real capital.",
    items: ["Historical Backtests", "Equity Curves", "Win Rate Analytics", "Per-Trade P&L", "Strategy Comparison"],
    accent: "var(--color-accent-amber)",
  },
];

const STATS = [
  { value: "3", label: "Pricing Models" },
  { value: "8+", label: "Strategies" },
  { value: "5", label: "Greeks" },
  { value: "∞", label: "Custom Legs" },
];

export default function LoginPage() {
  const router = useRouter();
  const { login, sendOtp, register, forgotPassword, resetPassword, restore, isAuthenticated, isLoading } = useAuth();
  const [mode, setMode] = useState<"login" | "register" | "verify" | "forgot" | "reset">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [otp, setOtp] = useState(["", "", "", "", "", ""]);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const [showAuthModal, setShowAuthModal] = useState(false);
  const otpRefs = useRef<(HTMLInputElement | null)[]>([]);

  useEffect(() => { restore(); }, [restore]);
  useEffect(() => { if (isAuthenticated) router.push("/dashboard"); }, [isAuthenticated, router]);
  useEffect(() => {
    if (countdown <= 0) return;
    const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
    return () => clearTimeout(timer);
  }, [countdown]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: "var(--color-bg-primary)" }}>
        <div className="w-10 h-10 border-2 border-[var(--color-accent-blue)] border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  /* ─── Auth handlers ─── */
  const handleOtpChange = (index: number, value: string) => {
    if (value.length > 1) value = value.slice(-1);
    if (value && !/^\d$/.test(value)) return;
    const newOtp = [...otp];
    newOtp[index] = value;
    setOtp(newOtp);
    if (value && index < 5) otpRefs.current[index + 1]?.focus();
  };
  const handleOtpKeyDown = (index: number, e: React.KeyboardEvent) => {
    if (e.key === "Backspace" && !otp[index] && index > 0) otpRefs.current[index - 1]?.focus();
  };
  const handleOtpPaste = (e: React.ClipboardEvent) => {
    e.preventDefault();
    const pasted = e.clipboardData.getData("text").replace(/\D/g, "").slice(0, 6);
    if (pasted.length === 6) { setOtp(pasted.split("")); otpRefs.current[5]?.focus(); }
  };
  const handleSendOtp = async () => {
    setError(""); setSuccess(""); setLoading(true);
    try { await sendOtp(email); setMode("verify"); setCountdown(300); setSuccess("Verification code sent!"); }
    catch (err: unknown) { setError(err instanceof Error ? err.message : "Failed to send code"); }
    finally { setLoading(false); }
  };
  const handleResendOtp = async () => {
    if (countdown > 0) return;
    setError(""); setLoading(true);
    try { await sendOtp(email); setCountdown(300); setSuccess("New code sent!"); setOtp(["", "", "", "", "", ""]); }
    catch (err: unknown) { setError(err instanceof Error ? err.message : "Failed to resend"); }
    finally { setLoading(false); }
  };
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault(); setError(""); setSuccess(""); setLoading(true);
    try {
      if (mode === "register") { await handleSendOtp(); setLoading(false); return; }
      if (mode === "forgot") {
        await forgotPassword(email);
        setMode("reset"); setCountdown(300);
        setSuccess("Reset code sent to your email!");
        setLoading(false); return;
      }
      if (mode === "verify") {
        const otpString = otp.join("");
        if (otpString.length !== 6) { setError("Enter 6-digit code"); setLoading(false); return; }
        await register({ email, password, full_name: fullName, otp: otpString });
        setSuccess("Account created!"); await login(email, password);
        setLoading(false); return;
      }
      if (mode === "reset") {
        const otpString = otp.join("");
        if (otpString.length !== 6) { setError("Enter 6-digit code"); setLoading(false); return; }
        if (password.length < 8) { setError("Password must be at least 8 characters"); setLoading(false); return; }
        await resetPassword({ email, otp: otpString, new_password: password });
        setSuccess("Password reset successfully!");
        setMode("login"); setPassword(""); setOtp(["", "", "", "", "", ""]);
        setLoading(false); return;
      }
      await login(email, password);
    } catch (err: unknown) { setError(err instanceof Error ? err.message : "Something went wrong"); }
    finally { setLoading(false); }
  };
  const formatCountdown = (s: number) => `${Math.floor(s / 60)}:${(s % 60).toString().padStart(2, "0")}`;

  /* ─── Auth Form Render Helper ─── */
  const renderAuthForm = (extraClass = "") => (
    <div className={`card p-6 lg:p-8 ${extraClass}`} style={{ background: "var(--color-bg-card)", border: "1px solid var(--color-border)" }}>
      <h2 className="text-xl font-bold mb-1" style={{ color: "var(--color-text-primary)" }}>
        {mode === "login" ? "Welcome back" : mode === "register" ? "Create account" : mode === "forgot" ? "Reset Password" : mode === "reset" ? "New Password" : "Verify email"}
      </h2>
      <p className="text-sm mb-6" style={{ color: "var(--color-text-secondary)" }}>
        {mode === "login" ? "Sign in to access your dashboard" : mode === "register" ? "Register to start trading" : mode === "forgot" ? "Enter your email to receive a reset code" : `Code sent to ${email}`}
      </p>
      <form onSubmit={handleSubmit} className="space-y-4">
        {(mode === "verify" || mode === "reset") && (
          <>
            <div className="flex justify-center gap-2.5" onPaste={handleOtpPaste}>
              {otp.map((digit, i) => (
                <input key={i} ref={(el) => { otpRefs.current[i] = el; }} type="text" inputMode="numeric" maxLength={1}
                  value={digit} onChange={(e) => handleOtpChange(i, e.target.value)} onKeyDown={(e) => handleOtpKeyDown(i, e)}
                  className="input-field text-center font-bold" style={{ width: 48, height: 52, fontSize: 20, fontFamily: "var(--font-mono)" }} />
              ))}
            </div>
            <div className="text-center text-sm" style={{ color: "var(--color-text-secondary)" }}>
              {countdown > 0 ? <span>Expires in <strong style={{ color: "var(--color-accent-blue)" }}>{formatCountdown(countdown)}</strong></span> : <span>Code expired.</span>}
              <br />
              {mode === "verify" && (
                <button type="button" onClick={handleResendOtp} disabled={loading} className="mt-1 font-semibold" style={{ color: "var(--color-accent-blue)", background: "none", border: "none" }}>Resend code</button>
              )}
            </div>
          </>
        )}
        {mode === "register" && (
          <div><label className="label">Full Name</label><input type="text" className="input-field" value={fullName} onChange={e => setFullName(e.target.value)} placeholder="John Doe" required /></div>
        )}
        {(mode === "login" || mode === "register" || mode === "forgot") && (
          <div><label className="label">Email</label><input type="email" className="input-field" value={email} onChange={e => setEmail(e.target.value)} placeholder="trader@institution.com" required /></div>
        )}
        {(mode === "login" || mode === "register" || mode === "reset") && (
          <div>
            <div className="flex justify-between">
              <label className="label">{mode === "reset" ? "New Password" : "Password"}</label>
              {mode === "login" && (
                <button type="button" onClick={() => { setMode("forgot"); setError(""); setSuccess(""); }} className="text-xs font-semibold hover:underline" style={{ color: "var(--color-accent-blue)" }}>Forgot?</button>
              )}
            </div>
            <input type="password" className="input-field" value={password} onChange={e => setPassword(e.target.value)} placeholder="••••••••" required minLength={8} />
          </div>
        )}
        {error && <div className="text-sm px-3 py-2 rounded-lg" style={{ background: "rgba(242,54,69,0.1)", color: "var(--color-accent-red)" }}>{error}</div>}
        {success && <div className="text-sm px-3 py-2 rounded-lg" style={{ background: "rgba(41,98,255,0.1)", color: "var(--color-accent-blue)" }}>{success}</div>}
        <button type="submit" className="btn-primary w-full" disabled={loading}>
          {loading ? "Processing..." : mode === "login" ? "Sign In" : mode === "register" ? "Send Verification Code" : mode === "forgot" ? "Send Reset Code" : mode === "reset" ? "Update Password" : "Verify & Create Account"}
        </button>
      </form>
      <div className="mt-5 text-center text-sm" style={{ color: "var(--color-text-secondary)" }}>
        {mode === "login" ? (
          <>Don&apos;t have an account? <button onClick={() => { setMode("register"); setError(""); setSuccess(""); }} className="font-semibold" style={{ color: "var(--color-accent-blue)" }}>Register</button></>
        ) : mode === "register" ? (
          <>Already have an account? <button onClick={() => { setMode("login"); setError(""); setSuccess(""); }} className="font-semibold" style={{ color: "var(--color-accent-blue)" }}>Sign in</button></>
        ) : mode === "forgot" ? (
          <button onClick={() => { setMode("login"); setError(""); setSuccess(""); }} className="font-semibold" style={{ color: "var(--color-accent-blue)" }}>← Back to login</button>
        ) : mode === "reset" ? (
          <button onClick={() => { setMode("forgot"); setError(""); setSuccess(""); setOtp(["", "", "", "", "", ""]); }} className="font-semibold" style={{ color: "var(--color-accent-blue)" }}>← Back</button>
        ) : (
          <button onClick={() => { setMode("register"); setError(""); setSuccess(""); setOtp(["", "", "", "", "", ""]); }} className="font-semibold" style={{ color: "var(--color-accent-blue)" }}>← Back</button>
        )}
      </div>
    </div>
  );

  return (
    <div className="min-h-screen" style={{ background: "var(--color-bg-primary)", color: "var(--color-text-primary)" }}>

      {/* ═══ NAVBAR ═══ */}
      <nav className="sticky top-0 z-50 backdrop-blur-xl border-b" style={{ background: "rgba(18,18,18,0.85)", borderColor: "var(--color-border-subtle)" }}>
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg flex items-center justify-center font-extrabold text-white text-sm" style={{ background: "var(--color-accent-blue)" }}>Q</div>
            <span className="text-lg font-bold">Quant Engine</span>
          </div>
          <div className="hidden md:flex items-center gap-8 text-sm" style={{ color: "var(--color-text-secondary)" }}>
            <a href="#features" className="hover:text-[var(--color-text-primary)] transition-colors">Features</a>
            <a href="#tools" className="hover:text-[var(--color-text-primary)] transition-colors">Tools</a>
            <a href="#pricing" className="hover:text-[var(--color-text-primary)] transition-colors">Pricing</a>
          </div>
          <button onClick={() => setShowAuthModal(true)} className="btn-primary px-5 py-2 text-sm font-semibold">Login</button>
        </div>
      </nav>

      {/* ═══ HERO ═══ */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0" style={{ background: "radial-gradient(ellipse at 20% 30%, rgba(41,98,255,0.08) 0%, transparent 60%), radial-gradient(ellipse at 80% 70%, rgba(157,78,221,0.06) 0%, transparent 50%)" }} />
        <div className="max-w-7xl mx-auto px-6 py-20 lg:py-28 relative z-10">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left — Copy */}
            <div>
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-semibold mb-6" style={{ background: "rgba(41,98,255,0.1)", color: "var(--color-accent-blue)", border: "1px solid rgba(41,98,255,0.2)" }}>
                <span className="w-2 h-2 rounded-full bg-[var(--color-accent-green)] pulse" />
                Live on Vercel • Free to Use
              </div>
              <h1 className="text-4xl lg:text-6xl font-extrabold leading-[1.1] mb-6">
                Institutional-Grade{" "}
                <span className="bg-gradient-to-r from-[var(--color-accent-blue)] to-[var(--color-accent-purple)] bg-clip-text text-transparent">Options Analytics</span>{" "}
                Platform
              </h1>
              <p className="text-lg leading-relaxed max-w-lg mb-8" style={{ color: "var(--color-text-secondary)" }}>
                Build strategies, backtest with historical data, analyze Greeks in real-time, and manage portfolio risk — all from one professional dashboard.
              </p>
              <div className="flex flex-wrap gap-3 mb-10">
                <button onClick={() => setShowAuthModal(true)} className="btn-primary px-8 py-3.5 text-sm font-bold">
                  Get Started — Free
                </button>
                <a href="#features" className="btn-ghost px-8 py-3.5 text-sm font-bold border border-[var(--color-border)]">
                  Explore Features
                </a>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-4 gap-6">
                {STATS.map(s => (
                  <div key={s.label}>
                    <div className="text-2xl lg:text-3xl font-bold" style={{ fontFamily: "var(--font-mono)", color: "var(--color-accent-blue)" }}>{s.value}</div>
                    <div className="text-[11px] mt-1" style={{ color: "var(--color-text-muted)" }}>{s.label}</div>
                  </div>
                ))}
              </div>
            </div>

            {/* Right — Auth Form */}
            <div className="hidden lg:block">
              {renderAuthForm()}
            </div>
          </div>
        </div>
      </section>

      {/* ═══ FEATURES ═══ */}
      <section id="features" className="max-w-7xl mx-auto px-6 py-20">
        <div className="text-center mb-16">
          <h2 className="text-3xl lg:text-4xl font-extrabold mb-4">Everything You Need to Trade Options</h2>
          <p className="text-sm max-w-xl mx-auto" style={{ color: "var(--color-text-secondary)" }}>Professional-grade tools for strategy building, risk management, backtesting, and real-time analytics.</p>
        </div>
        <div className="grid md:grid-cols-2 gap-6">
          {FEATURES.map((f, i) => (
            <div key={i} className="card p-6 group hover:border-[var(--color-border)] transition-all duration-300" style={{ border: "1px solid var(--color-border-subtle)" }}>
              <span className="text-[10px] font-bold uppercase tracking-widest" style={{ color: f.accent }}>{f.tag}</span>
              <h3 className="text-xl font-bold mt-2 mb-3">{f.title}</h3>
              <p className="text-sm leading-relaxed mb-4" style={{ color: "var(--color-text-secondary)" }}>{f.desc}</p>
              <div className="flex flex-wrap gap-2">
                {f.items.map(item => (
                  <span key={item} className="text-[10px] px-2.5 py-1 rounded-full font-medium" style={{ background: `${f.accent}15`, color: f.accent, border: `1px solid ${f.accent}25` }}>{item}</span>
                ))}
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ═══ TOOLS SECTION ═══ */}
      <section id="tools" className="border-t border-b" style={{ borderColor: "var(--color-border-subtle)", background: "var(--color-bg-card)" }}>
        <div className="max-w-7xl mx-auto px-6 py-20">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-extrabold mb-3">Powerful Tools, Zero Complexity</h2>
            <p className="text-sm" style={{ color: "var(--color-text-secondary)" }}>Access all analytics from a single professional dashboard</p>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {[
              { icon: "📊", name: "Strategy Builder", desc: "Build & analyze" },
              { icon: "📈", name: "Option Chain", desc: "Live strikes & IVs" },
              { icon: "🎯", name: "Backtester", desc: "Historical testing" },
              { icon: "⚡", name: "Live Streaming", desc: "WebSocket prices" },
              { icon: "🛡️", name: "Risk Engine", desc: "VaR & CVaR" },
              { icon: "📋", name: "Trade Blotter", desc: "Order management" },
            ].map(t => (
              <div key={t.name} className="text-center p-4 rounded-xl hover:bg-[var(--color-bg-hover)] transition-colors cursor-default">
                <div className="text-3xl mb-2">{t.icon}</div>
                <div className="text-sm font-bold mb-0.5">{t.name}</div>
                <div className="text-[10px]" style={{ color: "var(--color-text-muted)" }}>{t.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ═══ PRICING ═══ */}
      <section id="pricing" className="max-w-7xl mx-auto px-6 py-20">
        <div className="text-center mb-12">
          <h2 className="text-3xl font-extrabold mb-3">Simple Pricing</h2>
          <p className="text-sm" style={{ color: "var(--color-text-secondary)" }}>Full access. No hidden fees. Free forever for personal use.</p>
        </div>
        <div className="grid md:grid-cols-2 gap-6 max-w-2xl mx-auto">
          <div className="card p-6" style={{ border: "1px solid var(--color-border-subtle)" }}>
            <h3 className="text-lg font-bold mb-1">Free</h3>
            <div className="text-3xl font-extrabold mb-4" style={{ fontFamily: "var(--font-mono)" }}>₹0<span className="text-sm font-normal" style={{ color: "var(--color-text-muted)" }}> /forever</span></div>
            <ul className="space-y-2 text-sm mb-6" style={{ color: "var(--color-text-secondary)" }}>
              {["Strategy Builder", "Payoff Charts", "Option Chain", "OI Analysis", "Greeks Calculator", "Risk Engine"].map(f => (
                <li key={f} className="flex items-center gap-2"><span style={{ color: "var(--color-accent-green)" }}>✓</span> {f}</li>
              ))}
            </ul>
            <button onClick={() => setShowAuthModal(true)} className="btn-primary w-full py-2.5 text-sm font-bold">Get Started</button>
          </div>
          <div className="card p-6 relative" style={{ border: "2px solid var(--color-accent-blue)" }}>
            <div className="absolute -top-3 right-4 text-[10px] font-bold px-3 py-1 rounded-full" style={{ background: "var(--color-accent-blue)", color: "#fff" }}>COMING SOON</div>
            <h3 className="text-lg font-bold mb-1">Pro</h3>
            <div className="text-3xl font-extrabold mb-4" style={{ fontFamily: "var(--font-mono)" }}>₹499<span className="text-sm font-normal" style={{ color: "var(--color-text-muted)" }}> /month</span></div>
            <ul className="space-y-2 text-sm mb-6" style={{ color: "var(--color-text-secondary)" }}>
              {["Everything in Free", "Live NSE Data", "WebSocket Streaming", "Trade Execution", "Backtesting Engine", "Portfolio Greeks"].map(f => (
                <li key={f} className="flex items-center gap-2"><span style={{ color: "var(--color-accent-blue)" }}>✓</span> {f}</li>
              ))}
            </ul>
            <button disabled className="btn-ghost w-full py-2.5 text-sm font-bold border border-[var(--color-border)] opacity-50 cursor-not-allowed">Coming Soon</button>
          </div>
        </div>
      </section>

      {/* ═══ FOOTER ═══ */}
      <footer className="border-t" style={{ borderColor: "var(--color-border-subtle)" }}>
        <div className="max-w-7xl mx-auto px-6 py-10 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <div className="w-7 h-7 rounded-md flex items-center justify-center font-bold text-white text-xs" style={{ background: "var(--color-accent-blue)" }}>Q</div>
            <span className="text-sm font-bold">Quant Engine</span>
          </div>
          <div className="text-xs" style={{ color: "var(--color-text-muted)" }}>
            Built by <a href="https://github.com/kaushalnandaniya" className="font-semibold hover:text-[var(--color-text-secondary)]">Kaushal Nandaniya</a> • Open Source on <a href="https://github.com/kaushalnandaniya/Quantitative-Derivatives-Pricing-Risk-Engine" className="font-semibold hover:text-[var(--color-text-secondary)]">GitHub</a>
          </div>
        </div>
      </footer>

      {/* ═══ AUTH MODAL (Mobile + CTA clicks) ═══ */}
      {showAuthModal && (
        <div className="fixed inset-0 z-50 bg-black/70 flex items-center justify-center p-4" onClick={() => setShowAuthModal(false)}>
          <div className="w-full max-w-md" onClick={e => e.stopPropagation()}>
              {renderAuthForm()}
            </div>
        </div>
      )}
    </div>
  );
}

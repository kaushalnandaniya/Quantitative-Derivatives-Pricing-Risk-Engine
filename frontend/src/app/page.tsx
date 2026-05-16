"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const { login, register, restore, isAuthenticated, isLoading } = useAuth();
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    restore();
  }, [restore]);

  useEffect(() => {
    if (isAuthenticated) router.push("/dashboard");
  }, [isAuthenticated, router]);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: "var(--color-bg-primary)" }}>
        <div className="text-center">
          <div className="w-10 h-10 border-2 border-[var(--color-accent-blue)] border-t-transparent rounded-full animate-spin mx-auto mb-4" />
          <p style={{ color: "var(--color-text-secondary)" }}>Loading...</p>
        </div>
      </div>
    );
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);

    try {
      if (mode === "register") {
        await register({ email, password, full_name: fullName });
        setMode("login");
        setError("");
      }
      await login(email, password);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Something went wrong";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex" style={{ background: "var(--color-bg-primary)" }}>
      {/* Left Panel — Branding */}
      <div className="hidden lg:flex lg:w-1/2 flex-col justify-center px-16 relative overflow-hidden">
        <div className="absolute inset-0 opacity-10" style={{
          background: "radial-gradient(ellipse at 30% 50%, var(--color-accent-blue) 0%, transparent 70%)"
        }} />
        <div className="relative z-10">
          <div className="flex items-center gap-3 mb-8">
            <div className="w-12 h-12 rounded-xl flex items-center justify-center font-extrabold text-xl text-white"
              style={{ background: "var(--color-accent-blue)" }}>Q</div>
            <span className="text-2xl font-bold" style={{ color: "var(--color-text-primary)" }}>
              Quant Engine
            </span>
          </div>
          <h1 className="text-5xl font-extrabold leading-tight mb-6" style={{ color: "var(--color-text-primary)" }}>
            Institutional-Grade<br />
            <span style={{ color: "var(--color-accent-blue)" }}>Derivatives Platform</span>
          </h1>
          <p className="text-lg leading-relaxed max-w-md" style={{ color: "var(--color-text-secondary)" }}>
            Black-Scholes, Monte Carlo, Binomial pricing. Portfolio risk analytics with VaR/CVaR.
            Strategy simulation. Real-time market data. All in one platform.
          </p>

          <div className="mt-12 grid grid-cols-3 gap-6">
            {[
              { label: "Pricing Models", value: "3" },
              { label: "Greeks", value: "5" },
              { label: "Strategies", value: "8" },
            ].map((stat) => (
              <div key={stat.label}>
                <div className="text-3xl font-bold" style={{ fontFamily: "var(--font-mono)", color: "var(--color-accent-blue)" }}>
                  {stat.value}
                </div>
                <div className="text-xs font-medium mt-1" style={{ color: "var(--color-text-muted)" }}>
                  {stat.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Right Panel — Login Form */}
      <div className="flex-1 flex items-center justify-center px-6">
        <div className="w-full max-w-md">
          <div className="card p-8">
            <h2 className="text-2xl font-bold mb-2" style={{ color: "var(--color-text-primary)" }}>
              {mode === "login" ? "Welcome back" : "Create account"}
            </h2>
            <p className="text-sm mb-8" style={{ color: "var(--color-text-secondary)" }}>
              {mode === "login" ? "Sign in to access your dashboard" : "Register to start trading"}
            </p>

            <form onSubmit={handleSubmit} className="space-y-5">
              {mode === "register" && (
                <div>
                  <label className="label">Full Name</label>
                  <input
                    type="text"
                    className="input-field"
                    value={fullName}
                    onChange={(e) => setFullName(e.target.value)}
                    placeholder="John Doe"
                    required
                  />
                </div>
              )}

              <div>
                <label className="label">Email</label>
                <input
                  type="email"
                  className="input-field"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="trader@institution.com"
                  required
                />
              </div>

              <div>
                <label className="label">Password</label>
                <input
                  type="password"
                  className="input-field"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  required
                  minLength={8}
                />
              </div>

              {error && (
                <div className="text-sm px-3 py-2 rounded-lg" style={{
                  background: "rgba(255,59,48,0.1)", color: "var(--color-accent-red)"
                }}>
                  {error}
                </div>
              )}

              <button type="submit" className="btn-primary w-full" disabled={loading}>
                {loading ? "Processing..." : mode === "login" ? "Sign In" : "Create Account"}
              </button>
            </form>

            <div className="mt-6 text-center text-sm" style={{ color: "var(--color-text-secondary)" }}>
              {mode === "login" ? (
                <>
                  Don&apos;t have an account?{" "}
                  <button onClick={() => { setMode("register"); setError(""); }}
                    className="font-semibold" style={{ color: "var(--color-accent-blue)" }}>
                    Register
                  </button>
                </>
              ) : (
                <>
                  Already have an account?{" "}
                  <button onClick={() => { setMode("login"); setError(""); }}
                    className="font-semibold" style={{ color: "var(--color-accent-blue)" }}>
                    Sign in
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

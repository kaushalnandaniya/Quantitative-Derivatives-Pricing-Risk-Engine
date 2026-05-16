"use client";

import { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth";

export default function LoginPage() {
  const router = useRouter();
  const { login, sendOtp, register, restore, isAuthenticated, isLoading } = useAuth();
  const [mode, setMode] = useState<"login" | "register" | "verify">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [fullName, setFullName] = useState("");
  const [otp, setOtp] = useState(["", "", "", "", "", ""]);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [loading, setLoading] = useState(false);
  const [countdown, setCountdown] = useState(0);
  const otpRefs = useRef<(HTMLInputElement | null)[]>([]);

  useEffect(() => {
    restore();
  }, [restore]);

  useEffect(() => {
    if (isAuthenticated) router.push("/dashboard");
  }, [isAuthenticated, router]);

  // Countdown timer for OTP resend
  useEffect(() => {
    if (countdown <= 0) return;
    const timer = setTimeout(() => setCountdown(countdown - 1), 1000);
    return () => clearTimeout(timer);
  }, [countdown]);

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

  const handleOtpChange = (index: number, value: string) => {
    if (value.length > 1) value = value.slice(-1);
    if (value && !/^\d$/.test(value)) return;

    const newOtp = [...otp];
    newOtp[index] = value;
    setOtp(newOtp);

    // Auto-focus next input
    if (value && index < 5) {
      otpRefs.current[index + 1]?.focus();
    }
  };

  const handleOtpKeyDown = (index: number, e: React.KeyboardEvent) => {
    if (e.key === "Backspace" && !otp[index] && index > 0) {
      otpRefs.current[index - 1]?.focus();
    }
  };

  const handleOtpPaste = (e: React.ClipboardEvent) => {
    e.preventDefault();
    const pasted = e.clipboardData.getData("text").replace(/\D/g, "").slice(0, 6);
    if (pasted.length === 6) {
      const newOtp = pasted.split("");
      setOtp(newOtp);
      otpRefs.current[5]?.focus();
    }
  };

  const handleSendOtp = async () => {
    setError("");
    setSuccess("");
    setLoading(true);
    try {
      await sendOtp(email);
      setMode("verify");
      setCountdown(300); // 5 minutes
      setSuccess("Verification code sent! Check your email.");
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to send verification code";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleResendOtp = async () => {
    if (countdown > 0) return;
    setError("");
    setLoading(true);
    try {
      await sendOtp(email);
      setCountdown(300);
      setSuccess("New verification code sent!");
      setOtp(["", "", "", "", "", ""]);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to resend code";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    setLoading(true);

    try {
      if (mode === "register") {
        // Step 1: Send OTP
        await handleSendOtp();
        setLoading(false);
        return;
      }

      if (mode === "verify") {
        // Step 2: Verify OTP & register
        const otpString = otp.join("");
        if (otpString.length !== 6) {
          setError("Please enter the complete 6-digit code");
          setLoading(false);
          return;
        }
        await register({ email, password, full_name: fullName, otp: otpString });
        setSuccess("Account created! Signing you in...");
        // Auto-login after registration
        await login(email, password);
        setLoading(false);
        return;
      }

      // Login mode
      await login(email, password);
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Something went wrong";
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const formatCountdown = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, "0")}`;
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

      {/* Right Panel — Auth Form */}
      <div className="flex-1 flex items-center justify-center px-6">
        <div className="w-full max-w-md">
          <div className="card p-8">
            {/* Header */}
            <h2 className="text-2xl font-bold mb-2" style={{ color: "var(--color-text-primary)" }}>
              {mode === "login" ? "Welcome back" : mode === "register" ? "Create account" : "Verify email"}
            </h2>
            <p className="text-sm mb-8" style={{ color: "var(--color-text-secondary)" }}>
              {mode === "login"
                ? "Sign in to access your dashboard"
                : mode === "register"
                  ? "Register to start trading"
                  : `Enter the 6-digit code sent to ${email}`}
            </p>

            <form onSubmit={handleSubmit} className="space-y-5">
              {/* ===== VERIFY MODE: OTP Input ===== */}
              {mode === "verify" && (
                <>
                  {/* OTP Digit Inputs */}
                  <div className="flex justify-center gap-3" onPaste={handleOtpPaste}>
                    {otp.map((digit, i) => (
                      <input
                        key={i}
                        ref={(el) => { otpRefs.current[i] = el; }}
                        type="text"
                        inputMode="numeric"
                        maxLength={1}
                        value={digit}
                        onChange={(e) => handleOtpChange(i, e.target.value)}
                        onKeyDown={(e) => handleOtpKeyDown(i, e)}
                        className="input-field text-center font-bold"
                        style={{
                          width: "52px",
                          height: "56px",
                          fontSize: "22px",
                          fontFamily: "var(--font-mono)",
                          letterSpacing: "2px",
                          caretColor: "var(--color-accent-blue)",
                        }}
                      />
                    ))}
                  </div>

                  {/* Timer & Resend */}
                  <div className="text-center text-sm" style={{ color: "var(--color-text-secondary)" }}>
                    {countdown > 0 ? (
                      <span>Code expires in <strong style={{ color: "var(--color-accent-blue)" }}>{formatCountdown(countdown)}</strong></span>
                    ) : (
                      <span>Code expired.</span>
                    )}
                    <br />
                    <button
                      type="button"
                      onClick={handleResendOtp}
                      disabled={loading}
                      className="mt-2 font-semibold"
                      style={{
                        color: "var(--color-accent-blue)",
                        opacity: loading ? 0.5 : 1,
                        cursor: loading ? "not-allowed" : "pointer",
                        background: "none",
                        border: "none",
                      }}
                    >
                      Resend code
                    </button>
                  </div>
                </>
              )}

              {/* ===== REGISTER MODE: Name, Email, Password ===== */}
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

              {/* ===== LOGIN & REGISTER MODE: Email & Password ===== */}
              {mode !== "verify" && (
                <>
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
                </>
              )}

              {/* ===== Messages ===== */}
              {error && (
                <div className="text-sm px-3 py-2 rounded-lg" style={{
                  background: "rgba(255,59,48,0.1)", color: "var(--color-accent-red)"
                }}>
                  {error}
                </div>
              )}

              {success && (
                <div className="text-sm px-3 py-2 rounded-lg" style={{
                  background: "rgba(88,166,255,0.1)", color: "var(--color-accent-blue)"
                }}>
                  {success}
                </div>
              )}

              {/* ===== Submit Button ===== */}
              <button type="submit" className="btn-primary w-full" disabled={loading}>
                {loading
                  ? "Processing..."
                  : mode === "login"
                    ? "Sign In"
                    : mode === "register"
                      ? "Send Verification Code"
                      : "Verify & Create Account"}
              </button>
            </form>

            {/* ===== Mode Switcher ===== */}
            <div className="mt-6 text-center text-sm" style={{ color: "var(--color-text-secondary)" }}>
              {mode === "login" ? (
                <>
                  Don&apos;t have an account?{" "}
                  <button onClick={() => { setMode("register"); setError(""); setSuccess(""); }}
                    className="font-semibold" style={{ color: "var(--color-accent-blue)" }}>
                    Register
                  </button>
                </>
              ) : mode === "register" ? (
                <>
                  Already have an account?{" "}
                  <button onClick={() => { setMode("login"); setError(""); setSuccess(""); }}
                    className="font-semibold" style={{ color: "var(--color-accent-blue)" }}>
                    Sign in
                  </button>
                </>
              ) : (
                <>
                  <button onClick={() => { setMode("register"); setError(""); setSuccess(""); setOtp(["", "", "", "", "", ""]); }}
                    className="font-semibold" style={{ color: "var(--color-accent-blue)" }}>
                    ← Back to registration
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

"use client";

import { useEffect } from "react";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/lib/auth";

const navItems = [
  { href: "/dashboard", label: "Overview", icon: "📊" },
  { href: "/dashboard/orders", label: "OMS Blotter", icon: "⚡" },
  { href: "/dashboard/trades", label: "Positions", icon: "💼" },
  { href: "/dashboard/pricing", label: "Pricing Lab", icon: "💹" },
  { href: "/dashboard/greeks", label: "Greeks", icon: "Δ" },
  { href: "/dashboard/risk", label: "Risk Engine", icon: "🛡" },
  { href: "/dashboard/regulatory", label: "Basel III/IV", icon: "🏛" },
  { href: "/dashboard/strategy", label: "Strategy Sim", icon: "♟" },
  { href: "/dashboard/scenario", label: "Scenarios", icon: "🔥" },
  { href: "/dashboard/market", label: "Market Data", icon: "📈" },
  { href: "/dashboard/admin", label: "Admin Panel", icon: "⚙️" },
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const { user, isAuthenticated, isLoading, restore, logout } = useAuth();

  useEffect(() => {
    restore();
  }, [restore]);

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push("/");
    }
  }, [isLoading, isAuthenticated, router]);

  if (isLoading || !isAuthenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: "var(--color-bg-primary)" }}>
        <div className="w-8 h-8 border-2 border-[var(--color-accent-blue)] border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex" style={{ background: "var(--color-bg-primary)" }}>
      {/* Sidebar */}
      <aside className="w-60 flex-shrink-0 flex flex-col border-r"
        style={{ background: "var(--color-bg-card)", borderColor: "var(--color-border-subtle)" }}>

        {/* Logo */}
        <div className="p-5 flex items-center gap-3 border-b" style={{ borderColor: "var(--color-border-subtle)" }}>
          <div className="w-9 h-9 rounded-lg flex items-center justify-center font-extrabold text-sm text-white"
            style={{ background: "var(--color-accent-blue)" }}>Q</div>
          <div>
            <div className="font-bold text-sm" style={{ color: "var(--color-text-primary)" }}>Quant Engine</div>
            <div className="text-[10px] font-medium" style={{ color: "var(--color-text-muted)" }}>v3.0.0</div>
          </div>
        </div>

        {/* Nav Links */}
        <nav className="flex-1 py-4 px-3 space-y-1">
          {navItems.map((item) => {
            const isActive = pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href));
            return (
              <Link key={item.href} href={item.href}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-all duration-150 ${
                  isActive ? "" : "hover:opacity-80"
                }`}
                style={{
                  background: isActive ? "var(--color-accent-blue)" : "transparent",
                  color: isActive ? "white" : "var(--color-text-secondary)",
                }}>
                <span className="text-base w-5 text-center">{item.icon}</span>
                {item.label}
              </Link>
            );
          })}
        </nav>

        {/* User Footer */}
        <div className="p-4 border-t" style={{ borderColor: "var(--color-border-subtle)" }}>
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-full flex items-center justify-center text-xs font-bold text-white"
              style={{ background: "var(--color-accent-purple)" }}>
              {user?.full_name?.charAt(0)?.toUpperCase()}
            </div>
            <div className="flex-1 min-w-0">
              <div className="text-xs font-medium truncate" style={{ color: "var(--color-text-primary)" }}>
                {user?.full_name}
              </div>
              <div className="text-[10px] uppercase font-semibold" style={{ color: "var(--color-text-muted)" }}>
                {user?.role?.replace("_", " ")}
              </div>
            </div>
            <button onClick={logout} className="text-xs px-2 py-1 rounded"
              style={{ color: "var(--color-accent-red)" }} title="Logout">
              ✕
            </button>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto p-6 page-enter">
        {children}
      </main>
    </div>
  );
}

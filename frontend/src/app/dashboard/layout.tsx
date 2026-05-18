"use client";

import { useEffect, useState, useRef } from "react";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import { useAuth } from "@/lib/auth";

const navGroups = [
  {
    label: "Trade",
    items: [
      { href: "/dashboard", label: "Overview" },
      { href: "/dashboard/orders", label: "OMS Blotter" },
      { href: "/dashboard/trades", label: "Positions" },
    ]
  },
  {
    label: "Analytics",
    items: [
      { href: "/dashboard/market", label: "Market Data" },
      { href: "/dashboard/pricing", label: "Pricing Lab" },
      { href: "/dashboard/greeks", label: "Greeks" },
      { href: "/dashboard/strategy", label: "Strategy Builder" },
      { href: "/dashboard/backtest", label: "Backtester" },
      { href: "/dashboard/scenario", label: "Scenarios" },
    ]
  },
  {
    label: "Risk",
    items: [
      { href: "/dashboard/risk", label: "Risk Engine" },
      { href: "/dashboard/portfolio-greeks", label: "Portfolio Greeks" },
      { href: "/dashboard/regulatory", label: "Basel III/IV" },
    ]
  }
];

export default function DashboardLayout({ children }: { children: React.ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const { user, isAuthenticated, isLoading, restore, logout } = useAuth();
  
  const [activeGroup, setActiveGroup] = useState<string | null>(null);
  const navRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    restore();
  }, [restore]);

  useEffect(() => {
    if (!isLoading && !isAuthenticated) {
      router.push("/");
    }
  }, [isLoading, isAuthenticated, router]);

  // Click outside to close dropdowns
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (navRef.current && !navRef.current.contains(event.target as Node)) {
        setActiveGroup(null);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  if (isLoading || !isAuthenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center" style={{ background: "var(--color-bg-primary)" }}>
        <div className="w-8 h-8 border-2 border-[var(--color-accent-blue)] border-t-transparent rounded-full animate-spin" />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col" style={{ background: "var(--color-bg-primary)" }}>
      {/* Top Navbar */}
      <header className="h-14 flex-shrink-0 border-b flex items-center px-6 sticky top-0 z-50"
        style={{ background: "var(--color-bg-card)", borderColor: "var(--color-border)" }}>
        
        {/* Logo */}
        <Link href="/dashboard" className="flex items-center gap-3 mr-8 cursor-pointer">
          <div className="w-8 h-8 rounded flex items-center justify-center font-extrabold text-sm text-white"
            style={{ background: "var(--color-accent-blue)" }}>Q</div>
          <div>
            <div className="font-bold text-sm tracking-tight" style={{ color: "var(--color-text-primary)" }}>Quant Engine</div>
          </div>
        </Link>

        {/* Navigation Groups */}
        <nav className="flex-1 flex items-center gap-6" ref={navRef}>
          {navGroups.map((group) => {
            const isGroupActive = group.items.some(item => 
              pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href))
            );
            
            return (
              <div key={group.label} className="relative">
                <button 
                  onClick={() => setActiveGroup(activeGroup === group.label ? null : group.label)}
                  className={`flex items-center gap-1 text-sm font-medium py-4 border-b-2 transition-colors ${
                    isGroupActive ? "border-[var(--color-accent-blue)] text-[var(--color-text-primary)]" : "border-transparent text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
                  }`}
                >
                  {group.label}
                  <svg className="w-3.5 h-3.5 opacity-70" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" /></svg>
                </button>
                
                {/* Dropdown */}
                {activeGroup === group.label && (
                  <div className="absolute top-[3.2rem] left-0 w-48 rounded shadow-xl border overflow-hidden animate-in fade-in slide-in-from-top-2"
                    style={{ background: "var(--color-bg-elevated)", borderColor: "var(--color-border)" }}>
                    {group.items.map(item => {
                      const isActive = pathname === item.href || (item.href !== "/dashboard" && pathname.startsWith(item.href));
                      return (
                        <Link key={item.href} href={item.href}
                          onClick={() => setActiveGroup(null)}
                          className="block px-4 py-2.5 text-sm transition-colors hover:bg-[var(--color-bg-hover)]"
                          style={{
                            color: isActive ? "var(--color-accent-blue)" : "var(--color-text-primary)",
                            fontWeight: isActive ? 600 : 400
                          }}>
                          {item.label}
                        </Link>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}
          
          {user?.role === "admin" && (
            <Link href="/dashboard/admin" 
              className={`text-sm font-medium py-4 border-b-2 transition-colors ${
                pathname.startsWith("/dashboard/admin") ? "border-[var(--color-accent-blue)] text-[var(--color-text-primary)]" : "border-transparent text-[var(--color-text-secondary)] hover:text-[var(--color-text-primary)]"
              }`}>
              Admin
            </Link>
          )}
        </nav>

        {/* User Actions */}
        <div className="flex items-center gap-4 border-l pl-4" style={{ borderColor: "var(--color-border)" }}>
          <div className="text-right hidden sm:block">
            <div className="text-xs font-semibold" style={{ color: "var(--color-text-primary)" }}>
              {user?.full_name}
            </div>
            <div className="text-[10px] uppercase font-bold tracking-wider" style={{ color: "var(--color-text-muted)" }}>
              {user?.role?.replace("_", " ")}
            </div>
          </div>
          <div className="w-8 h-8 rounded bg-[var(--color-bg-elevated)] border border-[var(--color-border)] flex items-center justify-center text-xs font-bold text-[var(--color-text-primary)]">
            {user?.full_name?.charAt(0)?.toUpperCase()}
          </div>
          <button onClick={logout} className="text-sm font-medium transition-colors hover:text-[var(--color-accent-red)]"
            style={{ color: "var(--color-text-secondary)" }}>
            Logout
          </button>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto bg-[var(--color-bg-primary)] p-6 page-enter">
        <div className="max-w-[1600px] mx-auto">
          {children}
        </div>
      </main>
    </div>
  );
}

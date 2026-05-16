"use client";

import { useState, useEffect, useCallback } from "react";
import { useAuth } from "@/lib/auth";
import { adminApi, UserData } from "@/lib/api";

export default function AdminPage() {
  const { accessToken, user } = useAuth();
  const [users, setUsers] = useState<UserData[]>([]);
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const loadData = useCallback(async () => {
    if (!accessToken) return;
    try {
      setLoading(true);
      const [uRes, mRes] = await Promise.all([
        adminApi.listUsers(accessToken),
        adminApi.systemMetrics(accessToken)
      ]);
      setUsers(uRes.users);
      setMetrics(mRes);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [accessToken]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  if (user?.role !== "admin") {
    return (
      <div className="p-8 text-center text-red-500 bg-red-500/10 rounded-lg">
        <h2 className="text-xl font-bold">Access Denied</h2>
        <p>You must be an administrator to view this page.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold" style={{ color: "var(--color-text-primary)" }}>Admin Panel</h1>
        <p className="text-sm mt-1" style={{ color: "var(--color-text-secondary)" }}>
          System metrics and user management.
        </p>
      </div>

      {metrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="card p-4">
            <div className="text-xs text-gray-400 uppercase">Total Users</div>
            <div className="text-2xl font-bold mt-1 text-white">{metrics.counts.users}</div>
          </div>
          <div className="card p-4">
            <div className="text-xs text-gray-400 uppercase">Total Portfolios</div>
            <div className="text-2xl font-bold mt-1 text-white">{metrics.counts.portfolios}</div>
          </div>
          <div className="card p-4">
            <div className="text-xs text-gray-400 uppercase">Open Trades</div>
            <div className="text-2xl font-bold mt-1 text-green-500">{metrics.counts.trades_open}</div>
          </div>
          <div className="card p-4">
            <div className="text-xs text-gray-400 uppercase">Total Orders</div>
            <div className="text-2xl font-bold mt-1 text-blue-500">{metrics.counts.orders_total}</div>
          </div>
        </div>
      )}

      <div className="card overflow-hidden">
        <div className="p-4 border-b border-white/10 font-medium text-white">Users</div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs uppercase bg-black/20 text-gray-400">
              <tr>
                <th className="px-4 py-3">Email</th>
                <th className="px-4 py-3">Name</th>
                <th className="px-4 py-3">Role</th>
                <th className="px-4 py-3">Joined</th>
                <th className="px-4 py-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {loading ? (
                <tr><td colSpan={5} className="px-4 py-4 text-center">Loading...</td></tr>
              ) : (
                users.map(u => (
                  <tr key={u.id} className="hover:bg-white/5">
                    <td className="px-4 py-3 font-medium text-white">{u.email}</td>
                    <td className="px-4 py-3 text-gray-300">{u.full_name}</td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] uppercase font-bold ${
                        u.role === "admin" ? "bg-red-500/20 text-red-500" :
                        u.role === "risk_manager" ? "bg-purple-500/20 text-purple-500" :
                        "bg-blue-500/20 text-blue-500"
                      }`}>{u.role}</span>
                    </td>
                    <td className="px-4 py-3 text-gray-400 text-xs">
                      {new Date(u.created_at).toLocaleDateString()}
                    </td>
                    <td className="px-4 py-3">
                      {u.is_active ? <span className="text-green-500 text-xs">Active</span> : <span className="text-red-500 text-xs">Inactive</span>}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

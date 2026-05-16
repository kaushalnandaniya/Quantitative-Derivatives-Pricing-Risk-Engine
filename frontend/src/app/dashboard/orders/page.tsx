"use client";

import { useState, useEffect, useCallback } from "react";
import { useAuth } from "@/lib/auth";
import { ordersApi, OrderData } from "@/lib/api";

export default function OrdersPage() {
  const { accessToken } = useAuth();
  const [orders, setOrders] = useState<OrderData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const loadOrders = useCallback(async () => {
    if (!accessToken) return;
    try {
      setLoading(true);
      const res = await ordersApi.list(accessToken);
      setOrders(res.orders);
    } catch (err: any) {
      setError(err.message || "Failed to load orders");
    } finally {
      setLoading(false);
    }
  }, [accessToken]);

  useEffect(() => {
    loadOrders();
  }, [loadOrders]);

  const handleCancel = async (id: string) => {
    if (!accessToken) return;
    try {
      await ordersApi.cancel(id, accessToken);
      loadOrders();
    } catch (err: any) {
      alert(err.message || "Failed to cancel order");
    }
  };

  const handleFill = async (id: string) => {
    if (!accessToken) return;
    try {
      await ordersApi.manualFill(id, accessToken);
      loadOrders();
    } catch (err: any) {
      alert(err.message || "Failed to fill order");
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-bold" style={{ color: "var(--color-text-primary)" }}>OMS Blotter</h1>
          <p className="text-sm mt-1" style={{ color: "var(--color-text-secondary)" }}>
            Order Management System — View and manage order lifecycle.
          </p>
        </div>
        <button onClick={loadOrders} className="btn-secondary text-sm">
          ↻ Refresh
        </button>
      </div>

      {error && (
        <div className="p-4 rounded-lg bg-red-500/10 border border-red-500/50 text-red-500">
          {error}
        </div>
      )}

      <div className="card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs uppercase border-b" style={{ color: "var(--color-text-muted)", borderColor: "var(--color-border-subtle)" }}>
              <tr>
                <th className="px-4 py-3">ID / Time</th>
                <th className="px-4 py-3">Side</th>
                <th className="px-4 py-3">Type</th>
                <th className="px-4 py-3">Details</th>
                <th className="px-4 py-3">Status</th>
                <th className="px-4 py-3">Filled / Qty</th>
                <th className="px-4 py-3">Avg Fill</th>
                <th className="px-4 py-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y" style={{ borderColor: "var(--color-border-subtle)" }}>
              {loading ? (
                <tr>
                  <td colSpan={8} className="px-4 py-8 text-center text-gray-500">Loading orders...</td>
                </tr>
              ) : orders.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-4 py-8 text-center text-gray-500">No orders found.</td>
                </tr>
              ) : (
                orders.map((o) => (
                  <tr key={o.id} className="hover:bg-white/5 transition-colors">
                    <td className="px-4 py-3">
                      <div className="font-mono text-xs" title={o.id}>{o.id.substring(0, 8)}...</div>
                      <div className="text-[10px] text-gray-500">{new Date(o.submitted_at).toLocaleString()}</div>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase ${
                        o.side === "buy" ? "bg-green-500/20 text-green-500" : "bg-red-500/20 text-red-500"
                      }`}>
                        {o.side}
                      </span>
                    </td>
                    <td className="px-4 py-3">
                      <div className="font-medium uppercase">{o.option_type}</div>
                      <div className="text-xs text-gray-500">{o.order_type}</div>
                    </td>
                    <td className="px-4 py-3">
                      <div>S: <span className="font-mono">{o.spot_price.toFixed(2)}</span></div>
                      <div>K: <span className="font-mono">{o.strike.toFixed(2)}</span></div>
                    </td>
                    <td className="px-4 py-3">
                      <span className={`px-2 py-0.5 rounded-full text-[10px] font-medium border ${
                        o.status === "filled" ? "bg-green-500/10 border-green-500/30 text-green-500" :
                        o.status === "rejected" || o.status === "cancelled" ? "bg-red-500/10 border-red-500/30 text-red-500" :
                        "bg-amber-500/10 border-amber-500/30 text-amber-500"
                      }`}>
                        {o.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 font-mono">
                      {o.filled_quantity} / {o.quantity}
                    </td>
                    <td className="px-4 py-3 font-mono">
                      {o.avg_fill_price ? o.avg_fill_price.toFixed(2) : "-"}
                    </td>
                    <td className="px-4 py-3 text-right space-x-2">
                      {["pending", "validated", "partial_fill", "submitted"].includes(o.status) && (
                        <>
                          <button onClick={() => handleFill(o.id)} className="text-xs text-green-500 hover:underline">Simulate Fill</button>
                          <button onClick={() => handleCancel(o.id)} className="text-xs text-red-500 hover:underline">Cancel</button>
                        </>
                      )}
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

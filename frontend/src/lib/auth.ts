/**
 * Auth Store — Zustand store for authentication state
 */

import { create } from "zustand";
import { authApi, type UserData, type LoginResponse } from "./api";

interface AuthState {
  user: UserData | null;
  accessToken: string | null;
  refreshToken: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;

  login: (email: string, password: string) => Promise<void>;
  sendOtp: (email: string) => Promise<void>;
  register: (data: { email: string; password: string; full_name: string; otp: string; role?: string }) => Promise<void>;
  forgotPassword: (email: string) => Promise<void>;
  resetPassword: (data: { email: string; otp: string; new_password: string }) => Promise<void>;
  logout: () => void;
  restore: () => Promise<void>;
}

export const useAuth = create<AuthState>((set) => ({
  user: null,
  accessToken: null,
  refreshToken: null,
  isAuthenticated: false,
  isLoading: true,

  login: async (email, password) => {
    const res: LoginResponse = await authApi.login(email, password);
    localStorage.setItem("access_token", res.access_token);
    localStorage.setItem("refresh_token", res.refresh_token);
    set({
      user: res.user,
      accessToken: res.access_token,
      refreshToken: res.refresh_token,
      isAuthenticated: true,
    });
  },

  sendOtp: async (email) => {
    await authApi.sendOtp(email);
  },

  register: async (data) => {
    await authApi.register(data);
  },

  forgotPassword: async (email) => {
    await authApi.forgotPassword(email);
  },

  resetPassword: async (data) => {
    await authApi.resetPassword(data);
  },

  logout: () => {
    localStorage.removeItem("access_token");
    localStorage.removeItem("refresh_token");
    set({
      user: null,
      accessToken: null,
      refreshToken: null,
      isAuthenticated: false,
    });
  },

  restore: async () => {
    const token = typeof window !== "undefined" ? localStorage.getItem("access_token") : null;
    const refresh = typeof window !== "undefined" ? localStorage.getItem("refresh_token") : null;

    if (!token) {
      set({ isLoading: false });
      return;
    }

    try {
      const user = await authApi.me(token);
      set({ user, accessToken: token, refreshToken: refresh, isAuthenticated: true, isLoading: false });
    } catch {
      // Try refresh
      if (refresh) {
        try {
          const res = await authApi.refresh(refresh);
          localStorage.setItem("access_token", res.access_token);
          const user = await authApi.me(res.access_token);
          set({ user, accessToken: res.access_token, refreshToken: refresh, isAuthenticated: true, isLoading: false });
          return;
        } catch {
          // Refresh failed
        }
      }
      localStorage.removeItem("access_token");
      localStorage.removeItem("refresh_token");
      set({ isLoading: false });
    }
  },
}));

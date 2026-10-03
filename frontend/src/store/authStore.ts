import { create } from 'zustand';
import { api } from '../api/client';

interface UserInfo {
  id: string;
  name: string;
  email: string;
}

interface AuthState {
  user: UserInfo | null;
  token: string | null;
  isLoading: boolean;
  login: (email: string, pass: string) => Promise<void>;
  logout: () => void;
  loadUser: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: localStorage.getItem('token'),
  isLoading: true,
  login: async (email, pass) => {
    try {
      const { data } = await api.auth.login(email, pass);
      localStorage.setItem('token', data.token);
      set({ token: data.token, user: data.user });
    } catch (error) {
      throw error;
    }
  },
  logout: () => {
    localStorage.removeItem('token');
    set({ user: null, token: null });
  },
  loadUser: async () => {
    const token = localStorage.getItem('token');
    if (!token) {
      set({ isLoading: false });
      return;
    }
    try {
      const { data } = await api.auth.me();
      set({ user: data.user, isLoading: false });
    } catch (error) {
      localStorage.removeItem('token');
      set({ user: null, token: null, isLoading: false });
    }
  },
}));

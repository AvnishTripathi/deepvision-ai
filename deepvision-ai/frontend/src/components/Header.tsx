"use client";

import { useState, useEffect } from "react";
import { Menu, Activity, Cpu, Sparkles, RefreshCw } from "lucide-react";
import { fetchSystemStatus } from "@/lib/api";
import { SystemStatusResponse } from "@/types";

interface HeaderProps {
  onToggleSidebar: () => void;
}

export default function Header({ onToggleSidebar }: HeaderProps) {
  const [status, setStatus] = useState<SystemStatusResponse>({
    status: "checking",
    is_ready: false,
  });
  const [isLoading, setIsLoading] = useState(false);

  const checkStatus = async () => {
    setIsLoading(true);
    try {
      const res = await fetchSystemStatus();
      setStatus(res);
    } catch {
      setStatus({ status: "offline", is_ready: false });
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    checkStatus();
    const interval = setInterval(checkStatus, 30000);
    return () => clearInterval(interval);
  }, []);

  const isOnline = status.status === "online" || status.is_ready;

  return (
    <header className="sticky top-0 z-30 flex h-20 items-center justify-between border-b border-slate-800/80 bg-dark-950/80 px-4 backdrop-blur-xl sm:px-8">
      {/* Left: Mobile Toggle & Breadcrumbs */}
      <div className="flex items-center gap-4">
        <button
          onClick={onToggleSidebar}
          aria-label="Toggle navigation"
          className="flex h-10 w-10 items-center justify-center rounded-xl border border-slate-800 bg-dark-850 text-slate-300 hover:border-slate-700 hover:text-white lg:hidden"
        >
          <Menu className="h-5 w-5" />
        </button>

        <div className="hidden sm:block">
          <h2 className="text-sm font-semibold tracking-wide text-white">
            DeepVision AI Studio
          </h2>
          <p className="text-xs text-slate-400">
            Real-Time Face Analytics & MOT Tracking Pipeline
          </p>
        </div>
      </div>

      {/* Right: Hardware & Online Indicator */}
      <div className="flex items-center gap-3">
        {/* Hardware Specs Badge */}
        <div className="hidden items-center gap-2 rounded-xl border border-slate-800/80 bg-dark-850/80 px-3 py-1.5 text-xs text-slate-300 md:flex">
          <Cpu className="h-4 w-4 text-brand-cyan" />
          <span>Device:</span>
          <span className="font-semibold text-white">
            {status.device || "CPU"}
          </span>
          <span className="text-slate-500">|</span>
          <span className="text-slate-400">Model:</span>
          <span className="font-semibold text-white">
            {status.model || "YuNet ONNX"}
          </span>
        </div>

        {/* Real-time Status Badge */}
        <div
          className={`flex items-center gap-2.5 rounded-xl border px-3.5 py-1.5 text-xs font-medium transition-all ${
            isOnline
              ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-400"
              : "border-rose-500/30 bg-rose-500/10 text-rose-400"
          }`}
        >
          <span className="relative flex h-2.5 w-2.5">
            {isOnline && (
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75"></span>
            )}
            <span
              className={`relative inline-flex h-2.5 w-2.5 rounded-full ${
                isOnline ? "bg-emerald-400" : "bg-rose-500"
              }`}
            ></span>
          </span>
          <span className="font-semibold">
            {isOnline ? "System Online" : "System Offline"}
          </span>
          <button
            onClick={checkStatus}
            disabled={isLoading}
            title="Refresh status"
            className="ml-1 text-slate-400 hover:text-white"
          >
            <RefreshCw
              className={`h-3 w-3 ${isLoading ? "animate-spin" : ""}`}
            />
          </button>
        </div>
      </div>
    </header>
  );
}

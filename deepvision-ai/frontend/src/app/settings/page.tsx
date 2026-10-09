"use client";

import { useState, useEffect } from "react";
import {
  Settings as SettingsIcon,
  Sliders,
  ShieldCheck,
  Check,
  Sun,
  Moon,
  Activity,
  Server,
  RefreshCw,
  Cpu,
} from "lucide-react";
import { useTheme } from "@/context/ThemeContext";
import { fetchSystemStatus, checkHealth } from "@/lib/api";
import { SystemStatusResponse } from "@/types";

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();

  // Inference preferences
  const [defaultThreshold, setDefaultThreshold] = useState("0.50");
  const [detectionInterval, setDetectionInterval] = useState("1");
  const [analysisInterval, setAnalysisInterval] = useState("5");
  const [enablePose, setEnablePose] = useState(true);
  const [enableQuality, setEnableQuality] = useState(true);

  // Connection diagnostics
  const [apiStatus, setApiStatus] = useState<"checking" | "online" | "offline">("checking");
  const [pingLatency, setPingLatency] = useState<number | null>(null);
  const [systemInfo, setSystemInfo] = useState<SystemStatusResponse | null>(null);
  const [isTesting, setIsTesting] = useState(false);
  const [saved, setSaved] = useState(false);

  // Load saved preferences from localStorage
  useEffect(() => {
    try {
      const savedThresh = localStorage.getItem("dv_pref_threshold");
      if (savedThresh) setDefaultThreshold(savedThresh);

      const savedDetInt = localStorage.getItem("dv_pref_det_interval");
      if (savedDetInt) setDetectionInterval(savedDetInt);

      const savedAnInt = localStorage.getItem("dv_pref_an_interval");
      if (savedAnInt) setAnalysisInterval(savedAnInt);

      const savedPose = localStorage.getItem("dv_pref_enable_pose");
      if (savedPose !== null) setEnablePose(savedPose === "true");

      const savedQual = localStorage.getItem("dv_pref_enable_quality");
      if (savedQual !== null) setEnableQuality(savedQual === "true");
    } catch {
      // Safe fallback
    }

    testConnection();
  }, []);

  const testConnection = async () => {
    setIsTesting(true);
    const start = performance.now();
    try {
      const isHealthy = await checkHealth();
      const end = performance.now();
      setPingLatency(Math.round(end - start));
      if (isHealthy) {
        setApiStatus("online");
        const sys = await fetchSystemStatus();
        setSystemInfo(sys);
      } else {
        setApiStatus("offline");
      }
    } catch {
      setApiStatus("offline");
      setPingLatency(null);
    } finally {
      setIsTesting(false);
    }
  };

  const handleSave = () => {
    try {
      localStorage.setItem("dv_pref_threshold", defaultThreshold);
      localStorage.setItem("dv_pref_det_interval", detectionInterval);
      localStorage.setItem("dv_pref_an_interval", analysisInterval);
      localStorage.setItem("dv_pref_enable_pose", String(enablePose));
      localStorage.setItem("dv_pref_enable_quality", String(enableQuality));
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch {
      // Fallback
    }
  };

  return (
    <div className="space-y-8 max-w-4xl py-2">
      <div>
        <h1 className="text-2xl font-black tracking-tight text-white sm:text-3xl flex items-center gap-3">
          <SettingsIcon className="h-7 w-7 text-brand-cyan" />
          <span>System & Engine Preferences</span>
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Configure application theme, inference parameters, and inspect backend connection diagnostics.
        </p>
      </div>

      {/* 1. Theme Configuration */}
      <div className="rounded-2xl border border-slate-800 bg-dark-900/90 p-6 shadow-card space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-3">
          <Sun className="h-5 w-5 text-amber-400" />
          <span>Interface Theme</span>
        </h3>
        <p className="text-xs text-slate-400">
          Choose between Dark AI Workspace and High-Contrast Light theme. Preference is persisted locally.
        </p>

        <div className="flex flex-wrap gap-3 pt-1">
          <button
            onClick={() => setTheme("dark")}
            className={`flex items-center gap-2.5 rounded-xl border px-4 py-2.5 text-xs font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-cyan ${
              theme === "dark"
                ? "border-brand-cyan bg-brand-cyan/15 text-brand-cyan shadow-glow"
                : "border-slate-800 bg-dark-850 text-slate-400 hover:text-white"
            }`}
          >
            <Moon className="h-4 w-4" />
            <span>Dark AI Studio (Default)</span>
          </button>

          <button
            onClick={() => setTheme("light")}
            className={`flex items-center gap-2.5 rounded-xl border px-4 py-2.5 text-xs font-semibold transition-all focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-cyan ${
              theme === "light"
                ? "border-amber-400 bg-amber-400/15 text-amber-300"
                : "border-slate-800 bg-dark-850 text-slate-400 hover:text-white"
            }`}
          >
            <Sun className="h-4 w-4" />
            <span>Light Mode</span>
          </button>
        </div>
      </div>

      {/* 2. API Connection Status Diagnostics */}
      <div className="rounded-2xl border border-slate-800 bg-dark-900/90 p-6 shadow-card space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <h3 className="text-base font-bold text-white flex items-center gap-2">
            <Server className="h-5 w-5 text-brand-blue" />
            <span>Backend Connection Status</span>
          </h3>
          <button
            onClick={testConnection}
            disabled={isTesting}
            className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-dark-850 px-3 py-1 text-xs text-slate-300 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-cyan"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${isTesting ? "animate-spin" : ""}`} />
            <span>Test Connection</span>
          </button>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div className="rounded-xl border border-slate-800/80 bg-dark-850/60 p-3.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
              API Status
            </span>
            <div className="mt-2 flex items-center gap-2">
              <span
                className={`h-2.5 w-2.5 rounded-full ${
                  apiStatus === "online"
                    ? "bg-emerald-400 shadow-glow"
                    : apiStatus === "checking"
                    ? "bg-amber-400"
                    : "bg-rose-500"
                }`}
              />
              <span className="text-sm font-bold text-white capitalize">
                {apiStatus}
              </span>
            </div>
          </div>

          <div className="rounded-xl border border-slate-800/80 bg-dark-850/60 p-3.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
              Ping Latency
            </span>
            <div className="mt-2 text-sm font-mono font-bold text-white">
              {pingLatency !== null ? `${pingLatency} ms` : "—"}
            </div>
          </div>

          <div className="rounded-xl border border-slate-800/80 bg-dark-850/60 p-3.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">
              Inference Hardware
            </span>
            <div className="mt-2 text-sm font-bold text-white flex items-center gap-1.5">
              <Cpu className="h-4 w-4 text-brand-cyan" />
              <span>{systemInfo?.device || "CPU"}</span>
            </div>
          </div>
        </div>

        <p className="text-[11px] text-slate-500">
          Target API Endpoint: <code className="font-mono text-slate-400">/api/v1/health</code> & <code className="font-mono text-slate-400">/api/v1/ai/status</code>. Credentials and private tokens are never exposed.
        </p>
      </div>

      {/* 3. Inference Parameters */}
      <div className="rounded-2xl border border-slate-800 bg-dark-900/90 p-6 shadow-card space-y-6">
        <h3 className="text-base font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-3">
          <Sliders className="h-5 w-5 text-brand-cyan" />
          <span>Inference Parameters</span>
        </h3>

        <div className="space-y-4">
          <div>
            <label className="text-xs font-semibold text-slate-300">
              Default Detection Confidence Threshold
            </label>
            <p className="text-[11px] text-slate-500 mb-2">
              Minimum probability required for candidate face proposals (0.10 to 0.95).
            </p>
            <input
              type="number"
              step="0.05"
              min="0.1"
              max="0.95"
              value={defaultThreshold}
              onChange={(e) => setDefaultThreshold(e.target.value)}
              className="w-full sm:w-48 rounded-xl border border-slate-800 bg-dark-850 px-3.5 py-2 text-xs font-mono text-white focus:border-brand-cyan focus:outline-none focus-visible:ring-1 focus-visible:ring-brand-cyan"
            />
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-300">
              Video Detection Interval (Frames)
            </label>
            <p className="text-[11px] text-slate-500 mb-2">
              Runs face detector every N frames. Intermediate frames coast with Kalman predictions.
            </p>
            <input
              type="number"
              min="1"
              max="10"
              value={detectionInterval}
              onChange={(e) => setDetectionInterval(e.target.value)}
              className="w-full sm:w-48 rounded-xl border border-slate-800 bg-dark-850 px-3.5 py-2 text-xs font-mono text-white focus:border-brand-cyan focus:outline-none focus-visible:ring-1 focus-visible:ring-brand-cyan"
            />
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-300">
              Video Analysis Interval (Frames)
            </label>
            <p className="text-[11px] text-slate-500 mb-2">
              Runs 3D head pose and quality score analysis every N frames per active track.
            </p>
            <input
              type="number"
              min="1"
              max="30"
              value={analysisInterval}
              onChange={(e) => setAnalysisInterval(e.target.value)}
              className="w-full sm:w-48 rounded-xl border border-slate-800 bg-dark-850 px-3.5 py-2 text-xs font-mono text-white focus:border-brand-cyan focus:outline-none focus-visible:ring-1 focus-visible:ring-brand-cyan"
            />
          </div>

          <div className="pt-2 border-t border-slate-800/80 space-y-3">
            <label className="text-xs font-semibold text-slate-300 block">
              Analysis Preferences
            </label>
            <div className="flex flex-col gap-2.5 sm:flex-row sm:gap-6">
              <label className="flex items-center gap-2.5 text-xs text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enablePose}
                  onChange={(e) => setEnablePose(e.target.checked)}
                  className="rounded border-slate-700 bg-dark-850 text-brand-cyan focus:ring-brand-cyan"
                />
                <span>3D Head Pose Estimation (Yaw/Pitch/Roll)</span>
              </label>

              <label className="flex items-center gap-2.5 text-xs text-slate-300 cursor-pointer">
                <input
                  type="checkbox"
                  checked={enableQuality}
                  onChange={(e) => setEnableQuality(e.target.checked)}
                  className="rounded border-slate-700 bg-dark-850 text-brand-cyan focus:ring-brand-cyan"
                />
                <span>Multi-metric Face Quality Scoring</span>
              </label>
            </div>
          </div>
        </div>

        <div className="border-t border-slate-800 pt-4 flex items-center justify-between">
          <button
            onClick={handleSave}
            className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-brand-cyan to-brand-blue px-6 py-2.5 text-xs font-bold text-dark-950 shadow-glow hover:opacity-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-cyan"
          >
            {saved ? (
              <>
                <Check className="h-4 w-4" />
                <span>Preferences Saved</span>
              </>
            ) : (
              <span>Save Configuration</span>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}


"use client";

import { useState, useEffect } from "react";
import {
  BarChart3,
  Cpu,
  Layers,
  Zap,
  Activity,
  ShieldCheck,
  CheckCircle2,
} from "lucide-react";
import { fetchSystemStatus } from "@/lib/api";
import { SystemStatusResponse } from "@/types";

export default function AnalyticsPage() {
  const [systemStatus, setSystemStatus] = useState<SystemStatusResponse>({
    status: "checking",
  });

  const [stats, setStats] = useState({
    images: 0,
    videos: 0,
    faces: 0,
  });

  useEffect(() => {
    fetchSystemStatus().then(setSystemStatus);
    try {
      setStats({
        images: parseInt(sessionStorage.getItem("dv_images_analyzed") || "0", 10),
        videos: parseInt(sessionStorage.getItem("dv_videos_processed") || "0", 10),
        faces: parseInt(sessionStorage.getItem("dv_faces_detected") || "0", 10),
      });
    } catch {}
  }, []);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-black tracking-tight text-white sm:text-3xl">
          System Analytics & Telemetry
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Neural model operational metrics, runtime telemetry, and pipeline throughput statistics.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
        <div className="glass-panel rounded-2xl p-6">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-semibold uppercase tracking-wider">
              AI Inference Device
            </span>
            <Cpu className="h-5 w-5 text-brand-cyan" />
          </div>
          <p className="mt-4 text-2xl font-extrabold text-white">
            {systemStatus.device || "CPU"}
          </p>
          <p className="mt-1 text-xs text-slate-400">
            OpenCV DNN SIMD acceleration
          </p>
        </div>

        <div className="glass-panel rounded-2xl p-6">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-semibold uppercase tracking-wider">
              Active Neural Engine
            </span>
            <Zap className="h-5 w-5 text-purple-400" />
          </div>
          <p className="mt-4 text-2xl font-extrabold text-white">
            {systemStatus.model || "YuNet ONNX"}
          </p>
          <p className="mt-1 text-xs text-slate-400">
            232 KB multi-scale detection model
          </p>
        </div>

        <div className="glass-panel rounded-2xl p-6">
          <div className="flex items-center justify-between text-slate-400">
            <span className="text-xs font-semibold uppercase tracking-wider">
              MOT Tracking Architecture
            </span>
            <Activity className="h-5 w-5 text-emerald-400" />
          </div>
          <p className="mt-4 text-2xl font-extrabold text-white">
            ByteTrack
          </p>
          <p className="mt-1 text-xs text-slate-400">
            8-state constant velocity Kalman filter
          </p>
        </div>
      </div>

      {/* Latency Benchmarks */}
      <div className="rounded-2xl border border-slate-800 bg-dark-900/90 p-6 shadow-card space-y-4">
        <h3 className="text-base font-bold text-white flex items-center gap-2">
          <BarChart3 className="h-5 w-5 text-brand-cyan" />
          <span>Stage Latency Benchmarks (Measured on x86 CPU)</span>
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="rounded-xl border border-slate-800/80 bg-dark-850/50 p-4">
            <span className="text-xs text-slate-400">Face Detection (YuNet)</span>
            <p className="mt-2 text-xl font-mono font-bold text-white">~16–25 ms</p>
            <p className="mt-1 text-[11px] text-slate-500">Forward pass inference</p>
          </div>

          <div className="rounded-xl border border-slate-800/80 bg-dark-850/50 p-4">
            <span className="text-xs text-slate-400">3D Pose Estimation</span>
            <p className="mt-2 text-xl font-mono font-bold text-white">~0.4–0.8 ms</p>
            <p className="mt-1 text-[11px] text-slate-500">Perspective-n-Point</p>
          </div>

          <div className="rounded-xl border border-slate-800/80 bg-dark-850/50 p-4">
            <span className="text-xs text-slate-400">Quality & Clarity Scoring</span>
            <p className="mt-2 text-xl font-mono font-bold text-white">~1.2–2.5 ms</p>
            <p className="mt-1 text-[11px] text-slate-500">Laplacian spatial frequency</p>
          </div>

          <div className="rounded-xl border border-slate-800/80 bg-dark-850/50 p-4">
            <span className="text-xs text-slate-400">ByteTrack Tracking</span>
            <p className="mt-2 text-xl font-mono font-bold text-white">~1.5–2.5 ms</p>
            <p className="mt-1 text-[11px] text-slate-500">Two-stage BYTE association</p>
          </div>
        </div>
      </div>
    </div>
  );
}

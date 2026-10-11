"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import {
  ScanFace,
  Video,
  Activity,
  Layers,
  ShieldCheck,
  Zap,
  Eye,
  Crosshair,
  Gauge,
  Lock,
  ArrowRight,
  Sparkles,
  CheckCircle2,
} from "lucide-react";
import { fetchSystemStatus } from "@/lib/api";
import { SystemStatusResponse } from "@/types";

export default function LandingPage() {
  const [systemStatus, setSystemStatus] = useState<SystemStatusResponse>({
    status: "checking",
    is_ready: false,
  });

  const [sessionStats, setSessionStats] = useState({
    imagesAnalyzed: 0,
    videosProcessed: 0,
    facesDetected: 0,
  });

  useEffect(() => {
    fetchSystemStatus().then(setSystemStatus).catch(() => {});

    try {
      const imgCount = parseInt(sessionStorage.getItem("dv_images_analyzed") || "0", 10);
      const vidCount = parseInt(sessionStorage.getItem("dv_videos_processed") || "0", 10);
      const faceCount = parseInt(sessionStorage.getItem("dv_faces_detected") || "0", 10);
      setSessionStats({
        imagesAnalyzed: imgCount,
        videosProcessed: vidCount,
        facesDetected: faceCount,
      });
    } catch {
      // Safe fallback
    }
  }, []);

  return (
    <div className="space-y-16 py-4">
      {/* Hero Section */}
      <section className="relative overflow-hidden rounded-3xl border border-slate-800/80 bg-gradient-to-b from-dark-900 via-dark-850 to-dark-900 p-8 sm:p-14 shadow-card">
        <div className="absolute -right-20 -top-20 h-80 w-80 rounded-full bg-brand-cyan/10 blur-3xl pointer-events-none" />
        <div className="absolute -left-20 -bottom-20 h-80 w-80 rounded-full bg-brand-violet/10 blur-3xl pointer-events-none" />

        <div className="relative z-10 flex flex-col lg:flex-row items-center justify-between gap-10">
          <div className="max-w-2xl">
            <div className="inline-flex items-center gap-2 rounded-xl border border-brand-cyan/30 bg-brand-cyan/10 px-3.5 py-1.5 text-xs font-semibold text-brand-cyan">
              <Zap className="h-4 w-4" />
              <span>DeepVision AI Platform</span>
            </div>

            <h1 className="mt-5 text-4xl font-extrabold tracking-tight text-white sm:text-6xl">
              DeepVision AI
            </h1>
            <p className="mt-2 text-xl font-semibold tracking-wide text-brand-cyan sm:text-2xl">
              See. Detect. Track. Analyze. Understand.
            </p>

            <p className="mt-5 text-base text-slate-300 sm:text-lg leading-relaxed">
              A professional AI-powered computer vision platform for real-time face detection, video tracking, facial analysis and intelligent visual insights.
            </p>

            <div className="mt-8 flex flex-wrap items-center gap-4">
              <Link
                href="/image"
                className="inline-flex items-center gap-2.5 rounded-xl bg-gradient-to-r from-brand-cyan to-brand-blue px-7 py-3.5 text-base font-bold text-dark-950 shadow-glow transition-all hover:opacity-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-cyan"
              >
                <span>Start Analyzing</span>
                <ArrowRight className="h-5 w-5" />
              </Link>
              <Link
                href="/video"
                className="inline-flex items-center gap-2.5 rounded-xl border border-slate-700 bg-dark-800/90 px-6 py-3.5 text-base font-semibold text-slate-200 transition-all hover:border-slate-600 hover:bg-dark-700 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-cyan"
              >
                <Video className="h-5 w-5" />
                <span>Video Tracking Studio</span>
              </Link>
            </div>
          </div>

          {/* Hero Logo Emblem */}
          <div className="relative shrink-0 flex items-center justify-center">
            <div className="absolute inset-0 rounded-3xl bg-brand-cyan/20 blur-2xl" />
            <div className="relative h-64 w-64 sm:h-72 sm:w-72 overflow-hidden rounded-3xl border border-brand-cyan/40 bg-dark-950/90 p-4 shadow-2xl backdrop-blur-xl">
              {/* eslint-disable-next-line @next/next/no-img-element */}
              <img
                src="/logo.png"
                alt="DeepVision AI Official Logo"
                className="h-full w-full object-contain rounded-2xl"
              />
            </div>
          </div>
        </div>
      </section>

      {/* Feature Sections (6 Pillars) */}
      <section className="space-y-6">
        <div>
          <h2 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
            Core Computer Vision Capabilities
          </h2>
          <p className="mt-1 text-sm text-slate-400">
            Engineered with modern neural networks and geometric computer vision algorithms.
          </p>
        </div>

        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 lg:grid-cols-3">
          {/* 1. Image Detection */}
          <div className="glass-panel glass-panel-hover rounded-2xl p-6 transition-all">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-brand-cyan/10 text-brand-cyan">
              <ScanFace className="h-6 w-6" />
            </div>
            <h3 className="mt-4 text-lg font-bold text-white">
              Image Detection
            </h3>
            <p className="mt-2 text-sm text-slate-400 leading-relaxed">
              Ultra-fast YuNet ONNX face detection delivering high recall across multi-scale faces, severe angles, low-light environments, and dense crowd scenes.
            </p>
            <div className="mt-4 pt-4 border-t border-slate-800/80 flex items-center gap-2 text-xs text-brand-cyan">
              <span>Sub-millisecond inference</span>
            </div>
          </div>

          {/* 2. Video Tracking */}
          <div className="glass-panel glass-panel-hover rounded-2xl p-6 transition-all">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-brand-blue/10 text-brand-blue">
              <Video className="h-6 w-6" />
            </div>
            <h3 className="mt-4 text-lg font-bold text-white">
              Video Tracking
            </h3>
            <p className="mt-2 text-sm text-slate-400 leading-relaxed">
              Multi-object face tracking powered by ByteTrack with 2D Kalman motion forecasting, smooth trajectory association across frame drops, and session track IDs.
            </p>
            <div className="mt-4 pt-4 border-t border-slate-800/80 flex items-center gap-2 text-xs text-brand-blue">
              <span>Zero biometric re-ID database</span>
            </div>
          </div>

          {/* 3. Face Analysis */}
          <div className="glass-panel glass-panel-hover rounded-2xl p-6 transition-all">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-purple-500/10 text-purple-400">
              <Crosshair className="h-6 w-6" />
            </div>
            <h3 className="mt-4 text-lg font-bold text-white">
              Face Analysis
            </h3>
            <p className="mt-2 text-sm text-slate-400 leading-relaxed">
              Geometric subpixel landmark alignment (eyes, nose, mouth corners) paired with 3D head pose estimation solving Yaw, Pitch, and Roll angles via perspective-n-point.
            </p>
            <div className="mt-4 pt-4 border-t border-slate-800/80 flex items-center gap-2 text-xs text-purple-400">
              <span>PnP 3D orientation</span>
            </div>
          </div>

          {/* 4. Quality Intelligence */}
          <div className="glass-panel glass-panel-hover rounded-2xl p-6 transition-all">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-amber-500/10 text-amber-400">
              <Gauge className="h-6 w-6" />
            </div>
            <h3 className="mt-4 text-lg font-bold text-white">
              Quality Intelligence
            </h3>
            <p className="mt-2 text-sm text-slate-400 leading-relaxed">
              Comprehensive optical diagnostics assessing Laplacian blur variance, photopic illumination, dynamic range contrast, and an objective 0–100 composite quality score.
            </p>
            <div className="mt-4 pt-4 border-t border-slate-800/80 flex items-center gap-2 text-xs text-amber-400">
              <span>Blur, lighting, & contrast</span>
            </div>
          </div>

          {/* 5. Real-Time Vision */}
          <div className="glass-panel glass-panel-hover rounded-2xl p-6 transition-all">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-emerald-500/10 text-emerald-400">
              <Eye className="h-6 w-6" />
            </div>
            <h3 className="mt-4 text-lg font-bold text-white">
              Real-Time Vision
            </h3>
            <p className="mt-2 text-sm text-slate-400 leading-relaxed">
              High-throughput frame ingestion with decoupled detection intervals and Kalman track prediction for responsive edge and server deployments.
            </p>
            <div className="mt-4 pt-4 border-t border-slate-800/80 flex items-center gap-2 text-xs text-emerald-400">
              <span>Adaptive interval skipping</span>
            </div>
          </div>

          {/* 6. Secure AI Processing */}
          <div className="glass-panel glass-panel-hover rounded-2xl p-6 transition-all">
            <div className="flex h-12 w-12 items-center justify-center rounded-xl bg-rose-500/10 text-rose-400">
              <Lock className="h-6 w-6" />
            </div>
            <h3 className="mt-4 text-lg font-bold text-white">
              Secure AI Processing
            </h3>
            <p className="mt-2 text-sm text-slate-400 leading-relaxed">
              Strict privacy by design. Temporary track IDs are session-scoped only. No biometric identity databases, face recognition profiles, or persistent facial storage.
            </p>
            <div className="mt-4 pt-4 border-t border-slate-800/80 flex items-center gap-2 text-xs text-rose-400">
              <span>Privacy-first architecture</span>
            </div>
          </div>
        </div>
      </section>

      {/* Session Telemetry (Verified Live Numbers Only) */}
      <section className="rounded-2xl border border-slate-800/80 bg-dark-900/70 p-6 sm:p-8">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-bold text-white flex items-center gap-2">
              <Activity className="h-5 w-5 text-brand-cyan" />
              <span>Current Session Activity</span>
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Live metrics recorded from this browser session. Zero mock counts.
            </p>
          </div>
        </div>

        <div className="mt-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div className="rounded-xl border border-slate-800 bg-dark-850 p-4">
            <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Images Analyzed
            </div>
            <div className="mt-2 text-2xl font-extrabold text-white">
              {sessionStats.imagesAnalyzed}
            </div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-dark-850 p-4">
            <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Videos Processed
            </div>
            <div className="mt-2 text-2xl font-extrabold text-white">
              {sessionStats.videosProcessed}
            </div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-dark-850 p-4">
            <div className="text-xs font-semibold uppercase tracking-wider text-slate-400">
              Cumulative Faces
            </div>
            <div className="mt-2 text-2xl font-extrabold text-white">
              {sessionStats.facesDetected}
            </div>
          </div>
        </div>
      </section>

      {/* Verified Pipeline Stack */}
      <section className="rounded-2xl border border-slate-800/80 bg-dark-900/50 p-6">
        <h3 className="text-sm font-bold uppercase tracking-wider text-slate-400 flex items-center gap-2">
          <ShieldCheck className="h-4 w-4 text-emerald-400" />
          <span>Operational Capabilities (Steps 1–5 Verified)</span>
        </h3>
        <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs text-slate-300">
          <div className="flex items-center gap-2 rounded-lg bg-dark-850/60 p-2.5">
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
            <span>YuNet ONNX Face Detector</span>
          </div>
          <div className="flex items-center gap-2 rounded-lg bg-dark-850/60 p-2.5">
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
            <span>5 Subpixel Landmarks</span>
          </div>
          <div className="flex items-center gap-2 rounded-lg bg-dark-850/60 p-2.5">
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
            <span>3D Head Pose (solvePnP)</span>
          </div>
          <div className="flex items-center gap-2 rounded-lg bg-dark-850/60 p-2.5">
            <CheckCircle2 className="h-3.5 w-3.5 text-emerald-400 shrink-0" />
            <span>ByteTrack Multi-Face MOT</span>
          </div>
        </div>
      </section>
    </div>
  );
}


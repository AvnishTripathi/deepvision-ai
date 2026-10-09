"use client";

import React from "react";
import { VideoJob } from "@/lib/api/types";
import { Loader2, Film, Activity, StopCircle, RefreshCw } from "lucide-react";

interface VideoProcessingProgressCardProps {
  job: VideoJob;
  onCancel: () => void;
}

export default function VideoProcessingProgressCard({
  job,
  onCancel,
}: VideoProcessingProgressCardProps) {
  const percent = Math.min(100, Math.max(0, Math.round(job.progress)));

  return (
    <div className="rounded-2xl border border-slate-800 bg-dark-900/90 p-6 shadow-card space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-blue/10 text-brand-blue">
            <Loader2 className="h-5 w-5 animate-spin" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white">
              Processing Video
            </h3>
            <p className="text-xs text-slate-400">
              Job ID: <code className="font-mono text-slate-300">{job.job_id.slice(0, 8)}...</code>
            </p>
          </div>
        </div>

        <button
          onClick={onCancel}
          className="flex items-center gap-1.5 rounded-xl border border-rose-500/30 bg-rose-500/10 px-3.5 py-1.5 text-xs font-semibold text-rose-400 transition-colors hover:bg-rose-500/20 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-rose-500"
        >
          <StopCircle className="h-3.5 w-3.5" />
          <span>Cancel Job</span>
        </button>
      </div>

      {/* Progress Bar & Percentage */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold text-slate-300">Progress</span>
          <span className="font-mono font-bold text-brand-cyan">{percent}%</span>
        </div>

        <div className="h-3 w-full overflow-hidden rounded-full bg-dark-800 p-0.5">
          <div
            className="h-full rounded-full bg-gradient-to-r from-brand-blue to-brand-cyan transition-all duration-300 ease-out shadow-glow"
            style={{ width: `${percent}%` }}
          />
        </div>
      </div>

      {/* Real Backend Decode Metrics Grid (Item 11) */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        {/* Frames */}
        <div className="rounded-xl border border-slate-800/80 bg-dark-850/60 p-3">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
            Frames
          </span>
          <p className="mt-1 text-sm font-mono font-bold text-white">
            {job.processed_frames.toLocaleString()} /{" "}
            {job.total_frames > 0 ? job.total_frames.toLocaleString() : "..."}
          </p>
        </div>

        {/* Status */}
        <div className="rounded-xl border border-slate-800/80 bg-dark-850/60 p-3">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
            Status
          </span>
          <p className="mt-1 text-sm font-bold capitalize text-brand-cyan">
            {job.status}
          </p>
        </div>

        {/* Current Faces */}
        <div className="rounded-xl border border-slate-800/80 bg-dark-850/60 p-3">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
            Faces
          </span>
          <p className="mt-1 text-sm font-mono font-bold text-white">
            {job.current_faces !== undefined ? job.current_faces : "—"}
          </p>
        </div>

        {/* Tracks */}
        <div className="rounded-xl border border-slate-800/80 bg-dark-850/60 p-3">
          <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400">
            Tracks
          </span>
          <p className="mt-1 text-sm font-mono font-bold text-emerald-400">
            {job.active_tracks !== undefined
              ? job.active_tracks
              : job.metrics?.unique_tracks_count !== undefined
              ? job.metrics.unique_tracks_count
              : "—"}
          </p>
        </div>
      </div>
    </div>
  );
}


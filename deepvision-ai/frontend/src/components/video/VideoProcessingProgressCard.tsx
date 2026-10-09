"use client";

import { VideoJob } from "@/lib/api/types";
import { Loader2, Ban, Layers, Activity, Film } from "lucide-react";

interface VideoProcessingProgressCardProps {
  job: VideoJob;
  onCancel: () => void;
}

export default function VideoProcessingProgressCard({
  job,
  onCancel,
}: VideoProcessingProgressCardProps) {
  const progressPercent = Math.min(100, Math.max(0, Math.round(job.progress)));
  const processedFrames = job.processed_frames.toLocaleString();
  const totalFrames = job.total_frames ? job.total_frames.toLocaleString() : "Unknown";

  const currentFaces =
    job.metrics?.peak_simultaneous_faces ??
    job.metrics?.total_faces_detected ??
    0;
  const currentTracks = job.metrics?.unique_track_count ?? 0;

  return (
    <div className="rounded-2xl border border-slate-800 bg-dark-900/95 p-6 shadow-card space-y-6">
      {/* Title & Live Status Indicator */}
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
        <div className="flex items-center gap-3">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-brand-cyan/10 text-brand-cyan">
            <Film className="h-5 w-5" />
          </div>
          <div>
            <h3 className="text-base font-bold text-white">
              Processing Video
            </h3>
            <p className="text-xs text-slate-400 font-mono truncate max-w-xs sm:max-w-md">
              Job: {job.job_id}
            </p>
          </div>
        </div>

        {/* Live Status Badge */}
        <div className="flex items-center gap-2">
          <span className="flex h-2.5 w-2.5 relative">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-brand-cyan opacity-75"></span>
            <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-brand-cyan"></span>
          </span>
          <span className="text-xs font-bold uppercase tracking-wider text-brand-cyan capitalize">
            {job.status}
          </span>
        </div>
      </div>

      {/* Progress Bar with Exact Percentage */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-xs">
          <span className="font-semibold text-slate-300">
            Pipeline Execution Progress
          </span>
          <span className="font-mono text-base font-black text-white">
            {progressPercent}%
          </span>
        </div>

        <div className="h-3 w-full overflow-hidden rounded-full bg-dark-800 border border-slate-700/60 p-0.5">
          <div
            className="h-full rounded-full bg-gradient-to-r from-brand-blue via-brand-cyan to-emerald-400 transition-all duration-300"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>

      {/* Structured Telemetry Grid (Frames, Status, Faces, Tracks) */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 text-xs">
        {/* Frames */}
        <div className="rounded-xl border border-slate-800/80 bg-dark-850/60 p-3.5">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Frames
          </span>
          <p className="text-base font-mono font-bold text-white mt-1">
            {processedFrames} / {totalFrames}
          </p>
          <p className="text-[10px] text-slate-500 mt-0.5">Sequential decode</p>
        </div>

        {/* Status */}
        <div className="rounded-xl border border-slate-800/80 bg-dark-850/60 p-3.5">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Status
          </span>
          <p className="text-base font-bold text-brand-cyan mt-1 capitalize">
            {job.status}
          </p>
          <p className="text-[10px] text-slate-500 mt-0.5">Worker thread</p>
        </div>

        {/* Faces */}
        <div className="rounded-xl border border-slate-800/80 bg-dark-850/60 p-3.5">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Faces
          </span>
          <p className="text-base font-mono font-bold text-white mt-1">
            {currentFaces}
          </p>
          <p className="text-[10px] text-slate-500 mt-0.5">Detected faces</p>
        </div>

        {/* Tracks */}
        <div className="rounded-xl border border-slate-800/80 bg-dark-850/60 p-3.5">
          <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Tracks
          </span>
          <p className="text-base font-mono font-bold text-purple-300 mt-1">
            {currentTracks}
          </p>
          <p className="text-[10px] text-slate-500 mt-0.5">Active trajectories</p>
        </div>
      </div>

      {/* Cancel Action */}
      <div className="flex items-center justify-between border-t border-slate-800/80 pt-4">
        <div className="flex items-center gap-2 text-xs text-slate-400">
          <Loader2 className="h-4 w-4 animate-spin text-brand-cyan" />
          <span>Real-time frame-by-frame streaming active...</span>
        </div>

        <button
          onClick={onCancel}
          className="inline-flex items-center gap-1.5 rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-2 text-xs font-bold text-rose-400 hover:bg-rose-500/20 transition-colors"
        >
          <Ban className="h-3.5 w-3.5" />
          <span>Cancel Processing</span>
        </button>
      </div>
    </div>
  );
}

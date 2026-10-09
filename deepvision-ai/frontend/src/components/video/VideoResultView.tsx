"use client";

import React from "react";
import { VideoJob } from "@/lib/api/types";
import {
  Film,
  Download,
  Activity,
  CheckCircle2,
  Clock,
  Gauge,
  Layers,
  Sparkles,
} from "lucide-react";

interface VideoResultViewProps {
  job: VideoJob;
}

// Visually distinct palette for track tags (Item 13)
const TRACK_BADGE_COLORS = [
  "border-cyan-500/40 bg-cyan-500/15 text-cyan-300",
  "border-emerald-500/40 bg-emerald-500/15 text-emerald-300",
  "border-purple-500/40 bg-purple-500/15 text-purple-300",
  "border-amber-500/40 bg-amber-500/15 text-amber-300",
  "border-blue-500/40 bg-blue-500/15 text-blue-300",
  "border-pink-500/40 bg-pink-500/15 text-pink-300",
  "border-teal-500/40 bg-teal-500/15 text-teal-300",
  "border-rose-500/40 bg-rose-500/15 text-rose-300",
];

export default function VideoResultView({ job }: VideoResultViewProps) {
  const metrics = job.metrics;

  const getTrackColor = (trackId: number) => {
    return TRACK_BADGE_COLORS[Math.abs(trackId) % TRACK_BADGE_COLORS.length];
  };

  return (
    <div className="space-y-6">
      {/* Annotated Video Player (Item 12) */}
      <div className="rounded-2xl border border-slate-800 bg-dark-900/90 p-5 shadow-card space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800 pb-3">
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-5 w-5 text-emerald-400" />
            <h3 className="text-base font-bold text-white">
              Annotated Output Video
            </h3>
          </div>

          {job.annotated_video_url && (
            <a
              href={job.annotated_video_url}
              download={`deepvision_job_${job.job_id}.mp4`}
              className="inline-flex items-center gap-1.5 rounded-xl bg-brand-cyan/15 border border-brand-cyan/40 px-3.5 py-1.5 text-xs font-semibold text-brand-cyan hover:bg-brand-cyan/25 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-cyan"
            >
              <Download className="h-3.5 w-3.5" />
              <span>Download MP4</span>
            </a>
          )}
        </div>

        {job.annotated_video_url ? (
          <div className="overflow-hidden rounded-xl border border-slate-800 bg-black">
            <video
              src={job.annotated_video_url}
              controls
              autoPlay
              muted
              loop
              className="block w-full max-h-[460px] object-contain"
            />
          </div>
        ) : (
          <div className="p-8 text-center text-xs text-slate-400">
            No video playback URL available.
          </div>
        )}
      </div>

      {/* Video Summary Section (Item 12: strictly real backend metrics) */}
      {metrics && (
        <div className="rounded-2xl border border-slate-800 bg-dark-900/90 p-6 shadow-card space-y-4">
          <h4 className="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
            <Film className="h-4 w-4 text-brand-cyan" />
            <span>Video Summary</span>
          </h4>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <div className="rounded-xl border border-slate-800 bg-dark-850 p-3">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
                Duration
              </span>
              <p className="mt-1 text-sm font-mono font-bold text-white">
                {metrics.duration_seconds.toFixed(2)}s
              </p>
            </div>

            <div className="rounded-xl border border-slate-800 bg-dark-850 p-3">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
                FPS
              </span>
              <p className="mt-1 text-sm font-mono font-bold text-white">
                {metrics.fps.toFixed(1)}
              </p>
            </div>

            <div className="rounded-xl border border-slate-800 bg-dark-850 p-3">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
                Unique Tracks
              </span>
              <p className="mt-1 text-sm font-mono font-bold text-emerald-400">
                {metrics.unique_tracks_count}
              </p>
            </div>

            <div className="rounded-xl border border-slate-800 bg-dark-850 p-3">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
                Peak Faces
              </span>
              <p className="mt-1 text-sm font-mono font-bold text-brand-cyan">
                {metrics.peak_simultaneous_faces}
              </p>
            </div>

            <div className="rounded-xl border border-slate-800 bg-dark-850 p-3">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
                Processing Speed
              </span>
              <p className="mt-1 text-sm font-mono font-bold text-white">
                {metrics.average_processing_fps.toFixed(1)} FPS
              </p>
            </div>

            <div className="rounded-xl border border-slate-800 bg-dark-850 p-3">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
                Processing Time
              </span>
              <p className="mt-1 text-sm font-mono font-bold text-white">
                {metrics.processing_time_seconds.toFixed(2)}s
              </p>
            </div>

            <div className="rounded-xl border border-slate-800 bg-dark-850 p-3">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
                Total Frames
              </span>
              <p className="mt-1 text-sm font-mono font-bold text-white">
                {metrics.total_frames.toLocaleString()}
              </p>
            </div>

            <div className="rounded-xl border border-slate-800 bg-dark-850 p-3">
              <span className="text-[10px] text-slate-400 uppercase tracking-wider font-semibold">
                Processed
              </span>
              <p className="mt-1 text-sm font-mono font-bold text-white">
                {metrics.processed_frames.toLocaleString()}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Tracking Visualization & Track IDs (Item 13) */}
      {metrics && metrics.tracks_summary && metrics.tracks_summary.length > 0 && (
        <div className="rounded-2xl border border-slate-800 bg-dark-900/90 p-6 shadow-card space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h4 className="text-sm font-bold uppercase tracking-wider text-slate-300 flex items-center gap-2">
              <Activity className="h-4 w-4 text-emerald-400" />
              <span>Detected Visual Tracks</span>
            </h4>
            <span className="text-xs text-slate-400 font-normal">
              Temporary session IDs (Zero identity recognition)
            </span>
          </div>

          <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3 md:grid-cols-4">
            {metrics.tracks_summary.map((track) => (
              <div
                key={track.track_id}
                className={`rounded-xl border p-3 ${getTrackColor(track.track_id)}`}
              >
                <div className="font-bold text-sm">
                  Track #{track.track_id}
                </div>
                <div className="mt-1 text-[11px] opacity-80 space-y-0.5">
                  <p>Active: {track.frames_active} frames</p>
                  <p>Conf: {(track.avg_confidence * 100).toFixed(0)}%</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}


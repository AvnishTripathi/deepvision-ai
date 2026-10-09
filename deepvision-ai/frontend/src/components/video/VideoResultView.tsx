"use client";

import { VideoJob, VideoMetrics } from "@/lib/api/types";
import { buildVideoDownloadUrl } from "@/lib/api/video";
import {
  Download,
  Film,
  Activity,
  Layers,
  Clock,
  Gauge,
  ShieldCheck,
  CheckCircle2,
} from "lucide-react";

interface VideoResultViewProps {
  job: VideoJob;
}

// Generate distinct color themes for Track ID tags
const TRACK_COLOR_PALETTES = [
  "border-cyan-500/40 bg-cyan-500/10 text-cyan-300",
  "border-violet-500/40 bg-violet-500/10 text-violet-300",
  "border-emerald-500/40 bg-emerald-500/10 text-emerald-300",
  "border-amber-500/40 bg-amber-500/10 text-amber-300",
  "border-rose-500/40 bg-rose-500/10 text-rose-300",
  "border-blue-500/40 bg-blue-500/10 text-blue-300",
  "border-fuchsia-500/40 bg-fuchsia-500/10 text-fuchsia-300",
  "border-teal-500/40 bg-teal-500/10 text-teal-300",
];

export default function VideoResultView({ job }: VideoResultViewProps) {
  const metrics = job.metrics;
  const downloadUrl = buildVideoDownloadUrl(job.job_id);

  // Derive unique track list from 1 up to unique_track_count
  const uniqueTrackCount = metrics?.unique_track_count ?? 0;
  const trackIdList = Array.from({ length: uniqueTrackCount }, (_, i) => i + 1);

  return (
    <div className="space-y-6">
      {/* 1. Annotated Output Video Player */}
      <div className="overflow-hidden rounded-2xl border border-slate-800 bg-dark-950 shadow-card">
        <div className="flex items-center justify-between border-b border-slate-800/80 bg-dark-900/90 px-4 py-3">
          <div className="flex items-center gap-2 text-xs font-semibold text-white">
            <Film className="h-4 w-4 text-brand-cyan" />
            <span>Annotated ByteTrack Video Stream</span>
          </div>

          <a
            href={downloadUrl}
            download={`deepvision_annotated_${job.job_id.slice(0, 8)}.mp4`}
            className="inline-flex items-center gap-1.5 rounded-lg bg-emerald-500/15 border border-emerald-500/30 px-3 py-1 text-xs font-bold text-emerald-400 hover:bg-emerald-500/25 transition-colors"
          >
            <Download className="h-3.5 w-3.5" />
            <span>Download MP4</span>
          </a>
        </div>

        <video
          src={downloadUrl}
          controls
          className="block w-full max-h-[460px] bg-black"
        />
      </div>

      {/* 2. Tracking Visualization: Visually Distinct Track Labels (Step 13) */}
      <div className="rounded-2xl border border-slate-800 bg-dark-900/90 p-5 shadow-card space-y-4">
        <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
          <div className="flex items-center gap-2">
            <Layers className="h-4 w-4 text-brand-cyan" />
            <h4 className="text-sm font-bold text-white">
              Temporary Video Tracks ({uniqueTrackCount})
            </h4>
          </div>
          <span className="text-[11px] text-slate-400">
            Session-scoped spatio-temporal trajectories
          </span>
        </div>

        {trackIdList.length > 0 ? (
          <div className="flex flex-wrap gap-2.5">
            {trackIdList.map((trackNum, idx) => {
              const colorClass =
                TRACK_COLOR_PALETTES[idx % TRACK_COLOR_PALETTES.length];
              return (
                <div
                  key={trackNum}
                  className={`flex items-center gap-2 rounded-xl border px-3 py-2 text-xs font-semibold ${colorClass}`}
                >
                  <span className="flex h-2 w-2 rounded-full bg-current opacity-80" />
                  <span>Track #{trackNum}</span>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="text-xs text-slate-400">
            No continuous face tracks observed during this clip.
          </p>
        )}

        <div className="rounded-xl border border-slate-800/60 bg-dark-850/40 p-3 text-[11px] text-slate-400 flex items-start gap-2.5">
          <ShieldCheck className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
          <span>
            <strong>Zero Biometric Identity Rule</strong>: Track IDs denote temporal geometric continuity only. They do not represent human names, person identities, or cross-video database profiles.
          </span>
        </div>
      </div>

      {/* 3. Video Summary & Processing Telemetry (Step 12) */}
      {metrics && (
        <div className="rounded-2xl border border-slate-800 bg-dark-900/90 p-5 shadow-card space-y-4">
          <div className="flex items-center gap-2 border-b border-slate-800/80 pb-3">
            <Activity className="h-4 w-4 text-brand-cyan" />
            <h4 className="text-sm font-bold text-white">Video Summary & Telemetry</h4>
          </div>

          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 text-xs">
            {/* Unique Tracks */}
            <div className="rounded-xl border border-slate-800/80 bg-dark-850/50 p-3.5">
              <span className="text-slate-400 text-[11px]">Unique Tracks</span>
              <p className="text-2xl font-black text-white mt-1">
                {metrics.unique_track_count}
              </p>
              <p className="text-[10px] text-slate-500 mt-0.5">Session trajectories</p>
            </div>

            {/* Peak Simultaneous Faces */}
            <div className="rounded-xl border border-slate-800/80 bg-dark-850/50 p-3.5">
              <span className="text-slate-400 text-[11px]">Peak Simultaneous</span>
              <p className="text-2xl font-black text-brand-cyan mt-1">
                {metrics.peak_simultaneous_faces}
              </p>
              <p className="text-[10px] text-slate-500 mt-0.5">Max faces per frame</p>
            </div>

            {/* Average Processing FPS */}
            <div className="rounded-xl border border-slate-800/80 bg-dark-850/50 p-3.5">
              <span className="text-slate-400 text-[11px]">Processing FPS</span>
              <p className="text-2xl font-black text-purple-300 mt-1">
                {metrics.average_fps.toFixed(1)}
              </p>
              <p className="text-[10px] text-slate-500 mt-0.5">Throughput rate</p>
            </div>

            {/* Processing Time */}
            <div className="rounded-xl border border-slate-800/80 bg-dark-850/50 p-3.5">
              <span className="text-slate-400 text-[11px]">Processing Time</span>
              <p className="text-2xl font-black text-emerald-400 mt-1">
                {(metrics.total_processing_time / 1000).toFixed(2)}s
              </p>
              <p className="text-[10px] text-slate-500 mt-0.5">
                {metrics.average_frame_processing_time.toFixed(1)} ms/frame
              </p>
            </div>
          </div>

          {/* Additional Video Stats */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 text-xs border-t border-slate-800/80 pt-3">
            <div className="text-slate-400">
              <span>Total Processed Frames: </span>
              <strong className="text-white font-mono">{metrics.processed_frames} / {metrics.total_frames}</strong>
            </div>
            <div className="text-slate-400">
              <span>Total Detections: </span>
              <strong className="text-white font-mono">{metrics.total_faces_detected ?? "N/A"}</strong>
            </div>
            <div className="text-slate-400">
              <span>Track Lost Events: </span>
              <strong className="text-white font-mono">{metrics.track_lost_count}</strong>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

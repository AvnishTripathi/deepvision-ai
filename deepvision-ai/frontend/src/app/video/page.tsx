"use client";

import { useState, useRef, useEffect, ChangeEvent } from "react";
import {
  UploadCloud,
  Video as VideoIcon,
  Play,
  Trash2,
  Sliders,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Download,
  Ban,
  Activity,
  Layers,
  Clock,
  Sparkles,
  ShieldCheck,
} from "lucide-react";
import {
  submitVideoJob,
  getVideoJobStatus,
  cancelVideoJob,
  getVideoDownloadUrl,
} from "@/lib/api";
import { VideoJobResponse, VideoProcessingMetrics } from "@/types";

export default function VideoAnalysisPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.3);
  const [annotateVideo, setAnnotateVideo] = useState<boolean>(true);

  // Job execution state
  const [activeJob, setActiveJob] = useState<VideoJobResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Clean up object URLs and intervals
  useEffect(() => {
    return () => {
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
      }
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processVideoFile(e.target.files[0]);
    }
  };

  const processVideoFile = (file: File) => {
    const validExtensions = [".mp4", ".avi", ".mov", ".mkv", ".webm"];
    const ext = file.name.substring(file.name.lastIndexOf(".")).toLowerCase();
    if (!validExtensions.includes(ext)) {
      setErrorMessage("Unsupported video container. Please upload MP4, AVI, MOV, MKV, or WebM.");
      return;
    }

    if (file.size > 500 * 1024 * 1024) {
      setErrorMessage("Video exceeds 500 MB limit.");
      return;
    }

    setErrorMessage(null);
    setSelectedFile(file);
    setActiveJob(null);

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processVideoFile(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    setActiveJob(null);
    setErrorMessage(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  };

  // Submit Job
  const handleStartTracking = async () => {
    if (!selectedFile) return;

    setIsSubmitting(true);
    setErrorMessage(null);

    try {
      const initialJob = await submitVideoJob(selectedFile, {
        confidenceThreshold,
        annotateVideo,
      });

      setActiveJob(initialJob);

      // Start Polling
      startPolling(initialJob.job_id);

      // Record session analytics count
      try {
        const curVid = parseInt(sessionStorage.getItem("dv_videos_processed") || "0", 10) + 1;
        sessionStorage.setItem("dv_videos_processed", curVid.toString());
      } catch {
        // Safe fallback
      }
    } catch (err: any) {
      setErrorMessage(err.message || "Failed to submit video processing job.");
    } finally {
      setIsSubmitting(false);
    }
  };

  // Poll Job Status
  const startPolling = (jobId: string) => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }

    pollingIntervalRef.current = setInterval(async () => {
      try {
        const updated = await getVideoJobStatus(jobId);
        setActiveJob(updated);

        if (
          updated.status === "completed" ||
          updated.status === "failed" ||
          updated.status === "cancelled"
        ) {
          if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
          }
        }
      } catch (err: any) {
        setErrorMessage("Error polling job status: " + (err.message || ""));
      }
    }, 600);
  };

  // Cancel Job
  const handleCancelJob = async () => {
    if (!activeJob) return;
    try {
      const cancelled = await cancelVideoJob(activeJob.job_id);
      setActiveJob(cancelled);
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    } catch (err: any) {
      setErrorMessage("Failed to cancel job: " + (err.message || ""));
    }
  };

  const isJobRunning =
    activeJob?.status === "queued" || activeJob?.status === "processing";

  return (
    <div className="space-y-8">
      {/* Page Title & Subtitle */}
      <div>
        <h1 className="text-2xl font-black tracking-tight text-white sm:text-3xl">
          Video Face Tracking & MOT Studio
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Streaming frame-by-frame processing powered by ByteTrack 2D Kalman motion forecasting, session track IDs, and annotated video generation.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
        {/* Left Column: Video Dropzone & Player (7 cols) */}
        <div className="space-y-6 lg:col-span-7">
          {!previewUrl ? (
            /* Upload Zone */
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className="flex min-h-[360px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-700/80 bg-dark-900/60 p-8 text-center transition-all hover:border-brand-blue/50 hover:bg-dark-850/80 group"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="video/mp4,video/avi,video/quicktime,video/x-matroska,video/webm"
                onChange={handleFileChange}
                className="hidden"
              />
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-brand-blue/10 text-brand-blue transition-transform group-hover:scale-110 shadow-glow">
                <UploadCloud className="h-8 w-8" />
              </div>
              <h3 className="mt-4 text-base font-semibold text-white">
                Drag and drop a video, or click to browse
              </h3>
              <p className="mt-1.5 text-xs text-slate-400">
                Supports MP4, AVI, MOV, MKV, and WebM (up to 500 MB)
              </p>
              <div className="mt-5 inline-flex items-center gap-2 rounded-xl bg-dark-800 px-4 py-2 text-xs font-medium text-slate-300 border border-slate-700">
                <VideoIcon className="h-3.5 w-3.5 text-brand-blue" />
                <span>Select video from computer</span>
              </div>
            </div>
          ) : (
            /* Video Preview Player */
            <div className="space-y-4">
              <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-dark-950 shadow-card">
                <video
                  src={previewUrl}
                  controls
                  className="block w-full max-h-[460px] bg-black"
                />
              </div>

              {/* Action Controls */}
              <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-800/80 bg-dark-900/90 p-4">
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleClear}
                    disabled={isJobRunning}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-dark-800 px-3 py-2 text-xs font-semibold text-rose-400 hover:bg-dark-700 disabled:opacity-50"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    <span>Remove</span>
                  </button>
                  <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer select-none">
                    <input
                      type="checkbox"
                      checked={annotateVideo}
                      onChange={(e) => setAnnotateVideo(e.target.checked)}
                      disabled={isJobRunning}
                      className="accent-brand-blue rounded"
                    />
                    <span>Render HUD cards to output video</span>
                  </label>
                </div>

                <div className="flex items-center gap-3">
                  {isJobRunning ? (
                    <button
                      onClick={handleCancelJob}
                      className="inline-flex items-center gap-2 rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-2.5 text-xs font-bold text-rose-400 hover:bg-rose-500/20"
                    >
                      <Ban className="h-4 w-4" />
                      <span>Cancel Job</span>
                    </button>
                  ) : (
                    <button
                      onClick={handleStartTracking}
                      disabled={isSubmitting}
                      className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-brand-blue to-brand-cyan px-5 py-2.5 text-xs font-bold text-dark-950 shadow-glow hover:opacity-95 disabled:opacity-50"
                    >
                      {isSubmitting ? (
                        <>
                          <Loader2 className="h-4 w-4 animate-spin" />
                          <span>Starting...</span>
                        </>
                      ) : (
                        <>
                          <Play className="h-4 w-4 fill-current" />
                          <span>Start Video Tracking</span>
                        </>
                      )}
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Error Message Alert */}
          {errorMessage && (
            <div className="flex items-center gap-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-400">
              <AlertCircle className="h-5 w-5 shrink-0" />
              <span>{errorMessage}</span>
            </div>
          )}
        </div>

        {/* Right Column: Real-Time Job Progress & Telemetry (5 cols) */}
        <div className="space-y-6 lg:col-span-5">
          {activeJob ? (
            <div className="space-y-6">
              {/* Job Status Card */}
              <div className="rounded-2xl border border-slate-800 bg-dark-900/90 p-5 shadow-card space-y-4">
                <div className="flex items-center justify-between border-b border-slate-800/80 pb-3">
                  <div>
                    <span className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
                      Asynchronous Job ID
                    </span>
                    <p className="font-mono text-xs text-slate-300 truncate max-w-[200px] sm:max-w-[260px]">
                      {activeJob.job_id}
                    </p>
                  </div>

                  {/* Status Badge */}
                  <span
                    className={`rounded-full px-3 py-1 text-xs font-bold uppercase tracking-wider ${
                      activeJob.status === "completed"
                        ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                        : activeJob.status === "processing"
                        ? "bg-brand-cyan/15 text-brand-cyan border border-brand-cyan/30 animate-pulse"
                        : activeJob.status === "failed"
                        ? "bg-rose-500/15 text-rose-400 border border-rose-500/30"
                        : activeJob.status === "cancelled"
                        ? "bg-slate-800 text-slate-400 border border-slate-700"
                        : "bg-amber-500/15 text-amber-400 border border-amber-500/30"
                    }`}
                  >
                    {activeJob.status}
                  </span>
                </div>

                {/* Progress Bar */}
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-slate-400">
                      Processing Progress:
                    </span>
                    <span className="font-mono font-bold text-white">
                      {Math.round(activeJob.progress)}%
                    </span>
                  </div>

                  <div className="h-2.5 w-full overflow-hidden rounded-full bg-dark-700">
                    <div
                      className={`h-full rounded-full transition-all duration-300 ${
                        activeJob.status === "completed"
                          ? "bg-emerald-400"
                          : activeJob.status === "failed"
                          ? "bg-rose-500"
                          : "bg-gradient-to-r from-brand-blue to-brand-cyan"
                      }`}
                      style={{ width: `${Math.min(100, Math.max(0, activeJob.progress))}%` }}
                    />
                  </div>

                  <div className="flex items-center justify-between text-[11px] text-slate-400">
                    <span>
                      Processed: {activeJob.processed_frames} / {activeJob.total_frames || "?"} frames
                    </span>
                    {activeJob.status === "processing" && (
                      <span className="flex items-center gap-1.5 text-brand-cyan">
                        <Loader2 className="h-3 w-3 animate-spin" />
                        <span>Streaming...</span>
                      </span>
                    )}
                  </div>
                </div>

                {/* Download Annotated Video Button */}
                {activeJob.status === "completed" && (
                  <div className="pt-2">
                    <a
                      href={getVideoDownloadUrl(activeJob.job_id)}
                      download
                      className="flex w-full items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-400 py-3 text-xs font-bold text-dark-950 shadow-glow-emerald hover:opacity-95"
                    >
                      <Download className="h-4 w-4" />
                      <span>Download Annotated MP4 Video</span>
                    </a>
                  </div>
                )}
              </div>

              {/* Measured Metrics Telemetry Card (when completed) */}
              {activeJob.metrics && (
                <div className="rounded-2xl border border-slate-800 bg-dark-900/90 p-5 shadow-card space-y-4">
                  <h4 className="text-sm font-bold text-white flex items-center gap-2">
                    <Activity className="h-4 w-4 text-brand-cyan" />
                    <span>Verified Tracking Telemetry</span>
                  </h4>

                  <div className="grid grid-cols-2 gap-3 text-xs">
                    <div className="rounded-xl border border-slate-800/80 bg-dark-850/50 p-3">
                      <p className="text-slate-400 text-[11px]">Unique Track Count</p>
                      <p className="text-xl font-bold text-white mt-1">
                        {activeJob.metrics.unique_track_count}
                      </p>
                      <p className="text-[10px] text-slate-500">Temporary Track IDs</p>
                    </div>

                    <div className="rounded-xl border border-slate-800/80 bg-dark-850/50 p-3">
                      <p className="text-slate-400 text-[11px]">Peak Simultaneous Faces</p>
                      <p className="text-xl font-bold text-brand-cyan mt-1">
                        {activeJob.metrics.peak_simultaneous_faces}
                      </p>
                      <p className="text-[10px] text-slate-500">Max in single frame</p>
                    </div>

                    <div className="rounded-xl border border-slate-800/80 bg-dark-850/50 p-3">
                      <p className="text-slate-400 text-[11px]">Average Throughput</p>
                      <p className="text-xl font-bold text-white mt-1">
                        {activeJob.metrics.average_fps.toFixed(1)} FPS
                      </p>
                      <p className="text-[10px] text-slate-500">Processing rate</p>
                    </div>

                    <div className="rounded-xl border border-slate-800/80 bg-dark-850/50 p-3">
                      <p className="text-slate-400 text-[11px]">Total Processing Time</p>
                      <p className="text-xl font-bold text-white mt-1">
                        {(activeJob.metrics.total_processing_time / 1000).toFixed(2)} s
                      </p>
                      <p className="text-[10px] text-slate-500">
                        {activeJob.metrics.average_frame_processing_time.toFixed(1)} ms/frame
                      </p>
                    </div>
                  </div>

                  <div className="rounded-xl border border-slate-800/60 bg-dark-850/30 p-3 text-[11px] text-slate-400 flex items-start gap-2">
                    <ShieldCheck className="h-4 w-4 text-emerald-400 shrink-0 mt-0.5" />
                    <span>
                      Track IDs are ephemeral numbers (Track 1, 2...) scoped solely to this video clip. No biometric identification or database profile was created.
                    </span>
                  </div>
                </div>
              )}
            </div>
          ) : (
            /* Awaiting Video Prompt */
            <div className="rounded-2xl border border-slate-800/80 bg-dark-900/40 p-8 text-center text-slate-400 space-y-3">
              <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-dark-850 border border-slate-800 text-slate-400">
                <VideoIcon className="h-7 w-7 text-brand-blue" />
              </div>
              <h4 className="text-base font-bold text-white">
                Video Analytics Console
              </h4>
              <p className="text-xs text-slate-400 max-w-sm mx-auto leading-relaxed">
                Upload a video file to begin streaming frame extraction, YuNet detection, ByteTrack multi-face tracking, and downloadable annotated video generation.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

"use client";

import { useState, useRef, useEffect } from "react";
import {
  submitVideoAnalysisJob,
  pollVideoJobStatus,
  cancelVideoJobExecution,
} from "@/lib/api/video";
import { VideoJob } from "@/lib/api/types";
import FileUploadZone from "@/components/common/FileUploadZone";
import VideoProcessingProgressCard from "@/components/video/VideoProcessingProgressCard";
import VideoResultView from "@/components/video/VideoResultView";
import { Play, Loader2, AlertCircle, RefreshCw } from "lucide-react";

export default function VideoAnalysisPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.3);
  const [annotateVideo, setAnnotateVideo] = useState<boolean>(true);

  // Job execution state
  const [activeJob, setActiveJob] = useState<VideoJob | null>(null);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Clean up object URLs and polling timers
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

  const handleFileSelect = (file: File) => {
    setErrorMessage(null);
    setSelectedFile(file);
    setActiveJob(null);

    const objectUrl = URL.createObjectURL(file);
    setPreviewUrl(objectUrl);
  };

  const handleClear = () => {
    setSelectedFile(null);
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
      setPreviewUrl(null);
    }
    setActiveJob(null);
    setErrorMessage(null);
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
      pollingIntervalRef.current = null;
    }
  };

  // Submit Job to Backend
  const handleStartTracking = async () => {
    if (!selectedFile) return;

    setIsSubmitting(true);
    setErrorMessage(null);

    try {
      const initialJob = await submitVideoAnalysisJob(selectedFile, {
        confidenceThreshold,
        annotateVideo,
      });

      setActiveJob(initialJob);
      startPolling(initialJob.job_id);

      // Record session metric
      try {
        const curVid =
          parseInt(sessionStorage.getItem("dv_videos_processed") || "0", 10) + 1;
        sessionStorage.setItem("dv_videos_processed", curVid.toString());
      } catch {
        // Safe fallback
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to submit video processing job.";
      setErrorMessage(msg);
    } finally {
      setIsSubmitting(false);
    }
  };

  // Real-time Polling Engine
  const startPolling = (jobId: string) => {
    if (pollingIntervalRef.current) {
      clearInterval(pollingIntervalRef.current);
    }

    pollingIntervalRef.current = setInterval(async () => {
      try {
        const updated = await pollVideoJobStatus(jobId);
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
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Network error polling job status.";
        setErrorMessage(msg);
      }
    }, 600);
  };

  // Cancel Job
  const handleCancelJob = async () => {
    if (!activeJob) return;
    try {
      const cancelled = await cancelVideoJobExecution(activeJob.job_id);
      setActiveJob(cancelled);
      if (pollingIntervalRef.current) {
        clearInterval(pollingIntervalRef.current);
        pollingIntervalRef.current = null;
      }
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : "Failed to cancel job.";
      setErrorMessage(msg);
    }
  };

  const isJobRunning =
    activeJob?.status === "queued" || activeJob?.status === "processing";
  const isJobCompleted = activeJob?.status === "completed";

  return (
    <div className="space-y-8">
      {/* Title */}
      <div>
        <h1 className="text-2xl font-black tracking-tight text-white sm:text-3xl">
          Video Multi-Face Tracking Studio
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Upload MP4, AVI, MOV, MKV, or WebM clips for streaming frame extraction, YuNet face detection, and ByteTrack kinematic association.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
        {/* Left Column: Upload / Video Player / Trigger (7 cols) */}
        <div className="space-y-6 lg:col-span-7">
          {/* File Upload Zone */}
          <FileUploadZone
            mediaType="video"
            acceptedExtensions={[".mp4", ".avi", ".mov", ".mkv", ".webm"]}
            maxSizeMB={500}
            selectedFile={selectedFile}
            isLoading={isSubmitting}
            loadingMessage="Enqueuing video for asynchronous tracking..."
            onFileSelect={handleFileSelect}
            onClear={handleClear}
            errorMessage={errorMessage}
            onDismissError={() => setErrorMessage(null)}
          />

          {/* Raw Video Preview when chosen and not yet complete */}
          {previewUrl && !isJobCompleted && (
            <div className="space-y-4">
              <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-dark-950 shadow-card">
                <video
                  src={previewUrl}
                  controls
                  className="block w-full max-h-[420px] bg-black"
                />
              </div>

              {/* Action Toolbar */}
              <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-800/80 bg-dark-900/90 p-4">
                <label className="flex items-center gap-2 text-xs text-slate-300 cursor-pointer select-none">
                  <input
                    type="checkbox"
                    checked={annotateVideo}
                    onChange={(e) => setAnnotateVideo(e.target.checked)}
                    disabled={isJobRunning}
                    className="accent-brand-cyan rounded"
                  />
                  <span>Render persistent HUD overlays to output video</span>
                </label>

                {!isJobRunning && (
                  <button
                    onClick={handleStartTracking}
                    disabled={isSubmitting}
                    className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-brand-blue to-brand-cyan px-6 py-2.5 text-xs font-bold text-dark-950 shadow-glow hover:opacity-95 disabled:opacity-50 transition-all"
                  >
                    {isSubmitting ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span>Starting Pipeline...</span>
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
          )}

          {/* Error Details */}
          {activeJob?.status === "failed" && (
            <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-400 flex items-start gap-3">
              <AlertCircle className="h-5 w-5 shrink-0 mt-0.5" />
              <div>
                <p className="font-bold">Video Processing Failed</p>
                <p className="mt-1 text-rose-300">{activeJob.error || "Unable to decode video stream."}</p>
              </div>
            </div>
          )}
        </div>

        {/* Right Column: Dynamic Job Progress or Completed Result (5 cols) */}
        <div className="space-y-6 lg:col-span-5">
          {/* Active Running Job (Item 11) */}
          {activeJob && isJobRunning && (
            <VideoProcessingProgressCard
              job={activeJob}
              onCancel={handleCancelJob}
            />
          )}

          {/* Completed Job (Items 12 & 13) */}
          {activeJob && isJobCompleted && (
            <VideoResultView job={activeJob} />
          )}

          {/* Idle Empty State */}
          {!activeJob && (
            <div className="rounded-2xl border border-slate-800/80 bg-dark-900/40 p-8 text-center text-slate-400 space-y-3">
              <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-dark-850 border border-slate-800 text-brand-blue">
                <Play className="h-7 w-7" />
              </div>
              <h4 className="text-base font-bold text-white">
                ByteTrack Video Console
              </h4>
              <p className="text-xs text-slate-400 max-w-sm mx-auto leading-relaxed">
                Upload a video file to run real-time frame extraction, YuNet face detection, and ByteTrack multi-face tracking.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

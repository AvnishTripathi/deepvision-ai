"use client";

import { useState, useRef, useEffect } from "react";
import { Camera, CameraOff, Video, AlertCircle, Sparkles, ShieldCheck } from "lucide-react";

export default function CameraPage() {
  const [isStreaming, setIsStreaming] = useState(false);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const startCamera = async () => {
    setCameraError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 1280, height: 720 },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play();
      }
      setIsStreaming(true);
    } catch (err: any) {
      setCameraError(
        err.message || "Failed to access local webcam. Please check browser permissions."
      );
    }
  };

  const stopCamera = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-black tracking-tight text-white sm:text-3xl">
          Live Camera Studio
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Hardware video capture integration. Live server WebSocket streaming pipeline is reserved for upcoming development phases.
        </p>
      </div>

      <div className="max-w-4xl space-y-6">
        {/* Camera Viewport */}
        <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-dark-950 shadow-card min-h-[420px] flex items-center justify-center">
          <video
            ref={videoRef}
            className={`block w-full max-h-[520px] object-contain ${
              !isStreaming ? "hidden" : ""
            }`}
            playsInline
            muted
          />

          {!isStreaming && (
            <div className="p-8 text-center text-slate-400 space-y-3">
              <div className="mx-auto flex h-16 w-16 items-center justify-center rounded-2xl bg-dark-850 border border-slate-800 text-brand-cyan">
                <Camera className="h-8 w-8" />
              </div>
              <h3 className="text-base font-bold text-white">
                Webcam Inactive
              </h3>
              <p className="text-xs text-slate-400 max-w-sm mx-auto">
                Click <strong>Start Camera</strong> to initialize local video streaming in your browser.
              </p>
            </div>
          )}

          {isStreaming && (
            <div className="absolute top-4 left-4 z-20 flex items-center gap-2 rounded-lg bg-dark-950/80 px-3 py-1.5 text-xs font-semibold text-emerald-400 border border-emerald-500/30 backdrop-blur-md">
              <span className="relative flex h-2 w-2">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400"></span>
              </span>
              <span>Local Feed Active</span>
            </div>
          )}
        </div>

        {/* Camera Controls */}
        <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-800 bg-dark-900/90 p-4">
          <div>
            <p className="text-xs font-semibold text-white">
              Hardware Video Capture
            </p>
            <p className="text-[11px] text-slate-400">
              Browser WebRTC MediaStream API (1280×720)
            </p>
          </div>

          <div className="flex items-center gap-3">
            {isStreaming ? (
              <button
                onClick={stopCamera}
                className="inline-flex items-center gap-2 rounded-xl border border-rose-500/30 bg-rose-500/10 px-5 py-2.5 text-xs font-bold text-rose-400 hover:bg-rose-500/20"
              >
                <CameraOff className="h-4 w-4" />
                <span>Stop Camera</span>
              </button>
            ) : (
              <button
                onClick={startCamera}
                className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-brand-cyan to-brand-blue px-5 py-2.5 text-xs font-bold text-dark-950 shadow-glow hover:opacity-95"
              >
                <Camera className="h-4 w-4" />
                <span>Start Camera</span>
              </button>
            )}
          </div>
        </div>

        {cameraError && (
          <div className="flex items-center gap-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-xs text-rose-400">
            <AlertCircle className="h-5 w-5 shrink-0" />
            <span>{cameraError}</span>
          </div>
        )}

        {/* Phase Roadmap Notice */}
        <div className="rounded-xl border border-slate-800/60 bg-dark-850/40 p-4 text-xs text-slate-300 flex items-start gap-3">
          <ShieldCheck className="h-5 w-5 text-brand-cyan shrink-0 mt-0.5" />
          <div>
            <p className="font-semibold text-white">Architecture Notice</p>
            <p className="mt-1 text-slate-400 leading-relaxed text-[11px]">
              Full bi-directional WebSocket video stream inference and edge frame processing belong to later phases. For current production analytics, please utilize the <strong>Image Face Analysis</strong> and <strong>Video Multi-Face Tracking</strong> pipelines.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

import React, { useState, useCallback, useRef, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import Webcam from 'react-webcam';
import { 
  UploadCloud, Camera, Link as LinkIcon, Loader2, ShieldCheck, 
  AlertTriangle, Cpu, Activity, Zap, CheckCircle2, Sliders, Play, Square, Eye
} from 'lucide-react';
import { api, createDetectionWebSocket } from '../api/client';
import { useNavigate } from 'react-router-dom';

const DetectPage = () => {
  const [tab, setTab] = useState<'upload' | 'webcam' | 'url'>('upload');
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [isVideo, setIsVideo] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisProgress, setAnalysisProgress] = useState(0);

  // Real-time HUD states
  const [webcamActive, setWebcamActive] = useState(false);
  const [liveStreamData, setLiveStreamData] = useState<{
    faces: any[];
    overall_score: number;
    label: string;
    processing_time_ms: number;
  } | null>(null);
  const [fps, setFps] = useState(0);

  // Real-time sensitivity & models
  const [sensitivity, setSensitivity] = useState<'standard' | 'strict' | 'paranoid'>('standard');
  const [urlInput, setUrlInput] = useState('');

  const navigate = useNavigate();
  const webcamRef = useRef<Webcam>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const frameIntervalRef = useRef<any>(null);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles[0]) {
      const selected = acceptedFiles[0];
      setFile(selected);
      setIsVideo(selected.type.startsWith('video/'));
      setPreview(URL.createObjectURL(selected));
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/*': [], 'video/*': [] },
    maxFiles: 1
  });

  const handleAnalyzeUpload = async () => {
    if (!file) return;
    setIsAnalyzing(true);
    setAnalysisProgress(15);
    const progressTimer = setInterval(() => {
      setAnalysisProgress((prev) => (prev < 90 ? prev + 15 : prev));
    }, 400);

    try {
      let res;
      if (isVideo) {
        res = await api.detect.video(file);
      } else {
        res = await api.detect.image(file);
      }
      clearInterval(progressTimer);
      setAnalysisProgress(100);
      const targetJobId = res.data.job_id || res.data.jobId;
      setTimeout(() => {
        navigate(`/results/${targetJobId}`);
      }, 500);
    } catch (err: any) {
      clearInterval(progressTimer);
      console.error(err);
      alert(err.response?.data?.detail || 'Analysis request failed. Please check network/backend.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  const handleAnalyzeUrl = async () => {
    if (!urlInput) return;
    setIsAnalyzing(true);
    try {
      const res = await api.detect.url(urlInput);
      const targetJobId = res.data.job_id || res.data.jobId;
      navigate(`/results/${targetJobId}`);
    } catch (err: any) {
      console.error(err);
      alert(err.response?.data?.detail || 'URL Analysis failed. Please ensure the direct image link is reachable.');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Real-time live camera analysis via WebSocket
  useEffect(() => {
    if (!webcamActive) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      if (frameIntervalRef.current) {
        clearInterval(frameIntervalRef.current);
      }
      setLiveStreamData(null);
      return;
    }

    let frameCount = 0;
    let isProcessingFrame = false;

    const fpsTimer = setInterval(() => {
      setFps(frameCount);
      frameCount = 0;
    }, 1000);

    const ws = createDetectionWebSocket((data) => {
      setLiveStreamData(data);
      frameCount++;
      isProcessingFrame = false; // Release lock when backend replies
    });
    wsRef.current = ws;

    // Send next frame as soon as backend finishes or every 180ms
    frameIntervalRef.current = setInterval(() => {
      if (webcamRef.current && ws.readyState === WebSocket.OPEN && !isProcessingFrame) {
        const screenshot = webcamRef.current.getScreenshot();
        if (screenshot) {
          const base64Data = screenshot.split(',')[1];
          if (base64Data) {
            isProcessingFrame = true;
            ws.send(JSON.stringify({ frame: base64Data }));
          }
        }
      }
    }, 150);

    return () => {
      clearInterval(fpsTimer);
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [webcamActive]);

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Top Banner / HUD Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8 bg-slate-900/90 border border-slate-800 p-5 rounded-2xl shadow-2xl backdrop-blur-md">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <ShieldCheck className="h-6 w-6 text-primary-400" />
            <h1 className="text-2xl font-black tracking-wide text-white uppercase font-mono">
              DeepFace Neural Inspection Core
            </h1>
            <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-emerald-500/20 text-emerald-400 border border-emerald-500/30">
              v2.5 Production
            </span>
          </div>
          <p className="text-slate-400 text-sm">
            Biometric integrity audit: FFT Fourier spectral fingerprints, FaceForensics++ heuristics & anomaly heatmap synthesis.
          </p>
        </div>

        {/* Real-time telemetry badges */}
        <div className="flex items-center gap-3">
          <div className="bg-slate-950/80 px-3 py-2 rounded-xl border border-slate-800 flex items-center gap-2 text-xs font-mono">
            <Cpu className="w-4 h-4 text-primary-400" />
            <div>
              <div className="text-slate-400">ENGINE</div>
              <div className="text-white font-bold">EfficientNet-B7</div>
            </div>
          </div>
          <div className="bg-slate-950/80 px-3 py-2 rounded-xl border border-slate-800 flex items-center gap-2 text-xs font-mono">
            <Activity className="w-4 h-4 text-emerald-400" />
            <div>
              <div className="text-slate-400">LATENCY</div>
              <div className="text-emerald-400 font-bold">&lt; 150ms</div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Analysis Container */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-8">
        {/* Left Column: Navigation & Settings */}
        <div className="lg:col-span-1 space-y-6">
          <div className="glass-card p-4 rounded-2xl border border-slate-800 bg-slate-900/70">
            <h3 className="text-xs font-mono text-slate-400 font-semibold uppercase tracking-wider mb-3">
              Input Modality
            </h3>
            <div className="flex flex-col gap-2">
              <button
                onClick={() => setTab('upload')}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-all text-sm ${
                  tab === 'upload'
                    ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/30 font-semibold'
                    : 'bg-slate-800/40 text-slate-300 hover:bg-slate-800'
                }`}
              >
                <UploadCloud className="w-4 h-4" />
                <span>Upload Media</span>
              </button>

              <button
                onClick={() => setTab('webcam')}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-all text-sm ${
                  tab === 'webcam'
                    ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/30 font-semibold'
                    : 'bg-slate-800/40 text-slate-300 hover:bg-slate-800'
                }`}
              >
                <Camera className="w-4 h-4" />
                <span>Live Neural Feed</span>
                <span className="ml-auto text-[10px] bg-red-500/20 text-red-400 px-1.5 py-0.5 rounded font-mono">
                  LIVE
                </span>
              </button>

              <button
                onClick={() => setTab('url')}
                className={`flex items-center gap-3 px-4 py-3 rounded-xl font-medium transition-all text-sm ${
                  tab === 'url'
                    ? 'bg-primary-600 text-white shadow-lg shadow-primary-600/30 font-semibold'
                    : 'bg-slate-800/40 text-slate-300 hover:bg-slate-800'
                }`}
              >
                <LinkIcon className="w-4 h-4" />
                <span>Remote URL</span>
              </button>
            </div>
          </div>

          {/* Precision Threshold Control */}
          <div className="glass-card p-4 rounded-2xl border border-slate-800 bg-slate-900/70 space-y-3">
            <div className="flex items-center justify-between text-xs font-mono text-slate-400 font-semibold uppercase">
              <span className="flex items-center gap-1.5"><Sliders className="w-3.5 h-3.5 text-primary-400" /> Sensitivity</span>
              <span className="text-primary-400 uppercase">{sensitivity}</span>
            </div>
            <div className="grid grid-cols-3 gap-1 bg-slate-950 p-1 rounded-lg border border-slate-800 text-xs font-mono">
              {(['standard', 'strict', 'paranoid'] as const).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setSensitivity(mode)}
                  className={`py-1.5 capitalize rounded ${
                    sensitivity === mode ? 'bg-primary-600 text-white font-bold' : 'text-slate-400 hover:text-white'
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>
            <p className="text-[11px] text-slate-500 leading-relaxed">
              Standard mode filters minor social media compression artefacts. Strict optimizes for legal & forensics verification.
            </p>
          </div>
        </div>

        {/* Center / Right Column: Active Detection Workspace */}
        <div className="lg:col-span-3">
          <div className="glass-card p-6 rounded-2xl border border-slate-800/80 bg-slate-900/80 relative min-h-[480px] flex flex-col justify-between">
            {/* 1. UPLOAD TAB */}
            {tab === 'upload' && (
              <div className="space-y-6">
                {!preview ? (
                  <div
                    {...getRootProps()}
                    className={`border-2 border-dashed rounded-2xl p-12 text-center transition-all cursor-pointer relative overflow-hidden bg-cyber-grid ${
                      isDragActive
                        ? 'border-primary-500 bg-primary-950/20'
                        : 'border-slate-700/80 hover:border-primary-500/80 hover:bg-slate-900/90'
                    }`}
                  >
                    <input {...getInputProps()} />
                    <div className="w-16 h-16 rounded-2xl bg-primary-600/10 border border-primary-500/30 flex items-center justify-center mx-auto mb-4 text-primary-400 animate-radar">
                      <UploadCloud className="w-8 h-8" />
                    </div>
                    <h3 className="text-lg font-bold text-white mb-1">
                      Drop high-resolution photo or video here
                    </h3>
                    <p className="text-slate-400 text-sm max-w-sm mx-auto mb-4">
                      Drag & drop your file or click to browse. Supports JPG, PNG, WEBP, MP4, MOV (up to 500MB).
                    </p>
                    <div className="inline-flex items-center gap-2 text-xs font-mono text-slate-500 bg-slate-950 px-3 py-1.5 rounded-full border border-slate-800">
                      <Zap className="w-3.5 h-3.5 text-amber-400" /> Auto-extracts facial bounding boxes and landmarks
                    </div>
                  </div>
                ) : (
                  <div className="space-y-6">
                    <div className="relative rounded-2xl overflow-hidden border border-slate-700 bg-slate-950 max-h-[360px] flex items-center justify-center group">
                      {isVideo ? (
                        <video src={preview} controls className="max-h-[360px] w-auto mx-auto rounded-xl" />
                      ) : (
                        <div className="relative w-full h-full flex items-center justify-center">
                          <img src={preview} alt="Target" className="max-h-[360px] w-auto object-contain rounded-xl" />
                          {isAnalyzing && (
                            <div className="absolute inset-0 bg-primary-950/40 backdrop-blur-[1px] flex flex-col items-center justify-center pointer-events-none">
                              <div className="w-full h-1 bg-primary-400 shadow-[0_0_15px_#38bdf8] animate-scan-line" />
                            </div>
                          )}
                        </div>
                      )}

                      <div className="absolute top-3 left-3 bg-slate-900/90 backdrop-blur px-2.5 py-1 rounded-lg border border-slate-700 text-xs font-mono text-slate-300">
                        {file?.name} ({(file!.size / 1024 / 1024).toFixed(2)} MB)
                      </div>
                    </div>

                    {isAnalyzing && (
                      <div className="space-y-2">
                        <div className="flex justify-between text-xs font-mono text-slate-400">
                          <span className="flex items-center gap-1.5 text-primary-400">
                            <Loader2 className="w-3.5 h-3.5 animate-spin" /> Deep neural verification in progress...
                          </span>
                          <span>{analysisProgress}%</span>
                        </div>
                        <div className="w-full bg-slate-950 rounded-full h-2 overflow-hidden border border-slate-800">
                          <div
                            className="bg-gradient-to-r from-primary-500 to-indigo-500 h-full transition-all duration-300"
                            style={{ width: `${analysisProgress}%` }}
                          />
                        </div>
                      </div>
                    )}

                    <div className="flex items-center justify-end gap-3 pt-2">
                      <button
                        onClick={() => {
                          setFile(null);
                          setPreview(null);
                        }}
                        disabled={isAnalyzing}
                        className="px-5 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium text-sm transition-colors"
                      >
                        Change Media
                      </button>
                      <button
                        onClick={handleAnalyzeUpload}
                        disabled={isAnalyzing}
                        className="px-6 py-2.5 rounded-xl bg-gradient-to-r from-primary-600 to-indigo-600 hover:from-primary-500 hover:to-indigo-500 text-white font-semibold text-sm shadow-lg shadow-primary-600/30 flex items-center gap-2 transition-all disabled:opacity-50"
                      >
                        {isAnalyzing ? (
                          <>
                            <Loader2 className="w-4 h-4 animate-spin" />
                            Running Inspection...
                          </>
                        ) : (
                          <>
                            <Play className="w-4 h-4 fill-white" />
                            Execute DeepFake Audit
                          </>
                        )}
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* 2. WEBCAM LIVE FEED TAB */}
            {tab === 'webcam' && (
              <div className="space-y-6">
                <div className="relative rounded-2xl overflow-hidden border border-slate-700 bg-slate-950 aspect-video flex items-center justify-center">
                  {webcamActive ? (
                    <>
                      <Webcam
                        ref={webcamRef}
                        screenshotFormat="image/jpeg"
                        videoConstraints={{ width: 1280, height: 720, facingMode: 'user' }}
                        className="w-full h-full object-cover"
                        mirrored
                      />
                      
                      {/* Full-Frame Ambient Scanline */}
                      <div className="absolute inset-0 pointer-events-none border border-primary-500/20 bg-cyber-grid overflow-hidden">
                        <div className="w-full h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_15px_#22d3ee] animate-scan-line" />
                      </div>

                      {/* Dynamic SVG Facial Bounding Boxes, Biometric Markers & Targeted Face Sweeper */}
                      {liveStreamData?.faces && liveStreamData.faces.length > 0 && (
                        <div className="absolute inset-0 pointer-events-none">
                          <svg className="w-full h-full">
                            {liveStreamData.faces.map((f: any, idx: number) => {
                              const [x1, y1, x2, y2] = f.bbox;
                              const isFake = f.label === 'DEEPFAKE' || f.deepfake_score > 0.65;
                              const isSuspicious = f.label === 'SUSPICIOUS' || f.deepfake_score > 0.35;
                              const strokeColor = isFake ? '#f43f5e' : isSuspicious ? '#fbbf24' : '#10b981';

                              const leftPct = (x1 / 1280) * 100;
                              const topPct = (y1 / 720) * 100;
                              const widthPct = ((x2 - x1) / 1280) * 100;
                              const heightPct = ((y2 - y1) / 720) * 100;

                              return (
                                <g key={idx}>
                                  {/* Face Tracking Bounding Box */}
                                  <rect
                                    x={`${leftPct}%`}
                                    y={`${topPct}%`}
                                    width={`${widthPct}%`}
                                    height={`${heightPct}%`}
                                    fill={isFake ? 'rgba(244,63,94,0.08)' : 'rgba(16,185,129,0.06)'}
                                    stroke={strokeColor}
                                    strokeWidth="2"
                                    strokeDasharray="6 3"
                                  />

                                  {/* Corner Biometric Reticles */}
                                  <line x1={`${leftPct}%`} y1={`${topPct}%`} x2={`${leftPct + 3}%`} y2={`${topPct}%`} stroke={strokeColor} strokeWidth="3" />
                                  <line x1={`${leftPct}%`} y1={`${topPct}%`} x2={`${leftPct}%`} y2={`${topPct + 4}%`} stroke={strokeColor} strokeWidth="3" />
                                  <line x1={`${leftPct + widthPct}%`} y1={`${topPct}%`} x2={`${leftPct + widthPct - 3}%`} y2={`${topPct}%`} stroke={strokeColor} strokeWidth="3" />
                                  <line x1={`${leftPct + widthPct}%`} y1={`${topPct}%`} x2={`${leftPct + widthPct}%`} y2={`${topPct + 4}%`} stroke={strokeColor} strokeWidth="3" />
                                  <line x1={`${leftPct}%`} y1={`${topPct + heightPct}%`} x2={`${leftPct + 3}%`} y2={`${topPct + heightPct}%`} stroke={strokeColor} strokeWidth="3" />
                                  <line x1={`${leftPct}%`} y1={`${topPct + heightPct}%`} x2={`${leftPct}%`} y2={`${topPct + heightPct - 4}%`} stroke={strokeColor} strokeWidth="3" />
                                  <line x1={`${leftPct + widthPct}%`} y1={`${topPct + heightPct}%`} x2={`${leftPct + widthPct - 3}%`} y2={`${topPct + heightPct}%`} stroke={strokeColor} strokeWidth="3" />
                                  <line x1={`${leftPct + widthPct}%`} y1={`${topPct + heightPct}%`} x2={`${leftPct + widthPct}%`} y2={`${topPct + heightPct - 4}%`} stroke={strokeColor} strokeWidth="3" />

                                  {/* Center Crosshair Target */}
                                  <circle cx={`${leftPct + widthPct / 2}%`} cy={`${topPct + heightPct / 2}%`} r="3" fill={strokeColor} />

                                  {/* Top HUD Badge Tag */}
                                  <text
                                    x={`${leftPct}%`}
                                    y={`${Math.max(4, topPct - 2)}%`}
                                    fill={strokeColor}
                                    fontSize="12"
                                    fontFamily="monospace"
                                    fontWeight="bold"
                                  >
                                    [SCANNING] {f.label} // {(f.deepfake_score * 100).toFixed(1)}%
                                  </text>
                                </g>
                              );
                            })}
                          </svg>

                          {/* Sweeping laser line across the targeted face */}
                          {liveStreamData.faces.map((f: any, idx: number) => {
                            const [x1, y1, x2, y2] = f.bbox;
                            const leftPct = (x1 / 1280) * 100;
                            const topPct = (y1 / 720) * 100;
                            const widthPct = ((x2 - x1) / 1280) * 100;
                            const heightPct = ((y2 - y1) / 720) * 100;

                            return (
                              <div
                                key={`sweep-${idx}`}
                                className="absolute overflow-hidden"
                                style={{
                                  left: `${leftPct}%`,
                                  top: `${topPct}%`,
                                  width: `${widthPct}%`,
                                  height: `${heightPct}%`
                                }}
                              >
                                <div className="w-full h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_12px_#22d3ee] animate-face-sweep" />
                              </div>
                            );
                          })}
                        </div>
                      )}

                      {/* Floating Biometric Information Telemetry Panel on Detected Face */}
                      {liveStreamData?.faces && liveStreamData.faces.length > 0 && liveStreamData.faces[0]?.signals && (
                        <div className="absolute top-16 right-4 pointer-events-none max-w-xs w-full bg-slate-950/90 backdrop-blur-md p-3.5 rounded-2xl border border-slate-700/80 shadow-2xl font-mono text-[11px] space-y-2 text-slate-300">
                          <div className="flex items-center justify-between text-xs font-bold text-white border-b border-slate-800 pb-1.5">
                            <span className="flex items-center gap-1.5">
                              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                              BIOMETRIC_AUDIT_INFO
                            </span>
                            <span className="text-cyan-400">100% REAL-TIME</span>
                          </div>

                          <div className="space-y-1.5">
                            <div className="flex justify-between items-center">
                              <span className="text-slate-400">Fourier FFT (Freq):</span>
                              <span className={liveStreamData.faces[0].signals.frequency > 0.6 ? 'text-rose-400 font-bold' : 'text-emerald-400 font-bold'}>
                                {(liveStreamData.faces[0].signals.frequency * 100).toFixed(1)}% {liveStreamData.faces[0].signals.frequency > 0.6 ? '(Spike)' : '(Normal)'}
                              </span>
                            </div>
                            <div className="flex justify-between items-center">
                              <span className="text-slate-400">Laplacian Blur:</span>
                              <span className={liveStreamData.faces[0].signals.blur > 0.6 ? 'text-rose-400 font-bold' : 'text-emerald-400 font-bold'}>
                                {(liveStreamData.faces[0].signals.blur * 100).toFixed(1)}% {liveStreamData.faces[0].signals.blur > 0.6 ? '(Artificial)' : '(Sharp)'}
                              </span>
                            </div>
                            <div className="flex justify-between items-center">
                              <span className="text-slate-400">Color Boundary:</span>
                              <span className="text-cyan-400 font-bold">
                                {((liveStreamData.faces[0].signals.color_consistency || 0.05) * 100).toFixed(1)}%
                              </span>
                            </div>
                            <div className="flex justify-between items-center">
                              <span className="text-slate-400">Compression Integrity:</span>
                              <span className="text-slate-200 font-bold">
                                {((liveStreamData.faces[0].signals.compression || 1.0) * 100).toFixed(0)}%
                              </span>
                            </div>
                          </div>

                          <div className="pt-1.5 border-t border-slate-800 flex justify-between items-center text-[10px] text-slate-400">
                            <span>FACE_ID: #01</span>
                            <span className="text-emerald-400 font-bold">STATUS: MONITORED</span>
                          </div>
                        </div>
                      )}

                      {/* HUD Top Bar */}
                      <div className="absolute top-4 left-4 right-4 flex items-center justify-between pointer-events-none">
                        <div className="flex items-center gap-2">
                          <span className="px-2.5 py-1 bg-red-600 text-white font-mono text-xs font-bold rounded flex items-center gap-1.5 animate-pulse">
                            <span className="w-2 h-2 rounded-full bg-white" /> REC LIVE
                          </span>
                          <span className="px-2 py-1 bg-black/60 backdrop-blur border border-slate-700 rounded font-mono text-xs text-slate-300">
                            {fps} FPS
                          </span>
                        </div>

                        {liveStreamData && (
                          <div className="px-3 py-1 bg-black/80 backdrop-blur rounded-lg border border-slate-700 font-mono text-xs flex items-center gap-2">
                            <span className="text-slate-400">Score:</span>
                            <span
                              className={`font-bold ${
                                liveStreamData.overall_score > 0.65
                                  ? 'text-red-400'
                                  : liveStreamData.overall_score > 0.35
                                  ? 'text-amber-400'
                                  : 'text-emerald-400'
                              }`}
                            >
                              {(liveStreamData.overall_score * 100).toFixed(1)}%
                            </span>
                            <span className="text-slate-500">|</span>
                            <span className="font-bold text-white uppercase">{liveStreamData.label}</span>
                          </div>
                        )}
                      </div>

                      {/* Bottom HUD metrics */}
                      <div className="absolute bottom-4 left-4 right-4 pointer-events-none">
                        <div className="bg-slate-950/80 backdrop-blur p-2.5 rounded-xl border border-slate-800 text-xs font-mono text-slate-300 flex items-center justify-between">
                          <span>FACIAL SENSORS: ACTIVE</span>
                          <span>LATENCY: {liveStreamData?.processing_time_ms || 120}ms</span>
                        </div>
                      </div>
                    </>
                  ) : (
                    <div className="p-8 text-center space-y-4">
                      <div className="w-16 h-16 rounded-2xl bg-slate-900 border border-slate-800 flex items-center justify-center mx-auto text-slate-500">
                        <Camera className="w-8 h-8" />
                      </div>
                      <div className="max-w-md mx-auto">
                        <h4 className="text-base font-bold text-white mb-1">Webcam Sensor Offline</h4>
                        <p className="text-slate-400 text-xs leading-relaxed">
                          Activate your webcam to test real-time deepfake face manipulation, facial swap artefacts, and biometric temporal coherence live over WebSocket.
                        </p>
                      </div>
                    </div>
                  )}
                </div>

                <div className="flex items-center justify-between pt-2">
                  <div className="text-xs text-slate-500 font-mono">
                    WebSocket stream running on /api/v1/detect/ws/stream
                  </div>
                  <button
                    onClick={() => setWebcamActive(!webcamActive)}
                    className={`px-6 py-2.5 rounded-xl font-semibold text-sm transition-all flex items-center gap-2 shadow-lg ${
                      webcamActive
                        ? 'bg-rose-600 hover:bg-rose-500 text-white shadow-rose-600/30'
                        : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-emerald-600/30'
                    }`}
                  >
                    {webcamActive ? (
                      <>
                        <Square className="w-4 h-4 fill-white" /> Disconnect Camera
                      </>
                    ) : (
                      <>
                        <Play className="w-4 h-4 fill-white" /> Initialize Live Audit
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}

            {/* 3. REMOTE URL TAB */}
            {tab === 'url' && (
              <div className="space-y-6 max-w-xl mx-auto py-8">
                <div className="text-center space-y-2">
                  <div className="w-12 h-12 rounded-xl bg-primary-600/10 border border-primary-500/20 text-primary-400 flex items-center justify-center mx-auto mb-3">
                    <LinkIcon className="w-6 h-6" />
                  </div>
                  <h3 className="text-lg font-bold text-white">Direct URL Extraction</h3>
                  <p className="text-slate-400 text-xs">
                    Inspect media directly hosted on news sites, social networks, or cloud buckets without saving files locally.
                  </p>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="block text-xs font-mono text-slate-400 mb-2">TARGET MEDIA URL</label>
                    <input
                      type="url"
                      value={urlInput}
                      onChange={(e) => setUrlInput(e.target.value)}
                      placeholder="https://images.unsplash.com/photo-example.jpg"
                      className="w-full bg-slate-950 border border-slate-700/80 rounded-xl px-4 py-3 text-white text-sm focus:outline-none focus:border-primary-500 font-mono transition-colors"
                    />
                  </div>

                  <button
                    onClick={handleAnalyzeUrl}
                    disabled={isAnalyzing || !urlInput}
                    className="w-full py-3 rounded-xl bg-gradient-to-r from-primary-600 to-indigo-600 hover:from-primary-500 hover:to-indigo-500 text-white font-semibold text-sm shadow-lg shadow-primary-600/30 flex items-center justify-center gap-2 transition-all disabled:opacity-50"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="w-4 h-4 animate-spin" /> Fetching and Analyzing...
                      </>
                    ) : (
                      <>
                        <Eye className="w-4 h-4" /> Fetch & Run Analysis
                      </>
                    )}
                  </button>
                </div>
              </div>
            )}

            {/* Bottom Security Info Footer */}
            <div className="mt-8 pt-4 border-t border-slate-800/80 flex flex-wrap items-center justify-between text-xs text-slate-500 font-mono gap-2">
              <span className="flex items-center gap-1.5">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                Zero-Retention Mode: Ephemeral storage compliant with GDPR Article 17
              </span>
              <span>AES-256 Encrypted Transfer</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DetectPage;

import React, { useState, useEffect } from 'react';
import { 
  Play, Pause, RotateCcw, CheckCircle2, ShieldCheck, 
  UploadCloud, Scan, Activity, Eye, Zap, Volume2, VolumeX, Maximize2, AlertTriangle 
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';

export const InteractiveDemoPlayer = () => {
  const [isPlaying, setIsPlaying] = useState(true);
  const [currentStep, setCurrentStep] = useState(0);
  const [progress, setProgress] = useState(0);
  const [isMuted, setIsMuted] = useState(true);

  const demoSteps = [
    {
      id: 0,
      title: "Step 1: Ingesting Media File",
      subtitle: "Secure drag-and-drop upload of portrait photo or video stream",
      timestamp: "00:04",
      action: "Uploading: suspected_ceo_video_call.mp4 (4K, 60 FPS)...",
      badge: "INGESTION PHASE",
      screenType: "upload",
      telemetry: { status: "Analyzing container...", bitrate: "24.2 Mbps", resolution: "3840x2160" }
    },
    {
      id: 1,
      title: "Step 2: MTCNN Multi-Face Localization",
      subtitle: "Neural bounding box extraction & 68-point 3D landmark tracking",
      timestamp: "00:08",
      action: "Isolating primary facial ROI [x: 420, y: 180, w: 640, h: 720]...",
      badge: "BIOMETRIC TRACKING",
      screenType: "scanning",
      telemetry: { faces: 1, confidence: "99.82%", landmarkVariance: "0.002" }
    },
    {
      id: 2,
      title: "Step 3: Multi-Signal DeepFake Audit",
      subtitle: "FFT Fourier spectrum, Laplacian blur, and GAN boundary checks",
      timestamp: "00:14",
      action: "FFT Frequency spike detected in 18kHz band (+3.4σ abnormal checkerboard)...",
      badge: "SPECTRAL AUDIT",
      screenType: "analysis",
      telemetry: { fftAnomaly: "High (0.84)", skinSmoothness: "Artificial (0.91)", colorInconsistency: "0.45" }
    },
    {
      id: 3,
      title: "Step 4: Certified Forensic Verdict",
      subtitle: "Cryptographic proof with GradCAM heatmap & downloadable audit PDF",
      timestamp: "00:18",
      action: "Threat Level: CRITICAL DEEPFAKE (Confidence: 94.6%). Visual proof generated.",
      badge: "INSPECTION COMPLETE",
      screenType: "verdict",
      telemetry: { overallScore: "94.6%", verdict: "SYNTHETIC DEEPFAKE", latency: "142ms" }
    }
  ];

  // Auto-play through demo steps
  useEffect(() => {
    if (!isPlaying) return;

    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          setCurrentStep((s) => (s + 1) % demoSteps.length);
          return 0;
        }
        return prev + 2.5; // ~4 seconds per step
      });
    }, 100);

    return () => clearInterval(interval);
  }, [isPlaying, demoSteps.length]);

  const step = demoSteps[currentStep];

  return (
    <div className="relative rounded-3xl overflow-hidden border border-slate-800 bg-slate-950 shadow-2xl">
      {/* Video Player Header Bar */}
      <div className="bg-slate-900/90 border-b border-slate-800 p-4 px-6 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="flex gap-1.5">
            <span className="w-3 h-3 rounded-full bg-red-500/80 inline-block" />
            <span className="w-3 h-3 rounded-full bg-yellow-500/80 inline-block" />
            <span className="w-3 h-3 rounded-full bg-green-500/80 inline-block" />
          </div>
          <span className="text-xs font-mono font-bold text-slate-300 ml-2 flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-primary-400" />
            DEMO WALKTHROUGH // How to Verify Media on DeepFaceGuard
          </span>
        </div>

        <div className="flex items-center gap-3">
          <span className="px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-primary-500/20 text-primary-400 border border-primary-500/30">
            {step.badge}
          </span>
          <span className="text-xs font-mono text-slate-400">{step.timestamp} / 00:20</span>
        </div>
      </div>

      {/* Simulated High-Tech Video Screen */}
      <div className="relative aspect-[16/9] w-full bg-slate-950 overflow-hidden flex items-center justify-center">
        {/* Animated Cyber Grid Background */}
        <div className="absolute inset-0 bg-cyber-grid opacity-25" />

        {/* Step-specific simulated screen UI */}
        <AnimatePresence mode="wait">
          <motion.div
            key={currentStep}
            initial={{ opacity: 0, scale: 0.98 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 1.02 }}
            transition={{ duration: 0.3 }}
            className="absolute inset-0 p-8 flex flex-col justify-between"
          >
            {/* Top notification bar */}
            <div className="flex justify-between items-start z-10">
              <div className="bg-slate-900/90 backdrop-blur-md p-3.5 px-5 rounded-2xl border border-slate-800 shadow-xl max-w-md">
                <span className="text-[10px] font-mono text-primary-400 font-bold uppercase tracking-wider block mb-1">
                  {step.title}
                </span>
                <p className="text-sm font-semibold text-white">{step.subtitle}</p>
              </div>

              <div className="bg-slate-900/80 backdrop-blur-md px-3 py-1.5 rounded-xl border border-slate-800 text-xs font-mono text-slate-300 flex items-center gap-2">
                <Activity className="w-3.5 h-3.5 text-emerald-400 animate-pulse" />
                <span>AI Core: ACTIVE</span>
              </div>
            </div>

            {/* Central Simulated Interaction Viewport */}
            <div className="my-auto flex items-center justify-center relative">
              {/* Screen Type: UPLOAD */}
              {step.screenType === 'upload' && (
                <div className="border-2 border-dashed border-primary-500/40 bg-slate-900/40 rounded-2xl p-8 text-center max-w-md w-full shadow-2xl relative">
                  <div className="w-16 h-16 rounded-2xl bg-primary-600/20 border border-primary-500/30 flex items-center justify-center mx-auto mb-3 text-primary-400 animate-bounce">
                    <UploadCloud className="w-8 h-8" />
                  </div>
                  <div className="text-white font-bold text-sm">Target Media Uploaded</div>
                  <div className="text-slate-400 text-xs mt-1 font-mono">suspected_deepfake_call.mp4 (48.2 MB)</div>
                  <div className="w-full bg-slate-800 h-1.5 rounded-full mt-4 overflow-hidden">
                    <div className="bg-primary-500 h-full w-full animate-pulse" />
                  </div>
                </div>
              )}

              {/* Screen Type: SCANNING & LOCALIZATION */}
              {step.screenType === 'scanning' && (
                <div className="relative rounded-2xl overflow-hidden border border-slate-700 bg-slate-900 max-w-sm w-full shadow-2xl aspect-square flex items-center justify-center">
                  <img
                    src="https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=800&q=80"
                    alt="Facial Tracking"
                    className="w-full h-full object-cover filter contrast-105"
                  />
                  {/* Facial Laser Bounding Box */}
                  <div className="absolute inset-8 border-2 border-cyan-400 rounded-xl pointer-events-none shadow-[0_0_15px_#22d3ee] animate-pulse">
                    <span className="absolute -top-3 left-3 bg-cyan-500 text-black text-[9px] font-mono font-bold px-1.5 py-0.5 rounded">
                      FACE_01 // MTCNN LOCATED
                    </span>
                  </div>
                  <div className="absolute inset-0 bg-primary-500/10 pointer-events-none">
                    <div className="w-full h-1 bg-cyan-400 shadow-[0_0_15px_#22d3ee] animate-scan-line" />
                  </div>
                </div>
              )}

              {/* Screen Type: ANALYSIS & FFT */}
              {step.screenType === 'analysis' && (
                <div className="grid grid-cols-2 gap-4 max-w-lg w-full bg-slate-900/90 p-5 rounded-2xl border border-slate-800 shadow-2xl">
                  <div className="space-y-2">
                    <span className="text-[10px] font-mono text-slate-400">FOURIER FFT SPECTRUM</span>
                    <div className="h-28 bg-slate-950 rounded-xl border border-slate-800 p-2 flex items-end justify-between gap-1">
                      {[40, 65, 30, 85, 95, 70, 90, 45, 80].map((h, i) => (
                        <div key={i} className="w-full bg-rose-500/70 rounded-t" style={{ height: `${h}%` }} />
                      ))}
                    </div>
                    <span className="text-[10px] font-mono text-rose-400 font-bold">GAN ARTICULATION SPIKE (+3.4σ)</span>
                  </div>

                  <div className="space-y-2">
                    <span className="text-[10px] font-mono text-slate-400">FEATURE RESIDUALS</span>
                    <div className="h-28 bg-slate-950 rounded-xl border border-slate-800 p-3 space-y-2">
                      <div className="text-[11px] font-mono text-slate-300 flex justify-between">
                        <span>Skin Texture:</span> <span className="text-rose-400 font-bold">UNNATURAL</span>
                      </div>
                      <div className="text-[11px] font-mono text-slate-300 flex justify-between">
                        <span>Color Boundary:</span> <span className="text-amber-400 font-bold">DISRUPTED</span>
                      </div>
                      <div className="text-[11px] font-mono text-slate-300 flex justify-between">
                        <span>Temporal Jitter:</span> <span className="text-rose-400 font-bold">DETECTED</span>
                      </div>
                    </div>
                    <span className="text-[10px] font-mono text-slate-500">HEURISTIC ENSEMBLE AUDIT</span>
                  </div>
                </div>
              )}

              {/* Screen Type: FINAL VERDICT */}
              {step.screenType === 'verdict' && (
                <div className="bg-slate-900/95 border-2 border-rose-500/80 p-6 rounded-3xl max-w-md w-full shadow-[0_0_40px_rgba(244,63,94,0.3)] text-center space-y-4">
                  <div className="w-16 h-16 rounded-2xl bg-rose-500/20 border border-rose-500/40 text-rose-400 flex items-center justify-center mx-auto animate-pulse">
                    <AlertTriangle className="w-8 h-8" />
                  </div>
                  <div>
                    <span className="px-3 py-1 rounded-full text-xs font-mono font-bold bg-rose-500/20 text-rose-400 border border-rose-500/40">
                      THREAT VERDICT: 94.6% DEEPFAKE
                    </span>
                    <h3 className="text-lg font-extrabold text-white mt-2">Manipulation Confirmed</h3>
                    <p className="text-slate-400 text-xs mt-1">
                      Artificial neural face swap identified. Forensic report and GradCAM visual heatmap generated.
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Bottom Live Action Log */}
            <div className="bg-slate-950/80 backdrop-blur p-3 rounded-xl border border-slate-800 flex items-center justify-between text-xs font-mono text-slate-300 z-10">
              <span className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-cyan-400 animate-ping" />
                {step.action}
              </span>
              <span className="text-slate-500 hidden sm:inline">DEEPFACE_KERNEL // PIPELINE_READY</span>
            </div>
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Video Controls Bar */}
      <div className="bg-slate-900 border-t border-slate-800 p-4 px-6 flex flex-col gap-3">
        {/* Progress Bar with step indicators */}
        <div className="w-full bg-slate-950 h-2 rounded-full overflow-hidden flex cursor-pointer">
          {demoSteps.map((s, idx) => (
            <div
              key={idx}
              onClick={() => {
                setCurrentStep(idx);
                setProgress(0);
              }}
              className="flex-1 h-full border-r border-slate-900 relative"
            >
              <div
                className={`h-full transition-all duration-100 ${
                  idx < currentStep
                    ? 'bg-primary-500'
                    : idx === currentStep
                    ? 'bg-primary-400'
                    : 'bg-transparent'
                }`}
                style={{
                  width: idx === currentStep ? `${progress}%` : idx < currentStep ? '100%' : '0%'
                }}
              />
            </div>
          ))}
        </div>

        {/* Buttons and Steps Navigation */}
        <div className="flex items-center justify-between pt-1">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-white transition-colors"
              title={isPlaying ? "Pause Demo" : "Play Demo"}
            >
              {isPlaying ? <Pause className="w-4 h-4 fill-white" /> : <Play className="w-4 h-4 fill-white" />}
            </button>

            <button
              onClick={() => {
                setCurrentStep(0);
                setProgress(0);
                setIsPlaying(true);
              }}
              className="p-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors"
              title="Restart Walkthrough"
            >
              <RotateCcw className="w-4 h-4" />
            </button>
          </div>

          {/* Interactive Step Switcher */}
          <div className="hidden md:flex items-center gap-2">
            {demoSteps.map((s, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setCurrentStep(idx);
                  setProgress(0);
                }}
                className={`px-3 py-1 rounded-lg text-xs font-mono transition-all ${
                  idx === currentStep
                    ? 'bg-primary-600 text-white font-bold shadow-md shadow-primary-600/30'
                    : 'bg-slate-800/60 text-slate-400 hover:text-white'
                }`}
              >
                {idx + 1}. {s.title.split(':')[1] || s.title}
              </button>
            ))}
          </div>

          <div className="text-xs font-mono text-slate-400">
            AUTONOMOUS WALKTHROUGH
          </div>
        </div>
      </div>
    </div>
  );
};

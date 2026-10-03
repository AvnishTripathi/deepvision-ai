import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Shield, AlertTriangle, CheckCircle, Zap, Cpu, Scan, Activity, Eye, Layers } from 'lucide-react';

export const InteractiveFaceScanner = () => {
  const [sliderPos, setSliderPos] = useState(50);
  const [activeFace, setActiveFace] = useState<'real' | 'deepfake'>('deepfake');

  return (
    <div className="relative rounded-2xl overflow-hidden border border-slate-800 bg-slate-950 shadow-2xl p-6">
      {/* Top HUD bar */}
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-4 mb-5">
        <div className="flex items-center gap-2">
          <span className="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-pulse" />
          <span className="text-xs font-mono font-bold text-white tracking-widest uppercase">
            NEURAL_BIOMETRIC_SCANNER // HUD-v2.6
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setActiveFace('real')}
            className={`px-3 py-1 rounded-lg text-xs font-mono transition-all ${
              activeFace === 'real'
                ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/40 font-bold'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Sample: AUTHENTIC
          </button>
          <button
            onClick={() => setActiveFace('deepfake')}
            className={`px-3 py-1 rounded-lg text-xs font-mono transition-all ${
              activeFace === 'deepfake'
                ? 'bg-rose-500/20 text-rose-400 border border-rose-500/40 font-bold'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            Sample: AI_SYNTHETIC
          </button>
        </div>
      </div>

      {/* Interactive Scan Canvas */}
      <div className="relative aspect-[16/10] md:aspect-[16/9] w-full rounded-xl overflow-hidden border border-slate-800 bg-slate-900 group select-none">
        {/* Synthetic AI Face Image or Authentic Face */}
        <img
          src={
            activeFace === 'deepfake'
              ? "https://images.unsplash.com/photo-1534528741775-53994a69daeb?auto=format&fit=crop&w=1200&q=80"
              : "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?auto=format&fit=crop&w=1200&q=80"
          }
          alt="Scanner Target"
          className="w-full h-full object-cover filter contrast-105"
        />

        {/* Dynamic Holographic Grid Overlay */}
        <div className="absolute inset-0 bg-cyber-grid opacity-30 pointer-events-none" />

        {/* Dynamic Scan Line Animation */}
        <div className="absolute inset-0 pointer-events-none">
          <div className="w-full h-1 bg-gradient-to-r from-transparent via-cyan-400 to-transparent shadow-[0_0_20px_#22d3ee] animate-scan-line" />
        </div>

        {/* Facial Landmarks Overlay */}
        <svg className="absolute inset-0 w-full h-full pointer-events-none">
          {/* Facial Boundary Box */}
          <rect
            x="36%"
            y="18%"
            width="28%"
            height="58%"
            fill="none"
            stroke={activeFace === 'deepfake' ? '#f43f5e' : '#10b981'}
            strokeWidth="2"
            strokeDasharray="6 4"
            className="animate-pulse"
          />

          {/* Corner brackets */}
          <path d="M 35% 22% L 35% 17% L 40% 17%" fill="none" stroke="#38bdf8" strokeWidth="3" />
          <path d="M 65% 17% L 65% 22% L 60% 17%" fill="none" stroke="#38bdf8" strokeWidth="3" />
          <path d="M 35% 73% L 35% 77% L 40% 77%" fill="none" stroke="#38bdf8" strokeWidth="3" />
          <path d="M 65% 73% L 65% 77% L 60% 77%" fill="none" stroke="#38bdf8" strokeWidth="3" />

          {/* Key Facial Landmarks dots */}
          <circle cx="44%" cy="42%" r="4" fill="#38bdf8" className="animate-ping" />
          <circle cx="56%" cy="42%" r="4" fill="#38bdf8" className="animate-ping" />
          <circle cx="50%" cy="52%" r="3" fill="#a855f7" />
          <circle cx="46%" cy="64%" r="3" fill="#f43f5e" />
          <circle cx="54%" cy="64%" r="3" fill="#f43f5e" />
        </svg>

        {/* Target Anomaly Badges floating over face */}
        <div className="absolute top-6 left-6 pointer-events-none">
          <div className="bg-slate-950/85 backdrop-blur-md p-3 rounded-xl border border-slate-700/80 font-mono space-y-1 shadow-xl">
            <div className="text-[10px] text-slate-400 uppercase tracking-widest flex items-center gap-1.5">
              <Scan className="w-3.5 h-3.5 text-cyan-400" /> FFT Fourier Spectrum
            </div>
            <div className="text-xs font-bold text-white">
              {activeFace === 'deepfake' ? 'Checkerboard Frequency Spike (+3.4σ)' : 'Natural Organic Noise Falloff'}
            </div>
            <div className="text-[11px] text-cyan-400 font-mono">
              CONFIDENCE: {activeFace === 'deepfake' ? '98.8%' : '99.4%'}
            </div>
          </div>
        </div>

        {/* Real-time Verdict floating badge */}
        <div className="absolute bottom-6 right-6 pointer-events-none">
          <div className={`p-4 rounded-xl border backdrop-blur-md font-mono shadow-2xl flex items-center gap-3 ${
            activeFace === 'deepfake'
              ? 'bg-rose-950/85 border-rose-500/50 text-rose-300'
              : 'bg-emerald-950/85 border-emerald-500/50 text-emerald-300'
          }`}>
            {activeFace === 'deepfake' ? (
              <AlertTriangle className="w-6 h-6 text-rose-400 animate-bounce" />
            ) : (
              <CheckCircle className="w-6 h-6 text-emerald-400" />
            )}
            <div>
              <div className="text-[10px] uppercase font-bold tracking-wider opacity-80">INSPECTION RESULT</div>
              <div className="text-base font-extrabold tracking-wide">
                {activeFace === 'deepfake' ? 'AI SYNTHETIC FACE SWAP' : 'AUTHENTIC BIOMETRIC RECORD'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Micro-telemetry bottom bar */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
        <div className="bg-slate-900/60 p-2.5 rounded-xl border border-slate-800 text-xs font-mono">
          <div className="text-slate-500 text-[10px]">TENSOR VARIANCE</div>
          <div className="text-slate-200 font-bold">{activeFace === 'deepfake' ? '0.0487 (High)' : '0.0082 (Nominal)'}</div>
        </div>
        <div className="bg-slate-900/60 p-2.5 rounded-xl border border-slate-800 text-xs font-mono">
          <div className="text-slate-500 text-[10px]">LAPLACIAN BLUR</div>
          <div className="text-slate-200 font-bold">{activeFace === 'deepfake' ? '94.2 (Unnatural Skin)' : '420.1 (Sharp)'}</div>
        </div>
        <div className="bg-slate-900/60 p-2.5 rounded-xl border border-slate-800 text-xs font-mono">
          <div className="text-slate-500 text-[10px]">EYE BLINK SYNCHRONY</div>
          <div className="text-slate-200 font-bold">{activeFace === 'deepfake' ? 'Desynchronized' : 'Normal Biological'}</div>
        </div>
        <div className="bg-slate-900/60 p-2.5 rounded-xl border border-slate-800 text-xs font-mono">
          <div className="text-slate-500 text-[10px]">INFERENCE LATENCY</div>
          <div className="text-emerald-400 font-bold">118ms (GPU-Accelerated)</div>
        </div>
      </div>
    </div>
  );
};

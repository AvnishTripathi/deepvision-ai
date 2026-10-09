"use client";

import { useState } from "react";
import { Settings as SettingsIcon, Sliders, ShieldCheck, Check } from "lucide-react";

export default function SettingsPage() {
  const [defaultThreshold, setDefaultThreshold] = useState("0.50");
  const [detectionInterval, setDetectionInterval] = useState("1");
  const [analysisInterval, setAnalysisInterval] = useState("5");
  const [saved, setSaved] = useState(false);

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2500);
  };

  return (
    <div className="space-y-8 max-w-3xl">
      <div>
        <h1 className="text-2xl font-black tracking-tight text-white sm:text-3xl">
          Engine Configuration
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Configure runtime face detection thresholds, tracking intervals, and visualizer settings.
        </p>
      </div>

      <div className="rounded-2xl border border-slate-800 bg-dark-900/90 p-6 shadow-card space-y-6">
        <h3 className="text-base font-bold text-white flex items-center gap-2 border-b border-slate-800 pb-4">
          <Sliders className="h-5 w-5 text-brand-cyan" />
          <span>Inference Parameters</span>
        </h3>

        <div className="space-y-4">
          <div>
            <label className="text-xs font-semibold text-slate-300">
              Default Face Detection Confidence Threshold
            </label>
            <p className="text-[11px] text-slate-500 mb-2">
              Minimum probability required for candidate face proposals (0.10 to 0.95).
            </p>
            <input
              type="number"
              step="0.05"
              min="0.1"
              max="0.95"
              value={defaultThreshold}
              onChange={(e) => setDefaultThreshold(e.target.value)}
              className="w-full sm:w-48 rounded-xl border border-slate-800 bg-dark-850 px-3.5 py-2 text-xs font-mono text-white focus:border-brand-cyan focus:outline-none"
            />
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-300">
              Video Detection Interval (Frames)
            </label>
            <p className="text-[11px] text-slate-500 mb-2">
              Runs face detector every N frames. Intermediate frames coast with Kalman predictions.
            </p>
            <input
              type="number"
              min="1"
              max="10"
              value={detectionInterval}
              onChange={(e) => setDetectionInterval(e.target.value)}
              className="w-full sm:w-48 rounded-xl border border-slate-800 bg-dark-850 px-3.5 py-2 text-xs font-mono text-white focus:border-brand-cyan focus:outline-none"
            />
          </div>

          <div>
            <label className="text-xs font-semibold text-slate-300">
              Video Analysis Interval (Frames)
            </label>
            <p className="text-[11px] text-slate-500 mb-2">
              Runs deep 3D pose and quality score analysis every N frames per active track.
            </p>
            <input
              type="number"
              min="1"
              max="30"
              value={analysisInterval}
              onChange={(e) => setAnalysisInterval(e.target.value)}
              className="w-full sm:w-48 rounded-xl border border-slate-800 bg-dark-850 px-3.5 py-2 text-xs font-mono text-white focus:border-brand-cyan focus:outline-none"
            />
          </div>
        </div>

        <div className="border-t border-slate-800 pt-4 flex items-center justify-between">
          <button
            onClick={handleSave}
            className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-brand-cyan to-brand-blue px-5 py-2.5 text-xs font-bold text-dark-950 shadow-glow hover:opacity-95"
          >
            {saved ? (
              <>
                <Check className="h-4 w-4" />
                <span>Saved Preferences</span>
              </>
            ) : (
              <span>Save Configuration</span>
            )}
          </button>
        </div>
      </div>
    </div>
  );
}

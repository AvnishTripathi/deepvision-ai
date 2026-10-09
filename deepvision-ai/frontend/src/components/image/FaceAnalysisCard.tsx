"use client";

import { useState } from "react";
import {
  StructuredFaceAnalysis,
} from "@/types";
import {
  CheckCircle2,
  AlertTriangle,
  Compass,
  Eye,
  Maximize2,
  Sun,
  Contrast,
  Sparkles,
  ChevronDown,
  ChevronUp,
} from "lucide-react";

interface FaceAnalysisCardProps {
  face: StructuredFaceAnalysis;
  faceIndex: number;
}

export default function FaceAnalysisCard({
  face,
  faceIndex,
}: FaceAnalysisCardProps) {
  const [showLandmarks, setShowLandmarks] = useState(false);

  const confPercent = Math.round(face.confidence * 100);
  const qualityScore = face.quality?.quality_score ?? face.quality_score ?? 0;
  const qualityGrade = face.quality?.grade || face.classification || "GOOD";
  const isUsable = face.quality?.is_usable ?? true;

  // Sharpness formatting
  const blurStatus = face.blur?.blur_status || face.clarity?.blur_status || "sharp";
  const sharpnessLabel =
    blurStatus === "sharp"
      ? "Excellent"
      : blurStatus === "slightly_blurred"
      ? "Fair (Mild Blur)"
      : "Blurry";

  // Brightness formatting
  const brightnessStatus =
    face.brightness?.brightness_status ||
    face.illumination?.brightness_status ||
    "acceptable";
  const brightnessLabel =
    brightnessStatus === "acceptable"
      ? "Good (Optimal)"
      : brightnessStatus === "too_dark"
      ? "Underexposed"
      : "Overexposed";

  // Contrast formatting
  const contrastStatus = face.contrast?.contrast_status || "acceptable";
  const contrastLabel =
    contrastStatus === "acceptable"
      ? "Good"
      : contrastStatus === "low_contrast"
      ? "Low Contrast"
      : "Harsh Contrast";

  // Size formatting
  const sizeCat = face.size?.size_category || "MEDIUM";
  const sizeLabel =
    sizeCat === "LARGE"
      ? "Large"
      : sizeCat === "MEDIUM"
      ? "Good (Medium)"
      : sizeCat === "SMALL"
      ? "Small"
      : "Tiny";

  // Pose formatting
  const pose = face.pose;
  const yaw = pose ? (pose.yaw >= 0 ? `+${pose.yaw.toFixed(1)}°` : `${pose.yaw.toFixed(1)}°`) : "N/A";
  const pitch = pose ? (pose.pitch >= 0 ? `+${pose.pitch.toFixed(1)}°` : `${pose.pitch.toFixed(1)}°`) : "N/A";
  const roll = pose ? (pose.roll >= 0 ? `+${pose.roll.toFixed(1)}°` : `${pose.roll.toFixed(1)}°`) : "N/A";

  return (
    <div className="rounded-2xl border border-slate-800 bg-dark-900/90 p-6 shadow-card space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800/80 pb-4">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-xl font-bold tracking-tight text-white">
              FACE #{faceIndex + 1}
            </h3>
            <span
              className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${
                isUsable
                  ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                  : "bg-amber-500/10 text-amber-400 border border-amber-500/20"
              }`}
            >
              {isUsable ? "Biometrically Usable" : "Quality Warning"}
            </span>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Bounding Box: {Math.round(face.bbox.x1)}, {Math.round(face.bbox.y1)} to {Math.round(face.bbox.x2)}, {Math.round(face.bbox.y2)}
          </p>
        </div>

        {/* Confidence Badge */}
        <div className="text-right">
          <p className="text-[11px] font-medium text-slate-400 uppercase tracking-wider">
            Confidence
          </p>
          <p className="text-2xl font-black text-brand-cyan">
            {confPercent}%
          </p>
        </div>
      </div>

      {/* Quality Score Bar */}
      <div className="rounded-xl border border-slate-800/80 bg-dark-850/80 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-purple-400" />
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              Quality Score
            </span>
          </div>
          <div className="flex items-baseline gap-1.5">
            <span className="text-2xl font-extrabold text-white">
              {Math.round(qualityScore)}
            </span>
            <span className="text-xs text-slate-400">/ 100</span>
            <span className="ml-2 text-xs font-semibold text-emerald-400 capitalize">
              ({qualityGrade.toLowerCase()})
            </span>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-dark-700">
          <div
            className={`h-full rounded-full transition-all duration-500 ${
              qualityScore >= 80
                ? "bg-gradient-to-r from-brand-cyan to-emerald-400"
                : qualityScore >= 60
                ? "bg-gradient-to-r from-brand-cyan to-brand-blue"
                : "bg-gradient-to-r from-amber-500 to-rose-500"
            }`}
            style={{ width: `${Math.min(100, Math.max(0, qualityScore))}%` }}
          />
        </div>
      </div>

      {/* Optical Diagnostics Grid */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 text-xs">
        {/* Sharpness */}
        <div className="rounded-xl border border-slate-800/70 bg-dark-850/50 p-3">
          <div className="flex items-center gap-1.5 text-slate-400 mb-1">
            <Eye className="h-3.5 w-3.5 text-brand-cyan" />
            <span>Sharpness</span>
          </div>
          <p className="font-semibold text-white">{sharpnessLabel}</p>
          {face.blur?.blur_score !== undefined && (
            <p className="text-[10px] text-slate-500 mt-0.5">
              Score: {face.blur.blur_score.toFixed(1)}
            </p>
          )}
        </div>

        {/* Brightness */}
        <div className="rounded-xl border border-slate-800/70 bg-dark-850/50 p-3">
          <div className="flex items-center gap-1.5 text-slate-400 mb-1">
            <Sun className="h-3.5 w-3.5 text-amber-400" />
            <span>Brightness</span>
          </div>
          <p className="font-semibold text-white">{brightnessLabel}</p>
          {face.brightness?.brightness_score !== undefined && (
            <p className="text-[10px] text-slate-500 mt-0.5">
              Score: {face.brightness.brightness_score.toFixed(1)}
            </p>
          )}
        </div>

        {/* Contrast */}
        <div className="rounded-xl border border-slate-800/70 bg-dark-850/50 p-3">
          <div className="flex items-center gap-1.5 text-slate-400 mb-1">
            <Contrast className="h-3.5 w-3.5 text-purple-400" />
            <span>Contrast</span>
          </div>
          <p className="font-semibold text-white">{contrastLabel}</p>
          {face.contrast?.contrast_score !== undefined && (
            <p className="text-[10px] text-slate-500 mt-0.5">
              Score: {face.contrast.contrast_score.toFixed(1)}
            </p>
          )}
        </div>

        {/* Face Size */}
        <div className="rounded-xl border border-slate-800/70 bg-dark-850/50 p-3">
          <div className="flex items-center gap-1.5 text-slate-400 mb-1">
            <Maximize2 className="h-3.5 w-3.5 text-emerald-400" />
            <span>Face Size</span>
          </div>
          <p className="font-semibold text-white">{sizeLabel}</p>
          <p className="text-[10px] text-slate-500 mt-0.5">
            {face.size?.width && face.size?.height
              ? `${Math.round(face.size.width)} × ${Math.round(face.size.height)} px`
              : face.bbox?.width && face.bbox?.height
              ? `${Math.round(face.bbox.width)} × ${Math.round(face.bbox.height)} px`
              : "—"}
          </p>
        </div>
      </div>

      {/* 3D Head Pose Panel */}
      <div className="rounded-xl border border-slate-800/80 bg-dark-850/70 p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Compass className="h-4 w-4 text-brand-cyan" />
            <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
              3D Head Pose Orientation
            </span>
          </div>
          {pose?.pose_category && (
            <span className="rounded bg-brand-cyan/15 px-2 py-0.5 text-[11px] font-medium text-brand-cyan">
              {pose.pose_category}
            </span>
          )}
        </div>

        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="rounded-lg bg-dark-900/80 p-2.5 border border-slate-800/60">
            <p className="text-[11px] text-slate-400">Yaw</p>
            <p className="text-base font-mono font-bold text-white mt-0.5">{yaw}</p>
          </div>
          <div className="rounded-lg bg-dark-900/80 p-2.5 border border-slate-800/60">
            <p className="text-[11px] text-slate-400">Pitch</p>
            <p className="text-base font-mono font-bold text-white mt-0.5">{pitch}</p>
          </div>
          <div className="rounded-lg bg-dark-900/80 p-2.5 border border-slate-800/60">
            <p className="text-[11px] text-slate-400">Roll</p>
            <p className="text-base font-mono font-bold text-white mt-0.5">{roll}</p>
          </div>
        </div>
      </div>

      {/* Landmarks Coordinates Toggle */}
      {face.landmarks && (
        <div className="border-t border-slate-800/80 pt-3">
          <button
            onClick={() => setShowLandmarks(!showLandmarks)}
            className="flex w-full items-center justify-between text-xs font-medium text-slate-400 hover:text-slate-200"
          >
            <span>5 Canonical Subpixel Landmarks</span>
            {showLandmarks ? (
              <ChevronUp className="h-4 w-4" />
            ) : (
              <ChevronDown className="h-4 w-4" />
            )}
          </button>

          {showLandmarks && (
            <div className="mt-3 grid grid-cols-2 gap-2 text-[11px] font-mono text-slate-300 bg-dark-950/80 p-3 rounded-xl border border-slate-800">
              {face.landmarks.right_eye && (
                <div>Right Eye: ({Math.round(face.landmarks.right_eye.x)}, {Math.round(face.landmarks.right_eye.y)})</div>
              )}
              {face.landmarks.left_eye && (
                <div>Left Eye: ({Math.round(face.landmarks.left_eye.x)}, {Math.round(face.landmarks.left_eye.y)})</div>
              )}
              {face.landmarks.nose_tip && (
                <div>Nose Tip: ({Math.round(face.landmarks.nose_tip.x)}, {Math.round(face.landmarks.nose_tip.y)})</div>
              )}
              {face.landmarks.mouth_right && (
                <div>Mouth Right: ({Math.round(face.landmarks.mouth_right.x)}, {Math.round(face.landmarks.mouth_right.y)})</div>
              )}
              {face.landmarks.mouth_left && (
                <div>Mouth Left: ({Math.round(face.landmarks.mouth_left.x)}, {Math.round(face.landmarks.mouth_left.y)})</div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
}

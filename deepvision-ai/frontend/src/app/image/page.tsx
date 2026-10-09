"use client";

import { useState, useRef, ChangeEvent } from "react";
import {
  UploadCloud,
  Image as ImageIcon,
  Trash2,
  ScanFace,
  Sliders,
  CheckCircle2,
  AlertCircle,
  Loader2,
  Layers,
  Sparkles,
} from "lucide-react";
import { analyzeImage } from "@/lib/api";
import { AIAnalysisResponse } from "@/types";
import FaceVisualizerCanvas from "@/components/image/FaceVisualizerCanvas";
import FaceAnalysisCard from "@/components/image/FaceAnalysisCard";

export default function ImageAnalysisPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [confidenceThreshold, setConfidenceThreshold] = useState<number>(0.5);

  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [loadingStage, setLoadingStage] = useState<string>("Analyzing image...");
  const [analysisResult, setAnalysisResult] = useState<AIAnalysisResponse | null>(null);
  const [selectedFaceIndex, setSelectedFaceIndex] = useState<number>(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // File selection & validation
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const processFile = (file: File) => {
    const validTypes = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
    if (!validTypes.includes(file.type)) {
      setErrorMessage("Unsupported format. Please upload JPG, PNG, WebP, or BMP.");
      return;
    }

    if (file.size > 15 * 1024 * 1024) {
      setErrorMessage("File exceeds 15 MB limit.");
      return;
    }

    setErrorMessage(null);
    setSelectedFile(file);
    setAnalysisResult(null);
    setSelectedFaceIndex(0);

    const reader = new FileReader();
    reader.onload = () => {
      setPreviewUrl(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleClear = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setAnalysisResult(null);
    setErrorMessage(null);
    setSelectedFaceIndex(0);
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  // Run Real Backend Analysis
  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setIsAnalyzing(true);
    setErrorMessage(null);
    setLoadingStage("Analyzing image...");

    // Simulated realistic stage feedback during HTTP request
    const t1 = setTimeout(() => setLoadingStage("Detecting faces with YuNet..."), 300);
    const t2 = setTimeout(() => setLoadingStage("Calculating landmarks & 3D pose..."), 700);
    const t3 = setTimeout(() => setLoadingStage("Evaluating quality score & clarity..."), 1100);

    try {
      const result = await analyzeImage(selectedFile, confidenceThreshold);
      setAnalysisResult(result);
      setSelectedFaceIndex(0);

      // Record to session storage for dashboard
      try {
        const curImg = parseInt(sessionStorage.getItem("dv_images_analyzed") || "0", 10) + 1;
        const curFaces = parseInt(sessionStorage.getItem("dv_faces_detected") || "0", 10) + (result.face_count || 0);
        sessionStorage.setItem("dv_images_analyzed", curImg.toString());
        sessionStorage.setItem("dv_faces_detected", curFaces.toString());
      } catch {
        // Safe fallback
      }
    } catch (err: any) {
      setErrorMessage(err.message || "An unexpected error occurred during face analysis.");
    } finally {
      clearTimeout(t1);
      clearTimeout(t2);
      clearTimeout(t3);
      setIsAnalyzing(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Page Title & Subtitle */}
      <div>
        <h1 className="text-2xl font-black tracking-tight text-white sm:text-3xl">
          Image Face Analysis
        </h1>
        <p className="mt-1 text-sm text-slate-400">
          Upload portrait or group photography for multi-face detection, 5-point landmarks, 3D head pose, and quality analytics.
        </p>
      </div>

      {/* Main Analysis Section */}
      <div className="grid grid-cols-1 gap-8 lg:grid-cols-12">
        {/* Left Column: Dropzone & Image Canvas (7 cols on desktop) */}
        <div className="space-y-6 lg:col-span-7">
          {!previewUrl ? (
            /* Upload Dropzone */
            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              className="flex min-h-[360px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed border-slate-700/80 bg-dark-900/60 p-8 text-center transition-all hover:border-brand-cyan/50 hover:bg-dark-850/80 group"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp,image/bmp"
                onChange={handleFileChange}
                className="hidden"
              />
              <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-brand-cyan/10 text-brand-cyan transition-transform group-hover:scale-110 shadow-glow">
                <UploadCloud className="h-8 w-8" />
              </div>
              <h3 className="mt-4 text-base font-semibold text-white">
                Drag and drop an image, or click to browse
              </h3>
              <p className="mt-1.5 text-xs text-slate-400">
                Supports JPG, PNG, WebP, and BMP (up to 15 MB)
              </p>
              <div className="mt-5 inline-flex items-center gap-2 rounded-xl bg-dark-800 px-4 py-2 text-xs font-medium text-slate-300 border border-slate-700">
                <ImageIcon className="h-3.5 w-3.5 text-brand-cyan" />
                <span>Select from computer</span>
              </div>
            </div>
          ) : (
            /* Active Image Display */
            <div className="space-y-4">
              {/* Visualizer Canvas or Preview */}
              {analysisResult && previewUrl ? (
                <FaceVisualizerCanvas
                  imageUrl={previewUrl}
                  imageWidth={analysisResult.image_width}
                  imageHeight={analysisResult.image_height}
                  faces={analysisResult.faces}
                  selectedFaceIndex={selectedFaceIndex}
                  onSelectFace={setSelectedFaceIndex}
                />
              ) : (
                <div className="relative overflow-hidden rounded-2xl border border-slate-800 bg-dark-900 shadow-card">
                  <img
                    src={previewUrl}
                    alt="Uploaded preview"
                    className="block w-full max-h-[500px] object-contain"
                  />
                  {isAnalyzing && (
                    <div className="absolute inset-0 flex flex-col items-center justify-center bg-dark-950/80 backdrop-blur-sm">
                      <div className="scanline" />
                      <Loader2 className="h-10 w-10 animate-spin text-brand-cyan" />
                      <p className="mt-4 text-sm font-semibold text-white">
                        {loadingStage}
                      </p>
                      <p className="text-xs text-slate-400 mt-1">
                        Communicating with YuNet neural engine
                      </p>
                    </div>
                  )}
                </div>
              )}

              {/* Action Toolbar */}
              <div className="flex flex-wrap items-center justify-between gap-4 rounded-xl border border-slate-800/80 bg-dark-900/90 p-4">
                <div className="flex items-center gap-2">
                  <button
                    onClick={handleClear}
                    disabled={isAnalyzing}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-dark-800 px-3 py-2 text-xs font-semibold text-rose-400 hover:bg-dark-700 disabled:opacity-50"
                  >
                    <Trash2 className="h-3.5 w-3.5" />
                    <span>Remove</span>
                  </button>
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={isAnalyzing}
                    className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 bg-dark-800 px-3 py-2 text-xs font-semibold text-slate-300 hover:bg-dark-700 disabled:opacity-50"
                  >
                    <ImageIcon className="h-3.5 w-3.5" />
                    <span>Replace Image</span>
                  </button>
                </div>

                {/* Threshold Slider & Analyze Trigger */}
                <div className="flex items-center gap-4">
                  <div className="hidden sm:flex items-center gap-2 text-xs text-slate-400">
                    <Sliders className="h-3.5 w-3.5 text-brand-cyan" />
                    <span>Threshold:</span>
                    <input
                      type="range"
                      min="0.1"
                      max="0.95"
                      step="0.05"
                      value={confidenceThreshold}
                      onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                      disabled={isAnalyzing}
                      className="w-20 accent-brand-cyan"
                    />
                    <span className="font-mono text-white">
                      {Math.round(confidenceThreshold * 100)}%
                    </span>
                  </div>

                  <button
                    onClick={handleAnalyze}
                    disabled={isAnalyzing}
                    className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-brand-cyan to-brand-blue px-5 py-2.5 text-xs font-bold text-dark-950 shadow-glow hover:opacity-95 disabled:opacity-50"
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="h-4 w-4 animate-spin" />
                        <span>Analyzing...</span>
                      </>
                    ) : (
                      <>
                        <ScanFace className="h-4 w-4" />
                        <span>Analyze Face</span>
                      </>
                    )}
                  </button>
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

        {/* Right Column: Face Analysis Results Panel (5 cols on desktop) */}
        <div className="space-y-6 lg:col-span-5">
          {analysisResult ? (
            <div className="space-y-6">
              {/* Summary Header */}
              <div className="rounded-2xl border border-slate-800 bg-dark-900/90 p-5 shadow-card">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Layers className="h-5 w-5 text-brand-cyan" />
                    <span className="text-base font-bold text-white">
                      Faces Detected: {analysisResult.face_count}
                    </span>
                  </div>
                  <span className="rounded-md bg-dark-800 px-2.5 py-1 text-xs font-mono text-slate-300">
                    {analysisResult.processing_time_ms.toFixed(1)} ms
                  </span>
                </div>

                <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-slate-400">
                  <span>Model: <strong className="text-slate-200">{analysisResult.model}</strong></span>
                  <span>•</span>
                  <span>Device: <strong className="text-slate-200">{analysisResult.device}</strong></span>
                  <span>•</span>
                  <span>Resolution: <strong className="text-slate-200">{analysisResult.image_width}×{analysisResult.image_height}</strong></span>
                </div>
              </div>

              {/* Multi-Face Tab Navigation */}
              {analysisResult.face_count > 0 ? (
                <>
                  {analysisResult.face_count > 1 && (
                    <div className="flex items-center gap-2 overflow-x-auto pb-2">
                      {analysisResult.faces.map((f, idx) => (
                        <button
                          key={idx}
                          onClick={() => setSelectedFaceIndex(idx)}
                          className={`flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-semibold whitespace-nowrap transition-all ${
                            idx === selectedFaceIndex
                              ? "bg-brand-cyan text-dark-950 shadow-glow"
                              : "border border-slate-800 bg-dark-850 text-slate-300 hover:border-slate-700"
                          }`}
                        >
                          <span>Face {idx + 1}</span>
                          <span
                            className={`rounded px-1.5 py-0.5 text-[10px] ${
                              idx === selectedFaceIndex
                                ? "bg-dark-950/20 text-dark-950"
                                : "bg-dark-700 text-slate-300"
                            }`}
                          >
                            {Math.round(f.confidence * 100)}%
                          </span>
                        </button>
                      ))}
                    </div>
                  )}

                  {/* Active Face Detailed Analysis Card */}
                  {analysisResult.faces[selectedFaceIndex] && (
                    <FaceAnalysisCard
                      face={analysisResult.faces[selectedFaceIndex]}
                      faceIndex={selectedFaceIndex}
                    />
                  )}
                </>
              ) : (
                /* No Faces Found Empty State */
                <div className="rounded-2xl border border-slate-800/80 bg-dark-900/60 p-8 text-center text-slate-400">
                  <div className="mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-slate-800 text-slate-400">
                    <ScanFace className="h-6 w-6" />
                  </div>
                  <h4 className="mt-4 text-sm font-semibold text-white">
                    Zero Faces Detected
                  </h4>
                  <p className="mt-1 text-xs text-slate-400">
                    No human faces met the {Math.round(confidenceThreshold * 100)}% confidence threshold.
                  </p>
                </div>
              )}
            </div>
          ) : (
            /* Awaiting Analysis Prompt */
            <div className="rounded-2xl border border-slate-800/80 bg-dark-900/40 p-8 text-center text-slate-400 space-y-3">
              <div className="mx-auto flex h-14 w-14 items-center justify-center rounded-2xl bg-dark-850 border border-slate-800 text-slate-400">
                <ScanFace className="h-7 w-7 text-brand-cyan" />
              </div>
              <h4 className="text-base font-bold text-white">
                Face Analytics Panel
              </h4>
              <p className="text-xs text-slate-400 max-w-sm mx-auto leading-relaxed">
                Upload a photograph and click <strong>Analyze Face</strong> to extract real facial bounding boxes, 5-point landmarks, 3D Euler angles, and quality scores directly from the backend.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

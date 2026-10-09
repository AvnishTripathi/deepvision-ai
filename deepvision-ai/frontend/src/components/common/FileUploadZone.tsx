"use client";

import { useRef, useState, DragEvent, ChangeEvent } from "react";
import {
  UploadCloud,
  FileCheck,
  AlertCircle,
  X,
  File,
  Loader2,
  FileVideo,
  FileImage,
} from "lucide-react";

export interface FileUploadZoneProps {
  mediaType: "image" | "video";
  acceptedExtensions: string[];
  maxSizeMB: number;
  selectedFile: File | null;
  isLoading?: boolean;
  loadingMessage?: string;
  onFileSelect: (file: File) => void;
  onClear: () => void;
  errorMessage?: string | null;
  onDismissError?: () => void;
}

export default function FileUploadZone({
  mediaType,
  acceptedExtensions,
  maxSizeMB,
  selectedFile,
  isLoading = false,
  loadingMessage,
  onFileSelect,
  onClear,
  errorMessage,
  onDismissError,
}: FileUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    if (!isLoading) {
      setIsDragging(true);
    }
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const validateAndDispatch = (file: File) => {
    const fileName = file.name.toLowerCase();
    const ext = fileName.substring(fileName.lastIndexOf("."));

    // Check extension
    if (!acceptedExtensions.some((acc) => acc.toLowerCase() === ext)) {
      throw new Error(
        `Invalid file format. Supported extensions: ${acceptedExtensions.join(", ")}`
      );
    }

    // Check size limit
    const fileSizeMB = file.size / (1024 * 1024);
    if (fileSizeMB > maxSizeMB) {
      throw new Error(
        `File is too large (${fileSizeMB.toFixed(1)} MB). Maximum allowed size is ${maxSizeMB} MB.`
      );
    }

    onFileSelect(file);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (isLoading) return;

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      try {
        validateAndDispatch(e.dataTransfer.files[0]);
      } catch (err: unknown) {
        if (err instanceof Error && onDismissError) {
          // Parent handles error or alerts
        }
      }
    }
  };

  const handleInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      try {
        validateAndDispatch(e.target.files[0]);
      } catch {
        // Handled
      }
    }
  };

  const acceptString = acceptedExtensions.join(",");

  return (
    <div className="space-y-3">
      {!selectedFile ? (
        /* Empty Upload Dropzone */
        <div
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
          onClick={() => !isLoading && fileInputRef.current?.click()}
          className={`relative flex min-h-[300px] cursor-pointer flex-col items-center justify-center rounded-2xl border-2 border-dashed p-8 text-center transition-all ${
            isDragging
              ? "border-brand-cyan bg-brand-cyan/5 shadow-glow"
              : "border-slate-700/80 bg-dark-900/60 hover:border-brand-cyan/40 hover:bg-dark-850/70"
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept={acceptString}
            onChange={handleInputChange}
            disabled={isLoading}
            className="hidden"
          />

          <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-brand-cyan/10 text-brand-cyan transition-transform group-hover:scale-105 shadow-glow">
            <UploadCloud className="h-8 w-8" />
          </div>

          <h3 className="mt-4 text-base font-semibold text-white">
            Drag and drop your {mediaType}, or{" "}
            <span className="text-brand-cyan underline underline-offset-4">browse files</span>
          </h3>

          <p className="mt-1.5 text-xs text-slate-400">
            Formats: <strong className="text-slate-300">{acceptedExtensions.join(", ").toUpperCase()}</strong> • Max:{" "}
            <strong className="text-slate-300">{maxSizeMB} MB</strong>
          </p>

          <div className="mt-5 inline-flex items-center gap-2 rounded-xl border border-slate-700 bg-dark-800 px-4 py-2 text-xs font-medium text-slate-300">
            {mediaType === "video" ? (
              <FileVideo className="h-3.5 w-3.5 text-brand-blue" />
            ) : (
              <FileImage className="h-3.5 w-3.5 text-brand-cyan" />
            )}
            <span>Select from device</span>
          </div>
        </div>
      ) : (
        /* Selected File Card */
        <div className="flex items-center justify-between rounded-xl border border-slate-800 bg-dark-900/90 p-4 shadow-card">
          <div className="flex items-center gap-3 truncate">
            <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-brand-cyan/10 text-brand-cyan shrink-0">
              {mediaType === "video" ? (
                <FileVideo className="h-5 w-5 text-brand-blue" />
              ) : (
                <FileImage className="h-5 w-5 text-brand-cyan" />
              )}
            </div>

            <div className="truncate">
              <p className="text-xs font-bold text-white truncate max-w-xs sm:max-w-md">
                {selectedFile.name}
              </p>
              <p className="text-[11px] text-slate-400 mt-0.5">
                {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB • {selectedFile.type || "binary"}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {!isLoading && (
              <button
                type="button"
                onClick={onClear}
                title="Remove file"
                className="flex h-8 w-8 items-center justify-center rounded-lg border border-slate-700 bg-dark-800 text-slate-400 hover:text-rose-400 hover:border-rose-500/40"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>
      )}

      {/* Loading Overlay / Progress Message */}
      {isLoading && (
        <div className="flex items-center gap-3 rounded-xl border border-brand-cyan/30 bg-brand-cyan/10 p-3 text-xs text-brand-cyan">
          <Loader2 className="h-4 w-4 animate-spin shrink-0" />
          <span>{loadingMessage || "Processing file with AI model..."}</span>
        </div>
      )}

      {/* Validation Error Banner */}
      {errorMessage && (
        <div className="flex items-center justify-between gap-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3.5 text-xs text-rose-400">
          <div className="flex items-center gap-2">
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{errorMessage}</span>
          </div>
          {onDismissError && (
            <button
              onClick={onDismissError}
              className="text-rose-400 hover:text-white"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      )}
    </div>
  );
}

"use client";

import React, { useState, useRef } from "react";
import {
  UploadCloud,
  FileImage,
  FileVideo,
  X,
  AlertCircle,
  Loader2,
  CheckCircle2,
} from "lucide-react";

interface FileUploadZoneProps {
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
  loadingMessage = "Processing...",
  onFileSelect,
  onClear,
  errorMessage,
  onDismissError,
}: FileUploadZoneProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const validateAndSelect = (file: File) => {
    setValidationError(null);
    if (onDismissError) onDismissError();

    // Check extension
    const ext = `.${file.name.split(".").pop()?.toLowerCase()}`;
    if (!acceptedExtensions.includes(ext)) {
      setValidationError(
        `Invalid file type "${ext}". Supported formats: ${acceptedExtensions.join(", ")}`
      );
      return;
    }

    // Check size cap
    const maxBytes = maxSizeMB * 1024 * 1024;
    if (file.size > maxBytes) {
      setValidationError(
        `File size (${(file.size / (1024 * 1024)).toFixed(1)} MB) exceeds maximum allowed ${maxSizeMB} MB.`
      );
      return;
    }

    onFileSelect(file);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      validateAndSelect(e.dataTransfer.files[0]);
    }
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      validateAndSelect(e.target.files[0]);
    }
  };

  const activeError = validationError || errorMessage;

  const formatSize = (bytes: number) => {
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  };

  return (
    <div className="space-y-3">
      {/* Drop Zone Box */}
      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !isLoading && !selectedFile && inputRef.current?.click()}
        className={`relative flex min-h-[220px] flex-col items-center justify-center rounded-2xl border-2 border-dashed p-6 text-center transition-all ${
          isDragging
            ? "border-brand-cyan bg-brand-cyan/10 scale-[1.005]"
            : selectedFile
            ? "border-slate-700 bg-dark-900/90"
            : "border-slate-800 bg-dark-900/60 hover:border-slate-700 hover:bg-dark-900 cursor-pointer"
        }`}
      >
        <input
          ref={inputRef}
          type="file"
          accept={acceptedExtensions.join(",")}
          onChange={handleInputChange}
          className="hidden"
          disabled={isLoading}
        />

        {isLoading ? (
          <div className="flex flex-col items-center gap-3 py-6 text-center">
            <Loader2 className="h-10 w-10 animate-spin text-brand-cyan" />
            <p className="text-sm font-semibold text-white">{loadingMessage}</p>
            <p className="text-xs text-slate-400">Executing neural pipeline...</p>
          </div>
        ) : selectedFile ? (
          <div className="flex w-full max-w-md items-center justify-between rounded-xl border border-slate-800 bg-dark-850 p-4">
            <div className="flex items-center gap-3 overflow-hidden">
              <div className="flex h-11 w-11 shrink-0 items-center justify-center rounded-xl bg-brand-cyan/10 text-brand-cyan">
                {mediaType === "image" ? (
                  <FileImage className="h-6 w-6" />
                ) : (
                  <FileVideo className="h-6 w-6" />
                )}
              </div>
              <div className="overflow-hidden text-left">
                <p className="truncate text-sm font-semibold text-white">
                  {selectedFile.name}
                </p>
                <p className="text-xs text-slate-400">
                  {formatSize(selectedFile.size)}
                </p>
              </div>
            </div>

            <button
              onClick={(e) => {
                e.stopPropagation();
                if (inputRef.current) inputRef.current.value = "";
                onClear();
              }}
              aria-label="Remove selected file"
              className="flex h-8 w-8 items-center justify-center rounded-lg text-slate-400 transition-colors hover:bg-slate-800 hover:text-white focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-cyan"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2 py-4">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-dark-850 border border-slate-800 text-brand-cyan">
              <UploadCloud className="h-7 w-7" />
            </div>
            <p className="mt-2 text-sm font-bold text-white">
              Drag & drop {mediaType === "image" ? "an image" : "a video"} here, or{" "}
              <span className="text-brand-cyan underline decoration-brand-cyan/40">browse</span>
            </p>
            <p className="text-xs text-slate-400">
              Supports {acceptedExtensions.join(", ")} up to {maxSizeMB} MB
            </p>
          </div>
        )}
      </div>

      {/* Validation or API Error Banner */}
      {activeError && (
        <div
          role="alert"
          className="flex items-center justify-between gap-3 rounded-xl border border-rose-500/30 bg-rose-500/10 p-3.5 text-xs text-rose-300"
        >
          <div className="flex items-center gap-2">
            <AlertCircle className="h-4 w-4 shrink-0 text-rose-400" />
            <span>{activeError}</span>
          </div>
          <button
            onClick={() => {
              setValidationError(null);
              if (onDismissError) onDismissError();
            }}
            aria-label="Dismiss error"
            className="text-rose-400 hover:text-white"
          >
            <X className="h-3.5 w-3.5" />
          </button>
        </div>
      )}
    </div>
  );
}


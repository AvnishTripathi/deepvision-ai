"use client";

import { useEffect, useRef, useState } from "react";
import { StructuredFaceAnalysis } from "@/types";

interface FaceVisualizerCanvasProps {
  imageUrl: string;
  imageWidth: number;
  imageHeight: number;
  faces: StructuredFaceAnalysis[];
  selectedFaceIndex: number;
  onSelectFace: (index: number) => void;
  showLandmarks?: boolean;
}

export default function FaceVisualizerCanvas({
  imageUrl,
  imageWidth,
  imageHeight,
  faces,
  selectedFaceIndex,
  onSelectFace,
  showLandmarks = true,
}: FaceVisualizerCanvasProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const imgRef = useRef<HTMLImageElement | null>(null);

  useEffect(() => {
    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
      imgRef.current = img;
      setImageLoaded(true);
    };
  }, [imageUrl]);

  useEffect(() => {
    if (!imageLoaded || !imgRef.current || !canvasRef.current || !containerRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const containerWidth = containerRef.current.clientWidth;
    // Calculate display scale preserving aspect ratio
    const scale = containerWidth / imageWidth;
    const displayWidth = containerWidth;
    const displayHeight = imageHeight * scale;

    // Handle high-DPI displays
    const dpr = window.devicePixelRatio || 1;
    canvas.width = displayWidth * dpr;
    canvas.height = displayHeight * dpr;
    canvas.style.width = `${displayWidth}px`;
    canvas.style.height = `${displayHeight}px`;

    ctx.scale(dpr, dpr);

    // 1. Draw base image
    ctx.drawImage(imgRef.current, 0, 0, displayWidth, displayHeight);

    // 2. Draw face overlays
    faces.forEach((face, idx) => {
      const isSelected = idx === selectedFaceIndex;
      const bbox = face.bbox;

      // Scale coordinates from native image coordinates to canvas display coordinates
      const x1 = bbox.x1 * scale;
      const y1 = bbox.y1 * scale;
      const x2 = bbox.x2 * scale;
      const y2 = bbox.y2 * scale;
      const boxW = x2 - x1;
      const boxH = y2 - y1;

      // Styling based on selection
      const strokeColor = isSelected ? "#06b6d4" : "rgba(34, 211, 238, 0.65)";
      const fillColor = isSelected ? "rgba(6, 182, 212, 0.12)" : "rgba(6, 182, 212, 0.04)";

      // Draw bounding box
      ctx.fillStyle = fillColor;
      ctx.fillRect(x1, y1, boxW, boxH);

      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = isSelected ? 2.5 : 1.5;
      ctx.strokeRect(x1, y1, boxW, boxH);

      // Draw cyber corner accents
      const cornerLen = Math.min(16, boxW / 4, boxH / 4);
      ctx.strokeStyle = isSelected ? "#38bdf8" : "#22d3ee";
      ctx.lineWidth = isSelected ? 3.5 : 2.5;

      // Top-Left
      ctx.beginPath();
      ctx.moveTo(x1, y1 + cornerLen);
      ctx.lineTo(x1, y1);
      ctx.lineTo(x1 + cornerLen, y1);
      ctx.stroke();

      // Top-Right
      ctx.beginPath();
      ctx.moveTo(x2 - cornerLen, y1);
      ctx.lineTo(x2, y1);
      ctx.lineTo(x2, y1 + cornerLen);
      ctx.stroke();

      // Bottom-Left
      ctx.beginPath();
      ctx.moveTo(x1, y2 - cornerLen);
      ctx.lineTo(x1, y2);
      ctx.lineTo(x1 + cornerLen, y2);
      ctx.stroke();

      // Bottom-Right
      ctx.beginPath();
      ctx.moveTo(x2 - cornerLen, y2);
      ctx.lineTo(x2, y2);
      ctx.lineTo(x2, y2 - cornerLen);
      ctx.stroke();

      // Draw Header Badge: "Face #1   97%"
      const confPercent = Math.round(face.confidence * 100);
      const labelText = `Face #${idx + 1}  ${confPercent}%`;

      ctx.font = isSelected ? "bold 12px sans-serif" : "600 11px sans-serif";
      const textMetrics = ctx.measureText(labelText);
      const badgeW = textMetrics.width + 14;
      const badgeH = 22;
      const badgeY = Math.max(0, y1 - badgeH - 4);

      // Badge background
      ctx.fillStyle = isSelected ? "#06b6d4" : "rgba(11, 15, 25, 0.9)";
      ctx.strokeStyle = isSelected ? "#38bdf8" : "rgba(34, 211, 238, 0.5)";
      ctx.lineWidth = 1;

      ctx.beginPath();
      ctx.roundRect(x1, badgeY, badgeW, badgeH, 4);
      ctx.fill();
      ctx.stroke();

      // Badge text
      ctx.fillStyle = isSelected ? "#060911" : "#f8fafc";
      ctx.fillText(labelText, x1 + 7, badgeY + 15);

      // 3. Draw 5 Facial Landmarks if available
      if (showLandmarks && face.landmarks) {
        const lmPoints = [
          face.landmarks.right_eye,
          face.landmarks.left_eye,
          face.landmarks.nose_tip,
          face.landmarks.mouth_right,
          face.landmarks.mouth_left,
        ].filter(Boolean);

        lmPoints.forEach((pt, pIdx) => {
          if (!pt) return;
          const px = pt.x * scale;
          const py = pt.y * scale;

          ctx.beginPath();
          ctx.arc(px, py, isSelected ? 3.5 : 2.5, 0, Math.PI * 2);
          ctx.fillStyle = pIdx === 2 ? "#f59e0b" : "#10b981"; // Nose amber, eyes/mouth emerald
          ctx.fill();
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1;
          ctx.stroke();
        });
      }
    });
  }, [imageLoaded, imageUrl, imageWidth, imageHeight, faces, selectedFaceIndex, showLandmarks]);

  // Handle canvas click to select corresponding face
  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (!canvasRef.current || !containerRef.current) return;
    const rect = canvasRef.current.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const clickY = e.clientY - rect.top;

    const scale = containerRef.current.clientWidth / imageWidth;

    // Check hit test across faces
    for (let i = 0; i < faces.length; i++) {
      const bbox = faces[i].bbox;
      const x1 = bbox.x1 * scale;
      const y1 = bbox.y1 * scale;
      const x2 = bbox.x2 * scale;
      const y2 = bbox.y2 * scale;

      if (clickX >= x1 && clickX <= x2 && clickY >= y1 && clickY <= y2) {
        onSelectFace(i);
        return;
      }
    }
  };

  return (
    <div ref={containerRef} className="relative w-full overflow-hidden rounded-2xl border border-slate-800 bg-dark-900/90 shadow-card">
      <canvas
        ref={canvasRef}
        onClick={handleCanvasClick}
        className="block w-full cursor-pointer select-none"
      />
      {!imageLoaded && (
        <div className="flex h-80 items-center justify-center text-slate-400">
          <div className="h-6 w-6 animate-spin rounded-full border-2 border-brand-cyan border-t-transparent" />
        </div>
      )}
    </div>
  );
}

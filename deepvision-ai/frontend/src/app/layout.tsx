import type { Metadata } from "next";
import "./globals.css";
import LayoutShell from "@/components/LayoutShell";
import { ThemeProvider } from "@/context/ThemeContext";

export const metadata: Metadata = {
  title: "DeepVision AI — See. Detect. Track. Analyze. Understand.",
  description:
    "Enterprise AI Computer Vision platform for multi-face detection, ByteTrack tracking, 3D pose, and face quality analytics.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="antialiased selection:bg-brand-cyan/20 selection:text-brand-cyan transition-colors duration-200">
        <ThemeProvider>
          <LayoutShell>{children}</LayoutShell>
        </ThemeProvider>
      </body>
    </html>
  );
}


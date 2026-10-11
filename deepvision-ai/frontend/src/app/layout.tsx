import type { Metadata } from "next";
import "./globals.css";
import LayoutShell from "@/components/LayoutShell";
import { ThemeProvider } from "@/context/ThemeContext";

export const metadata: Metadata = {
  metadataBase: new URL(process.env.NEXT_PUBLIC_APP_URL || "http://localhost:3000"),
  title: "DeepVision AI — See. Detect. Track. Analyze. Understand.",
  description:
    "Enterprise AI Computer Vision platform for multi-face detection, ByteTrack tracking, 3D pose, and face quality analytics.",
  icons: {
    icon: [
      { url: "/favicon.ico", sizes: "any" },
      { url: "/logo.png", type: "image/png" },
    ],
    apple: [
      { url: "/apple-touch-icon.png", sizes: "180x180", type: "image/png" },
    ],
    shortcut: ["/logo.png"],
  },
  openGraph: {
    title: "DeepVision AI — See. Detect. Track. Analyze. Understand.",
    description:
      "Enterprise AI Computer Vision platform for multi-face detection, ByteTrack tracking, 3D pose, and face quality analytics.",
    images: [
      {
        url: "/logo.png",
        width: 1024,
        height: 1024,
        alt: "DeepVision AI Logo",
      },
    ],
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className="antialiased selection:bg-brand-cyan/20 selection:text-brand-cyan transition-colors duration-200">
        <ThemeProvider>
          <LayoutShell>{children}</LayoutShell>
        </ThemeProvider>
      </body>
    </html>
  );
}


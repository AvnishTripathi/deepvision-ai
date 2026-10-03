import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'DeepVision AI — Advanced AI-Powered Face Detection',
  description: 'Advanced AI-Powered Face Detection & Video Analysis Platform',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}


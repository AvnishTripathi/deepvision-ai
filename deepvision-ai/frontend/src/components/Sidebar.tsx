"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  ScanFace,
  Video,
  Camera,
  BarChart3,
  Settings,
  Sparkles,
  ChevronRight,
  ShieldCheck,
} from "lucide-react";

interface SidebarProps {
  isOpen: boolean;
  onClose?: () => void;
}

const navItems = [
  {
    label: "Dashboard",
    href: "/",
    icon: LayoutDashboard,
    badge: null,
  },
  {
    label: "Image Analysis",
    href: "/image",
    icon: ScanFace,
    badge: "AI 2.0",
  },
  {
    label: "Video & Tracking",
    href: "/video",
    icon: Video,
    badge: "ByteTrack",
  },
  {
    label: "Live Camera",
    href: "/camera",
    icon: Camera,
    badge: "Next Step",
  },
  {
    label: "Analytics",
    href: "/analytics",
    icon: BarChart3,
    badge: null,
  },
  {
    label: "Settings",
    href: "/settings",
    icon: Settings,
    badge: null,
  },
];

export default function Sidebar({ isOpen, onClose }: SidebarProps) {
  const pathname = usePathname();

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          onClick={onClose}
          className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm lg:hidden"
        />
      )}

      <aside
        className={`fixed top-0 bottom-0 left-0 z-50 flex w-72 flex-col border-r border-slate-800/80 bg-dark-900/95 backdrop-blur-xl transition-transform duration-300 ease-in-out lg:translate-x-0 ${
          isOpen ? "translate-x-0" : "-translate-x-full"
        }`}
      >
        {/* Brand Header */}
        <div className="flex h-20 items-center gap-3 border-b border-slate-800/80 px-6">
          <div className="flex h-11 w-11 items-center justify-center rounded-xl bg-gradient-to-tr from-brand-cyan via-brand-blue to-brand-violet p-0.5 shadow-glow">
            <div className="flex h-full w-full items-center justify-center rounded-[10px] bg-dark-950">
              <Sparkles className="h-5 w-5 text-brand-cyan" />
            </div>
          </div>
          <div>
            <div className="flex items-center gap-1.5">
              <span className="text-lg font-bold tracking-tight text-white">
                DeepVision
              </span>
              <span className="rounded bg-brand-cyan/20 px-1.5 py-0.5 text-[10px] font-semibold text-brand-cyan uppercase tracking-wider">
                AI
              </span>
            </div>
            <p className="text-[11px] font-medium text-slate-400">
              Enterprise Vision Platform
            </p>
          </div>
        </div>

        {/* Tagline Banner */}
        <div className="mx-4 my-3 rounded-lg border border-slate-800/60 bg-dark-850/50 p-2.5">
          <p className="text-center text-[10px] font-medium tracking-wide text-brand-cyan">
            See • Detect • Track • Analyze • Understand
          </p>
        </div>

        {/* Navigation Items */}
        <nav className="flex-1 space-y-1.5 overflow-y-auto px-4 py-2">
          {navItems.map((item) => {
            const isActive =
              item.href === "/"
                ? pathname === "/"
                : pathname?.startsWith(item.href);
            const Icon = item.icon;

            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={onClose}
                className={`group flex items-center justify-between rounded-xl px-3.5 py-3 text-sm font-medium transition-all ${
                  isActive
                    ? "bg-gradient-to-r from-brand-cyan/15 via-brand-blue/10 to-transparent text-brand-cyan border border-brand-cyan/30 shadow-sm"
                    : "text-slate-400 hover:bg-slate-800/50 hover:text-slate-200"
                }`}
              >
                <div className="flex items-center gap-3">
                  <Icon
                    className={`h-4.5 w-4.5 transition-colors ${
                      isActive
                        ? "text-brand-cyan"
                        : "text-slate-400 group-hover:text-slate-200"
                    }`}
                  />
                  <span>{item.label}</span>
                </div>

                <div className="flex items-center gap-1.5">
                  {item.badge && (
                    <span
                      className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${
                        item.badge === "ByteTrack"
                          ? "bg-emerald-500/10 text-emerald-400 border border-emerald-500/20"
                          : item.badge === "AI 2.0"
                          ? "bg-brand-violet/15 text-purple-300 border border-purple-500/20"
                          : "bg-slate-800 text-slate-400"
                      }`}
                    >
                      {item.badge}
                    </span>
                  )}
                  {isActive && (
                    <ChevronRight className="h-4 w-4 text-brand-cyan" />
                  )}
                </div>
              </Link>
            );
          })}
        </nav>

        {/* System Footnote */}
        <div className="border-t border-slate-800/80 p-4">
          <div className="flex items-center gap-3 rounded-xl border border-slate-800/60 bg-dark-850/80 p-3">
            <ShieldCheck className="h-5 w-5 text-emerald-400" />
            <div className="text-xs">
              <p className="font-semibold text-slate-200">YuNet + ByteTrack</p>
              <p className="text-[11px] text-slate-400">Zero Biometric Re-ID</p>
            </div>
          </div>
        </div>
      </aside>
    </>
  );
}

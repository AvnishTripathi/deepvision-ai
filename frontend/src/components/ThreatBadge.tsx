import React from 'react';
import { ShieldCheck, AlertTriangle, XCircle } from 'lucide-react';
import clsx from 'clsx';

type ThreatBadgeProps = {
  label: 'REAL' | 'SUSPICIOUS' | 'DEEPFAKE';
  size?: 'sm' | 'md' | 'lg';
};

export const ThreatBadge = ({ label, size = 'md' }: ThreatBadgeProps) => {
  const isDeepfake = label === 'DEEPFAKE';
  const isSuspicious = label === 'SUSPICIOUS';
  const isReal = label === 'REAL';

  return (
    <div
      className={clsx(
        'inline-flex items-center gap-1.5 font-semibold rounded-full border',
        {
          'px-2.5 py-0.5 text-xs': size === 'sm',
          'px-3 py-1 text-sm': size === 'md',
          'px-4 py-2 text-base': size === 'lg',
        },
        {
          'bg-danger-500/10 text-danger-500 border-danger-500/20 shadow-[0_0_10px_rgba(239,68,68,0.2)] animate-pulse-glow': isDeepfake,
          'bg-warning-500/10 text-warning-500 border-warning-500/20': isSuspicious,
          'bg-success-500/10 text-success-500 border-success-500/20': isReal,
        }
      )}
    >
      {isReal && <ShieldCheck className={clsx({'w-3 h-3': size==='sm', 'w-4 h-4': size==='md', 'w-5 h-5': size==='lg'})} />}
      {isSuspicious && <AlertTriangle className={clsx({'w-3 h-3': size==='sm', 'w-4 h-4': size==='md', 'w-5 h-5': size==='lg'})} />}
      {isDeepfake && <XCircle className={clsx({'w-3 h-3': size==='sm', 'w-4 h-4': size==='md', 'w-5 h-5': size==='lg'})} />}
      {label}
    </div>
  );
};

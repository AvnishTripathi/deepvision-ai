import React from 'react';
import { useAuthStore } from '../store/authStore';
import { ShieldAlert, Activity, Image as ImageIcon, Video, Eye } from 'lucide-react';
import { Link } from 'react-router-dom';
import { ThreatBadge } from '../components/ThreatBadge';

const DashboardPage = () => {
  const { user } = useAuthStore();

  // Mock data
  const stats = [
    { label: 'Total Scans', value: '1,248', icon: Activity, color: 'text-primary-500' },
    { label: 'Deepfakes Detected', value: '156', icon: ShieldAlert, color: 'text-danger-500' },
    { label: 'Average Risk Score', value: '14.2%', icon: Eye, color: 'text-warning-500' }
  ];

  const recent = [
    { id: 'job_123', type: 'image', date: '2 mins ago', score: 92, label: 'DEEPFAKE' as const },
    { id: 'job_124', type: 'video', date: '1 hr ago', score: 12, label: 'REAL' as const },
    { id: 'job_125', type: 'image', date: '3 hrs ago', score: 45, label: 'SUSPICIOUS' as const },
  ];

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-white">Welcome back, {user?.name || 'User'}</h1>
        <p className="text-slate-400">Here's an overview of your platform usage.</p>
      </div>

      <div className="grid md:grid-cols-3 gap-6 mb-8">
        {stats.map((stat, i) => (
          <div key={i} className="glass-card p-6 flex items-center gap-4">
            <div className={`p-4 rounded-xl bg-slate-800 ${stat.color}`}>
              <stat.icon className="w-8 h-8" />
            </div>
            <div>
              <div className="text-slate-400 text-sm font-medium">{stat.label}</div>
              <div className="text-3xl font-bold text-white mt-1">{stat.value}</div>
            </div>
          </div>
        ))}
      </div>

      <div className="glass-card overflow-hidden">
        <div className="p-6 border-b border-slate-700/50 flex justify-between items-center">
          <h2 className="text-xl font-bold text-white">Recent Detections</h2>
          <Link to="/detect" className="text-sm text-primary-400 hover:text-primary-300">New Scan &rarr;</Link>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm text-slate-400">
            <thead className="text-xs uppercase bg-slate-800/50 text-slate-300">
              <tr>
                <th className="px-6 py-4">Job ID</th>
                <th className="px-6 py-4">Type</th>
                <th className="px-6 py-4">Date</th>
                <th className="px-6 py-4">Score</th>
                <th className="px-6 py-4">Verdict</th>
                <th className="px-6 py-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-700/50">
              {recent.map((item, i) => (
                <tr key={i} className="hover:bg-slate-800/30 transition-colors">
                  <td className="px-6 py-4 font-mono text-slate-300">{item.id}</td>
                  <td className="px-6 py-4">
                    {item.type === 'image' ? <ImageIcon className="w-5 h-5 text-slate-500" /> : <Video className="w-5 h-5 text-slate-500" />}
                  </td>
                  <td className="px-6 py-4">{item.date}</td>
                  <td className="px-6 py-4 font-medium text-white">{item.score}%</td>
                  <td className="px-6 py-4">
                    <ThreatBadge label={item.label} size="sm" />
                  </td>
                  <td className="px-6 py-4 text-right">
                    <Link to={`/results/${item.id}`} className="text-primary-500 hover:text-primary-400 font-medium">View Report</Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage;

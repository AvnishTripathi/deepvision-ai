import React from 'react';
import { useAuthStore } from '../store/authStore';
import { ShieldAlert, Activity, Image as ImageIcon, Video, Eye, TrendingUp, BarChart2 } from 'lucide-react';
import { Link } from 'react-router-dom';
import { ThreatBadge } from '../components/ThreatBadge';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
} from 'recharts';

const DashboardPage = () => {
  const { user } = useAuthStore();

  // Mock data
  const stats = [
    { label: 'Total Scans', value: '1,248', icon: Activity, color: 'text-primary-500' },
    { label: 'Deepfakes Detected', value: '156', icon: ShieldAlert, color: 'text-danger-500' },
    { label: 'Average Risk Score', value: '14.2%', icon: Eye, color: 'text-warning-500' }
  ];

  // 7-Day Detection Activity Trend Data
  const activityData = [
    { day: 'Mon', totalScans: 142, realFaces: 125, deepfakes: 17, avgScore: 12 },
    { day: 'Tue', totalScans: 198, realFaces: 172, deepfakes: 26, avgScore: 15 },
    { day: 'Wed', totalScans: 165, realFaces: 145, deepfakes: 20, avgScore: 13 },
    { day: 'Thu', totalScans: 230, realFaces: 201, deepfakes: 29, avgScore: 16 },
    { day: 'Fri', totalScans: 285, realFaces: 249, deepfakes: 36, avgScore: 18 },
    { day: 'Sat', totalScans: 210, realFaces: 186, deepfakes: 24, avgScore: 14 },
    { day: 'Sun', totalScans: 248, realFaces: 218, deepfakes: 30, avgScore: 15 },
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
        <p className="text-slate-400">Here's an overview of your platform usage and detection analytics.</p>
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

      {/* Detection Activity & Threat Trends Graph */}
      <div className="glass-card p-6 mb-8 border border-slate-700/50">
        <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-2 mb-6">
          <div>
            <h2 className="text-xl font-bold text-white flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-primary-400" />
              Scan Activity & Threat Trends (Past 7 Days)
            </h2>
            <p className="text-slate-400 text-sm mt-0.5">
              Live volumetric scans breakdown comparing verified genuine media against flagged anomalies.
            </p>
          </div>
          <div className="flex items-center gap-2 text-xs font-medium text-slate-400 bg-slate-800/80 px-3 py-1.5 rounded-lg border border-slate-700">
            <span className="w-2 h-2 rounded-full bg-primary-400 animate-pulse" />
            Live Analytics
          </div>
        </div>

        <div className="h-72 w-full min-h-[280px]">
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={activityData} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="scansGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6366f1" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#6366f1" stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="deepfakeGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="realGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#22c55e" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.6} />
              <XAxis dataKey="day" stroke="#94a3b8" tick={{ fontSize: 12 }} />
              <YAxis stroke="#94a3b8" tick={{ fontSize: 12 }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#0f172a',
                  borderColor: '#334155',
                  borderRadius: '10px',
                  boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.5)',
                  color: '#fff',
                }}
                itemStyle={{ color: '#fff', fontSize: '13px' }}
              />
              <Legend verticalAlign="top" height={36} wrapperStyle={{ paddingBottom: '10px', fontSize: '12px' }} />
              <Area
                type="monotone"
                dataKey="totalScans"
                name="Total Scans"
                stroke="#6366f1"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#scansGradient)"
              />
              <Area
                type="monotone"
                dataKey="realFaces"
                name="Verified Real"
                stroke="#22c55e"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#realGradient)"
              />
              <Area
                type="monotone"
                dataKey="deepfakes"
                name="Deepfakes Flagged"
                stroke="#ef4444"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#deepfakeGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
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

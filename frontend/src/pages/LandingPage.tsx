import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Shield, Zap, CheckCircle, Video, Image as ImageIcon, Camera, Code, Brain } from 'lucide-react';
import { motion } from 'framer-motion';
import { TermsModal, PrivacyModal, ApiDocsModal } from '../components/FooterModals';
import { InteractiveFaceScanner } from '../components/InteractiveFaceScanner';
import { InteractiveDemoPlayer } from '../components/InteractiveDemoPlayer';

const LandingPage = () => {
  const [modalType, setModalType] = useState<'terms' | 'privacy' | 'api' | null>(null);
  return (
    <div className="bg-dark-bg min-h-screen text-slate-200">
      {/* Hero Section */}
      <div className="relative overflow-hidden pt-20 pb-32">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[1000px] h-[500px] bg-primary-600/20 blur-[120px] rounded-full pointer-events-none" />
        
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
          <div className="text-center max-w-4xl mx-auto">
            <motion.h1 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="text-5xl md:text-6xl font-extrabold tracking-tight mb-8"
            >
              Detect Deepfakes with <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-400 to-primary-600">Military-Grade AI</span>
            </motion.h1>
            <motion.p 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.1 }}
              className="text-xl text-slate-400 mb-10 max-w-2xl mx-auto"
            >
              Protect your digital identity and secure your platform with our advanced deepfake detection API. Real-time analysis for images and video streams.
            </motion.p>
            <motion.div 
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.2 }}
              className="flex justify-center gap-4 mb-16"
            >
              <Link to="/detect" className="bg-primary-600 hover:bg-primary-700 text-white px-8 py-3.5 rounded-xl font-semibold text-base transition-all shadow-xl shadow-primary-600/30 flex items-center gap-2">
                <Shield className="w-5 h-5" /> Start Detecting Free
              </Link>
              <button onClick={() => document.getElementById('scanner-demo')?.scrollIntoView({ behavior: 'smooth' })} className="glass hover:bg-slate-800 text-white px-8 py-3.5 rounded-xl font-semibold text-base transition-colors flex items-center gap-2">
                <Zap className="w-4 h-4 text-amber-400" /> Interactive Demo
              </button>
            </motion.div>

            {/* Interactive Face Scanner Showcase */}
            <motion.div
              id="scanner-demo"
              initial={{ opacity: 0, scale: 0.96 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.7, delay: 0.3 }}
              className="max-w-4xl mx-auto"
            >
              <InteractiveFaceScanner />
            </motion.div>
          </div>
        </div>
      </div>

      {/* Stats Bar */}
      <div className="border-y border-slate-800 bg-slate-900/50 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8 text-center">
            <div>
              <div className="text-3xl font-bold text-white mb-1">99.2%</div>
              <div className="text-slate-400 text-sm">Accuracy Rate</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-white mb-1">&lt;200ms</div>
              <div className="text-slate-400 text-sm">Detection Speed</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-white mb-1">5M+</div>
              <div className="text-slate-400 text-sm">Media Scanned</div>
            </div>
            <div>
              <div className="text-3xl font-bold text-white mb-1">GDPR</div>
              <div className="text-slate-400 text-sm">Compliant</div>
            </div>
          </div>
        </div>
      </div>

      {/* Features Grid */}
      <div className="py-24 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16 max-w-2xl mx-auto">
          <span className="text-xs font-mono font-bold text-primary-400 bg-primary-500/10 px-3 py-1 rounded-full border border-primary-500/20 uppercase tracking-wider">
            Multi-Modal Forensics
          </span>
          <h2 className="text-3xl md:text-4xl font-extrabold text-white mt-3 mb-4">
            Comprehensive Biometric Inspection Suite
          </h2>
          <p className="text-slate-400 text-sm">
            Trained on 1.4 million deepfake videos and photographs across 5 benchmark academic datasets.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[
            { 
              icon: ImageIcon, 
              title: 'Photo Detection', 
              badge: 'JPG / PNG / WEBP',
              image: 'https://images.unsplash.com/photo-1544005313-94ddf0286df2?auto=format&fit=crop&w=600&q=80',
              desc: 'Identifies GAN fingerprints, boundary warping, and skin pore smoothing artifacts in passport and profile photos.' 
            },
            { 
              icon: Video, 
              title: 'Video Forensics', 
              badge: '24-60 FPS Stream',
              image: 'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?auto=format&fit=crop&w=600&q=80',
              desc: 'Frame-by-frame temporal inspection targeting blink rate irregularities, facial twitching, and lighting coherence.' 
            },
            { 
              icon: Camera, 
              title: 'Live Camera KYC', 
              badge: '< 150ms WebSocket',
              image: 'https://images.unsplash.com/photo-1573496359142-b8d87734a5a2?auto=format&fit=crop&w=600&q=80',
              desc: 'Real-time anti-spoofing protection for automated user onboarding, biometric login, and virtual video calls.' 
            },
            { 
              icon: Code, 
              title: 'Enterprise API', 
              badge: 'REST + Python SDK',
              image: 'https://images.unsplash.com/photo-1526374965328-7f61d4dc18c5?auto=format&fit=crop&w=600&q=80',
              desc: 'Plug-and-play SDKs for Python, Node.js, and Go with automated webhooks and batch processing queues.' 
            },
          ].map((feat, i) => (
            <motion.div 
              key={i}
              whileHover={{ y: -6 }}
              transition={{ duration: 0.2 }}
              className="glass-card rounded-2xl overflow-hidden border border-slate-800 hover:border-primary-500/50 transition-all flex flex-col group"
            >
              <div className="relative h-44 overflow-hidden bg-slate-950">
                <img 
                  src={feat.image} 
                  alt={feat.title} 
                  className="w-full h-full object-cover group-hover:scale-105 transition-transform duration-500 filter brightness-90 contrast-105"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-slate-950 via-slate-950/40 to-transparent" />
                <span className="absolute top-3 right-3 text-[10px] font-mono bg-black/70 backdrop-blur px-2 py-0.5 rounded border border-slate-700 text-slate-300">
                  {feat.badge}
                </span>
                <div className="absolute bottom-3 left-3 p-2 rounded-xl bg-slate-900/80 backdrop-blur border border-slate-700/80 text-primary-400">
                  <feat.icon className="w-5 h-5" />
                </div>
              </div>

              <div className="p-5 flex-1 flex flex-col justify-between">
                <div>
                  <h3 className="text-lg font-bold text-white mb-2">{feat.title}</h3>
                  <p className="text-slate-400 text-xs leading-relaxed">{feat.desc}</p>
                </div>
                <div className="mt-4 pt-3 border-t border-slate-800/80 flex items-center justify-between text-[11px] font-mono text-primary-400">
                  <span>DEPLOYABLE TODAY</span>
                  <span>→</span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      </div>

      {/* How it works & Interactive Video Walkthrough */}
      <div id="demo" className="py-24 bg-slate-900/50 border-y border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12 max-w-2xl mx-auto">
            <span className="text-xs font-mono font-bold text-emerald-400 bg-emerald-500/10 px-3 py-1 rounded-full border border-emerald-500/20 uppercase tracking-wider">
              Visual Guided Walkthrough
            </span>
            <h2 className="text-3xl md:text-4xl font-extrabold text-white mt-3 mb-3">
              How DeepFaceGuard Operates
            </h2>
            <p className="text-slate-400 text-sm">
              Watch the automated forensic audit pipeline process high-definition video in under 200 milliseconds.
            </p>
          </div>

          {/* High-Tech Demo Video Simulation Player */}
          <div className="max-w-4xl mx-auto mb-20">
            <InteractiveDemoPlayer />
          </div>

          {/* 3 Steps Process Grid */}
          <div className="flex flex-col md:flex-row justify-center items-center gap-8">
            {[
              { step: '01', title: 'Media Ingestion', desc: 'Securely upload high-resolution images, stream 4K video, or paste remote URLs.' },
              { step: '02', title: 'Neural Analysis', desc: 'Ensemble models analyze Fourier frequency spectrums, blur variances, and GAN edges.' },
              { step: '03', title: 'Forensic Verdict', desc: 'Receive instant cryptographic confidence scores with GradCAM visual heatmaps.' },
            ].map((step, i) => (
              <div key={i} className="glass-card p-6 rounded-2xl border border-slate-800 flex-1 max-w-sm text-center relative group hover:border-primary-500/50 transition-all">
                <div className="w-14 h-14 rounded-2xl bg-primary-600/10 border border-primary-500/30 flex items-center justify-center text-primary-400 text-xl font-mono font-bold mx-auto mb-4 group-hover:scale-110 transition-transform">
                  {step.step}
                </div>
                <h3 className="text-lg font-bold text-white mb-2">{step.title}</h3>
                <p className="text-slate-400 text-xs leading-relaxed">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="bg-slate-950 py-12 border-t border-slate-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-2">
            <Brain className="h-6 w-6 text-primary-500" />
            <span className="text-lg font-bold text-white">DeepFaceGuard</span>
          </div>
          <p className="text-slate-500 text-sm">© 2026 DeepFaceGuard. All rights reserved.</p>
          <div className="flex gap-6">
            <button 
              onClick={() => setModalType('terms')} 
              className="text-slate-400 hover:text-white transition-colors text-sm underline-offset-4 hover:underline"
            >
              Terms
            </button>
            <button 
              onClick={() => setModalType('privacy')} 
              className="text-slate-400 hover:text-white transition-colors text-sm underline-offset-4 hover:underline"
            >
              Privacy
            </button>
            <button 
              onClick={() => setModalType('api')} 
              className="text-slate-400 hover:text-white transition-colors text-sm underline-offset-4 hover:underline"
            >
              API Docs
            </button>
          </div>
        </div>
      </footer>

      {/* Interactive Modals */}
      <TermsModal isOpen={modalType === 'terms'} onClose={() => setModalType(null)} />
      <PrivacyModal isOpen={modalType === 'privacy'} onClose={() => setModalType(null)} />
      <ApiDocsModal isOpen={modalType === 'api'} onClose={() => setModalType(null)} />
    </div>
  );
};

export default LandingPage;

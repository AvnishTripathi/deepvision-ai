import React, { useState } from 'react';
import { 
  FileText, Shield, Code, CheckCircle2, Copy, Check, ExternalLink, 
  Terminal, Lock, Eye, AlertCircle, Database, Cpu, Globe
} from 'lucide-react';

export const TermsModal = ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-3xl w-full max-h-[85vh] flex flex-col shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        <div className="p-6 border-b border-slate-800 flex items-center justify-between bg-slate-950/60">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-primary-500/10 text-primary-400 border border-primary-500/20">
              <FileText className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">Terms of Service & Forensic Usage</h2>
              <p className="text-xs text-slate-400">Effective Date: January 1, 2026 | Version 2.4-Enterprise</p>
            </div>
          </div>
          <button 
            onClick={onClose} 
            className="text-slate-400 hover:text-white p-2 rounded-lg hover:bg-slate-800 text-sm font-mono transition-colors"
          >
            ✕
          </button>
        </div>

        <div className="p-6 overflow-y-auto text-sm text-slate-300 space-y-5 font-sans leading-relaxed">
          <div className="p-4 rounded-xl bg-primary-950/30 border border-primary-800/40 text-xs text-primary-200 flex items-start gap-2.5">
            <AlertCircle className="w-4 h-4 text-primary-400 shrink-0 mt-0.5" />
            <span>
              DeepFaceGuard provides probabilistic cryptographic & heuristic deepfake analysis for journalistic, academic, security, and verification purposes.
            </span>
          </div>

          <section className="space-y-2">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              1. Acceptable Use & Ethical Standards
            </h3>
            <p className="text-xs text-slate-400">
              Users agree to submit only digital media for which they have lawful authorization to inspect. The platform must not be used for illicit surveillance, harassment, reverse-engineering biometric countermeasures, or infringing upon copyright protections.
            </p>
          </section>

          <section className="space-y-2">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              2. Forensic Accuracy & Legal Evidentiary Weight
            </h3>
            <p className="text-xs text-slate-400">
              DeepFaceGuard combines convolutional deep feature extraction (EfficientNet-B7) and 4-tier spatial/spectral Fourier heuristics. While models achieve benchmark detection rates of ~98% on benchmark corpuses (FaceForensics++, DFDC), outputs represent mathematical probabilities and should accompany expert witness testimony in formal judicial proceedings.
            </p>
          </section>

          <section className="space-y-2">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              3. API Rate Limiting & Programmatic Fair Use
            </h3>
            <p className="text-xs text-slate-400">
              Public API keys are subject to tier-based sliding-window rate limits (100 req/min for Enterprise tier, 10 req/min for Evaluation accounts). Automated continuous video scraping requires prior architectural clearance.
            </p>
          </section>

          <section className="space-y-2">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              4. Service Availability & SLA
            </h3>
            <p className="text-xs text-slate-400">
              DeepFaceGuard commits to a 99.9% uptime target for cloud inference nodes, excluding scheduled kernel maintenance and GPU cluster provisioning windows.
            </p>
          </section>
        </div>

        <div className="p-4 border-t border-slate-800 bg-slate-950/80 flex items-center justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl bg-primary-600 hover:bg-primary-500 text-white font-semibold text-xs transition-colors shadow-lg shadow-primary-600/20"
          >
            I Acknowledge & Agree
          </button>
        </div>
      </div>
    </div>
  );
};

export const PrivacyModal = ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => {
  if (!isOpen) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-3xl w-full max-h-[85vh] flex flex-col shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        <div className="p-6 border-b border-slate-800 flex items-center justify-between bg-slate-950/60">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
              <Shield className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">Privacy Policy & Biometric Zero-Retention</h2>
              <p className="text-xs text-slate-400">GDPR (EU) 2016/679 & CCPA Certified Architecture | 2026</p>
            </div>
          </div>
          <button onClick={onClose} className="text-slate-400 hover:text-white p-2 rounded-lg hover:bg-slate-800 text-sm font-mono transition-colors">
            ✕
          </button>
        </div>

        <div className="p-6 overflow-y-auto text-sm text-slate-300 space-y-5 leading-relaxed font-sans">
          <div className="p-4 rounded-xl bg-emerald-950/30 border border-emerald-800/40 text-xs text-emerald-200 flex items-start gap-2.5">
            <Lock className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
            <span>
              <strong>Zero-Retention Guarantee:</strong> Uploaded media files are processed in ephemeral memory and automatically purged after inference. We never build facial biometric templates or identity cross-references.
            </span>
          </div>

          <section className="space-y-2">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <Eye className="w-4 h-4 text-primary-400" /> 1. Ephemeral Media Processing Lifecycle
            </h3>
            <p className="text-xs text-slate-400">
              When an image or video is submitted via web or WebSocket, it is decoded into an in-memory buffer. Facial bounding boxes are identified, features extracted, and tensors evaluated. Temporary files are destroyed via atomic unlink within 24 hours of job completion.
            </p>
          </section>

          <section className="space-y-2">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <Database className="w-4 h-4 text-primary-400" /> 2. Data Encryption in Transit & At Rest
            </h3>
            <p className="text-xs text-slate-400">
              All communications are enforced over TLS 1.3 with ChaCha20-Poly1305 / AES-256-GCM ciphers. Encrypted payloads cannot be intercepted across public routing nodes.
            </p>
          </section>

          <section className="space-y-2">
            <h3 className="text-base font-semibold text-white flex items-center gap-2">
              <Globe className="w-4 h-4 text-primary-400" /> 3. International Compliance (GDPR Article 17)
            </h3>
            <p className="text-xs text-slate-400">
              Users possess an absolute "Right to Erasure". Invoking the account removal endpoint immediately cascades to delete all past detection logs, metadata keys, and audit receipts.
            </p>
          </section>
        </div>

        <div className="p-4 border-t border-slate-800 bg-slate-950/80 flex items-center justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs transition-colors shadow-lg shadow-emerald-600/20"
          >
            Understood
          </button>
        </div>
      </div>
    </div>
  );
};

export const ApiDocsModal = ({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) => {
  const [copied, setCopied] = useState<string | null>(null);

  if (!isOpen) return null;

  const copyCode = (code: string, id: string) => {
    navigator.clipboard.writeText(code);
    setCopied(id);
    setTimeout(() => setCopied(null), 2000);
  };

  const curlExample = `curl -X POST "http://localhost:8000/api/v1/detect/image" \\
  -H "accept: application/json" \\
  -F "file=@suspicious_face.jpg"`;

  const pythonExample = `import httpx

with open("face.jpg", "rb") as f:
    response = httpx.post(
        "http://localhost:8000/api/v1/detect/image",
        files={"file": ("face.jpg", f, "image/jpeg")},
        timeout=60.0
    )

result = response.json()
print("Verdict:", result["label"])          # "REAL" | "SUSPICIOUS" | "DEEPFAKE"
print("Confidence:", result["overall_score"]) # 0.0 - 1.0`;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/80 backdrop-blur-sm p-4 overflow-y-auto">
      <div className="bg-slate-900 border border-slate-800 rounded-2xl max-w-4xl w-full max-h-[88vh] flex flex-col shadow-2xl overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        <div className="p-6 border-b border-slate-800 flex items-center justify-between bg-slate-950/60">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
              <Code className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white">Developer API Reference & Integration</h2>
              <p className="text-xs text-slate-400">OpenAPI 3.1 & REST Architecture | Sub-150ms Neural Inference</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <a
              href="http://localhost:8000/docs"
              target="_blank"
              rel="noreferrer"
              className="text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 px-3 py-1.5 rounded-lg border border-slate-700 flex items-center gap-1.5 transition-colors font-mono"
            >
              Open Swagger UI <ExternalLink className="w-3.5 h-3.5" />
            </a>
            <button onClick={onClose} className="text-slate-400 hover:text-white p-2 rounded-lg hover:bg-slate-800 text-sm font-mono transition-colors">
              ✕
            </button>
          </div>
        </div>

        <div className="p-6 overflow-y-auto space-y-6 text-sm text-slate-300 font-sans leading-relaxed">
          {/* Quick specs */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 text-center">
              <div className="text-[10px] font-mono text-slate-500 uppercase">Protocol</div>
              <div className="text-xs font-bold text-white font-mono mt-0.5">HTTP/2 + WSS</div>
            </div>
            <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 text-center">
              <div className="text-[10px] font-mono text-slate-500 uppercase">Input Payload</div>
              <div className="text-xs font-bold text-white font-mono mt-0.5">Multipart / JSON</div>
            </div>
            <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 text-center">
              <div className="text-[10px] font-mono text-slate-500 uppercase">Authentication</div>
              <div className="text-xs font-bold text-white font-mono mt-0.5">Bearer JWT / API-Key</div>
            </div>
            <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 text-center">
              <div className="text-[10px] font-mono text-slate-500 uppercase">Base URL</div>
              <div className="text-xs font-bold text-emerald-400 font-mono mt-0.5">:8000/api/v1</div>
            </div>
          </div>

          {/* Endpoints Table */}
          <div>
            <h3 className="text-xs font-mono uppercase text-slate-400 font-semibold mb-2">Core Endpoints</h3>
            <div className="bg-slate-950 rounded-xl border border-slate-800 overflow-hidden divide-y divide-slate-800/80 text-xs font-mono">
              <div className="p-3 flex items-center gap-3">
                <span className="bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded font-bold">POST</span>
                <span className="text-white">/api/v1/detect/image</span>
                <span className="ml-auto text-slate-500">Synchronous single-image neural inspection</span>
              </div>
              <div className="p-3 flex items-center gap-3">
                <span className="bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded font-bold">POST</span>
                <span className="text-white">/api/v1/detect/video</span>
                <span className="ml-auto text-slate-500">Asynchronous frame-by-frame video task</span>
              </div>
              <div className="p-3 flex items-center gap-3">
                <span className="bg-blue-500/20 text-blue-400 px-2 py-0.5 rounded font-bold">POST</span>
                <span className="text-white">/api/v1/detect/url</span>
                <span className="ml-auto text-slate-500">Inspect remote URL directly</span>
              </div>
              <div className="p-3 flex items-center gap-3">
                <span className="bg-purple-500/20 text-purple-400 px-2 py-0.5 rounded font-bold">WSS</span>
                <span className="text-white">/api/v1/detect/ws/stream</span>
                <span className="ml-auto text-slate-500">Sub-150ms real-time webcam frame socket</span>
              </div>
            </div>
          </div>

          {/* Code Samples */}
          <div className="space-y-4">
            <div>
              <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-1.5">
                <span className="flex items-center gap-1.5"><Terminal className="w-3.5 h-3.5 text-primary-400" /> cURL Example</span>
                <button
                  onClick={() => copyCode(curlExample, 'curl')}
                  className="hover:text-white flex items-center gap-1 text-[11px]"
                >
                  {copied === 'curl' ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  {copied === 'curl' ? 'Copied' : 'Copy'}
                </button>
              </div>
              <pre className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 text-xs text-primary-300 font-mono overflow-x-auto">
                {curlExample}
              </pre>
            </div>

            <div>
              <div className="flex items-center justify-between text-xs font-mono text-slate-400 mb-1.5">
                <span className="flex items-center gap-1.5"><Code className="w-3.5 h-3.5 text-primary-400" /> Python (httpx) Integration</span>
                <button
                  onClick={() => copyCode(pythonExample, 'python')}
                  className="hover:text-white flex items-center gap-1 text-[11px]"
                >
                  {copied === 'python' ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                  {copied === 'python' ? 'Copied' : 'Copy'}
                </button>
              </div>
              <pre className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 text-xs text-emerald-300 font-mono overflow-x-auto">
                {pythonExample}
              </pre>
            </div>
          </div>
        </div>

        <div className="p-4 border-t border-slate-800 bg-slate-950/80 flex items-center justify-between">
          <div className="text-[11px] font-mono text-slate-500">
            OpenAPI Schema available at: <code className="text-slate-400">/openapi.json</code>
          </div>
          <button
            onClick={onClose}
            className="px-5 py-2 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs transition-colors shadow-lg shadow-indigo-600/20"
          >
            Close Documentation
          </button>
        </div>
      </div>
    </div>
  );
};

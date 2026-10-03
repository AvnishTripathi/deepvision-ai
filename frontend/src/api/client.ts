import axios from 'axios';

const apiInstance = axios.create({
  baseURL: '/api/v1',
});

apiInstance.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

apiInstance.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export const api = {
  auth: {
    login: (email: string, pass: string) => apiInstance.post('/auth/login', { email, password: pass }),
    register: (name: string, email: string, pass: string) => apiInstance.post('/auth/register', { name, email, password: pass }),
    me: () => apiInstance.get('/auth/me'),
  },
  detect: {
    image: (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      return apiInstance.post('/detect/image', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
    },
    video: (file: File) => {
      const formData = new FormData();
      formData.append('file', file);
      return apiInstance.post('/detect/video', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
    },
    url: (url: string) => apiInstance.post('/detect/url', { url }),
    getStatus: (jobId: string) => apiInstance.get(`/detect/status/${jobId}`),
    getResult: (jobId: string) => apiInstance.get(`/detect/result/${jobId}`),
  },
  history: {
    list: (page = 1, limit = 10) => apiInstance.get(`/history?page=${page}&limit=${limit}`),
    getStats: () => apiInstance.get('/history/stats'),
  },
};

export function createDetectionWebSocket(onMessage: (data: any) => void): WebSocket {
  // Directly connect to FastAPI backend WebSocket endpoint (port 8000 or proxied)
  const isHttps = window.location.protocol === 'https:';
  const host = window.location.hostname || 'localhost';
  const wsUrl = `${isHttps ? 'wss:' : 'ws:'}//${host}:8000/api/v1/detect/ws/stream`;
  
  const ws = new WebSocket(wsUrl);
  ws.onopen = () => {
    console.log('[DeepFaceGuard] Neural Live WebSocket Connected:', wsUrl);
  };
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onMessage(data);
    } catch (e) {
      console.error('WebSocket message parsing error', e);
    }
  };
  ws.onerror = (err) => {
    console.warn('[DeepFaceGuard] WebSocket error, trying proxy endpoint...', err);
  };
  return ws;
}

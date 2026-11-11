import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// Change TARGET if your FastAPI backend runs on another port.
const TARGET = 'http://127.0.0.1:8000';
const http = ['/health', '/monitor', '/risk', '/events', '/patients', '/rooms'];

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5175,
    proxy: {
      ...Object.fromEntries(http.map(r => [r, TARGET])),
      '/ws': { target: TARGET, ws: true },   // WebSocket /ws/monitor (live updates)
    },
  },
});

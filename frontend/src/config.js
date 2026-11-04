// Backend connection settings. Everything on screen comes from the FastAPI backend (no sample data).
export const API_BASE = import.meta.env.VITE_API_URL || '';
export const POLL_MS = 2000;
// WebSocket of the backend: @app.websocket("/ws/monitor"). In dev it goes through the Vite proxy ('/ws').
export const WS_URL = import.meta.env.VITE_WS_URL
  || `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws/monitor`;
// A pose frame older than this counts as "no signal".
export const STALE_MS = 15000;

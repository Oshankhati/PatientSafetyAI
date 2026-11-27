import { API_BASE } from '../config';

export async function apiRequest(path, { token, ...options } = {}) {
  const t0 = performance.now();
  const headers = new Headers(options.headers || {});
  headers.set('Accept', 'application/json');
  if (token) headers.set('Authorization', `Bearer ${token}`);
  if (options.body && !headers.has('Content-Type')) headers.set('Content-Type', 'application/json');
  const res = await fetch(`${API_BASE}${path}`, { ...options, headers });
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const error = new Error(data?.detail || `${path}: HTTP ${res.status}`);
    error.status = res.status;
    throw error;
  }
  return { data, ms: Math.round(performance.now() - t0) };
}

async function safe(path, fallback, token) {
  try { return { ...(await apiRequest(path, { token })), error: null }; }
  catch (e) { return { data: fallback, ms: null, status: e.status, error: e.message || `${path} unavailable` }; }
}

// Each endpoint is fetched independently so one failure never discards the others.
export async function loadMonitoringData(token) {
  const [health, state, events, patients, rooms, acknowledgements] = await Promise.all([
    safe('/health', null),
    safe('/monitor/state', null, token),
    safe('/events/latest?n=100', { events: [] }, token),
    safe('/patients', { patients: [] }, token),
    safe('/rooms', { rooms: [] }, token),
    safe('/events/acknowledgements', { acknowledgements: {} }, token),
  ]);
  return {
    health: health.data,
    latency: health.ms,
    state: state.data,
    events: Array.isArray(events.data?.events) ? events.data.events : [],
    patients: Array.isArray(patients.data?.patients) ? patients.data.patients : [],
    rooms: Array.isArray(rooms.data?.rooms) ? rooms.data.rooms : [],
    acknowledgements: acknowledgements.data?.acknowledgements || {},
    errors: [health, state, events, patients, rooms, acknowledgements].filter(r => r.error).map(r => r.error),
    unauthorized: [state, events, patients, rooms, acknowledgements].some(r => r.status === 401),
  };
}

export async function loadSettings(token) {
  const { data } = await apiRequest('/settings', { token });
  return data.settings;
}

export async function saveSettings(token, settings) {
  const { data } = await apiRequest('/settings', {
    token,
    method: 'PUT',
    body: JSON.stringify({ settings }),
  });
  return data.settings;
}

export async function acknowledgeEvent(token, eventId) {
  const { data } = await apiRequest(`/events/${encodeURIComponent(eventId)}/acknowledge`, {
    token,
    method: 'POST',
  });
  return { ...data.acknowledgement, staff_name: data.staff.full_name };
}

export async function getCameraStatus(token) {
  const { data } = await apiRequest('/camera/status', { token });
  return data;
}

export async function setCameraRunning(token, running) {
  const { data } = await apiRequest(running ? '/camera/start?source=0' : '/camera/stop', {
    token,
    method: 'POST',
  });
  return data;
}

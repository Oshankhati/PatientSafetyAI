import { API_BASE } from '../config';

async function get(path) {
  const t0 = performance.now();
  const res = await fetch(`${API_BASE}${path}`, { headers: { Accept: 'application/json' } });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  const data = await res.json();
  return { data, ms: Math.round(performance.now() - t0) };
}

async function safe(path, fallback) {
  try { return { ...(await get(path)), error: null }; }
  catch (e) { return { data: fallback, ms: null, error: e.message || `${path} unavailable` }; }
}

// Each endpoint is fetched independently so one failure never discards the others.
export async function loadMonitoringData() {
  const [health, state, events, patients, rooms] = await Promise.all([
    safe('/health', null),
    safe('/monitor/state', null),
    safe('/events/latest?n=100', { events: [] }),
    safe('/patients', { patients: [] }),
    safe('/rooms', { rooms: [] }),
  ]);
  return {
    health: health.data,
    latency: health.ms,
    state: state.data,
    events: Array.isArray(events.data?.events) ? events.data.events : [],
    patients: Array.isArray(patients.data?.patients) ? patients.data.patients : [],
    rooms: Array.isArray(rooms.data?.rooms) ? rooms.data.rooms : [],
    errors: [health, state, events, patients, rooms].filter(r => r.error).map(r => r.error),
  };
}

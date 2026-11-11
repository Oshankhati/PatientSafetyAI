import { useCallback, useEffect, useRef, useState } from 'react';
import { loadMonitoringData } from '../services/api';
import { POLL_MS, WS_URL } from '../config';

const EMPTY = { health: null, latency: null, state: null, events: [], patients: [], rooms: [], errors: [] };
const KEY = 'psai.risk-history.v1';
const KEEP_MS = 7 * 86400000, MAX_POINTS = 5000;

const riskPct = s => {
  const v = Number((s?.risk || s?.risk_assessment)?.risk_score);
  return Number.isFinite(v) ? Math.round(v <= 1 ? v * 100 : v) : null;
};
const hasPose = s => Boolean(s?.skeleton?.keypoints_normalized?.length || s?.skeleton?.keypoints?.length);

function loadHistory() {
  try {
    const h = JSON.parse(localStorage.getItem(KEY)) || {}, cut = Date.now() - KEEP_MS;
    return Object.fromEntries(Object.entries(h).map(([k, v]) => [k, v.filter(([t]) => t >= cut)]));
  } catch { return {}; }
}

/**
 * Talks to the FastAPI backend:
 *  - polls /health, /monitor/state, /events/latest, /patients, /rooms
 *  - listens on /ws/monitor so a new frame / fall shows up immediately
 *  - records every live risk score (per patient) for the risk-history charts
 */
export default function useMonitoring() {
  const [data, setData] = useState(EMPTY);
  const [updated, setUpdated] = useState(null);
  const [ws, setWs] = useState('connecting');
  const [poseAt, setPoseAt] = useState(null);
  const [history, setHistory] = useState(loadHistory);
  const histRef = useRef(history);
  const lastWrite = useRef(0);

  const record = useCallback(state => {
    if (hasPose(state)) setPoseAt(Date.now());
    const v = riskPct(state), id = state?.patient_id;
    if (v == null || !id) return;
    const h = { ...histRef.current, [id]: [...(histRef.current[id] || []).slice(-(MAX_POINTS - 1)), [Date.now(), v]] };
    histRef.current = h; setHistory(h);
    if (Date.now() - lastWrite.current > 10000) {
      lastWrite.current = Date.now();
      try { localStorage.setItem(KEY, JSON.stringify(h)); } catch { /* storage unavailable */ }
    }
  }, []);

  const refresh = useCallback(async () => {
    try {
      const d = await loadMonitoringData();
      setData(d); setUpdated(new Date()); record(d.state);
    } catch (e) { setData({ ...EMPTY, errors: [e.message || 'Backend unavailable'] }); setUpdated(new Date()); }
  }, [record]);

  useEffect(() => { refresh(); const id = setInterval(refresh, POLL_MS); return () => clearInterval(id); }, [refresh]);

  useEffect(() => {
    let closed = false, socket, retry, debounce;
    const open = () => {
      try { socket = new WebSocket(WS_URL); } catch { setWs('disconnected'); retry = setTimeout(open, 3000); return; }
      setWs('connecting');
      socket.onopen = () => setWs('connected');
      socket.onmessage = ev => {
        let msg; try { msg = JSON.parse(ev.data); } catch { return; }
        if (msg.type === 'update' && msg.result) {
          setData(d => ({ ...d, state: msg.result })); record(msg.result);
          clearTimeout(debounce); debounce = setTimeout(refresh, 400); // pick up the newly stored event
        } else if (msg.type === 'snapshot') { refresh(); }
      };
      socket.onclose = () => { if (!closed) { setWs('disconnected'); retry = setTimeout(open, 3000); } };
      socket.onerror = () => socket.close();
    };
    open();
    return () => { closed = true; clearTimeout(retry); clearTimeout(debounce); socket?.close(); };
  }, [record, refresh]);

  return { ...data, online: Boolean(data.health), updated, refresh, ws, poseAt, history };
}

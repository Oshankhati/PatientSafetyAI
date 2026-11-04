import { STALE_MS } from '../config';
import { ago } from '../utils';

/* Turns raw backend JSON into what the screens need. Field names are read tolerantly because the
   event / risk dictionaries are produced by the monitoring pipeline (src/pipeline/monitor.py). */

const first = (...v) => v.find(x => x !== undefined && x !== null && x !== '');
const num = v => (v == null || v === '' || Number.isNaN(Number(v)) ? null : Number(v));
const toPct = v => { const n = num(v); return n == null ? null : Math.round(n <= 1 ? n * 100 : n); };
const words = s => (s ? String(s).replace(/[_-]+/g, ' ').toLowerCase().replace(/^\w/, c => c.toUpperCase()) : null);
const pad = n => String(n).padStart(2, '0');

export const isToday = d => Boolean(d) && d.toDateString() === new Date().toDateString();

/** Backend timestamps may be epoch seconds/ms, ISO strings, or seconds into a video. */
function readTime(ts) {
  if (ts == null || ts === '') return { date: null, time: '—', raw: 0 };
  const n = Number(ts);
  if (Number.isNaN(n)) {
    const d = new Date(ts);
    return Number.isNaN(d.getTime()) ? { date: null, time: String(ts), raw: 0 } : { date: d, time: d.toLocaleTimeString('en-GB', { hour12: false }), raw: d.getTime() };
  }
  if (n > 1e9) {
    const d = new Date(n > 1e11 ? n : n * 1000);
    return { date: d, time: d.toLocaleTimeString('en-GB', { hour12: false }), raw: d.getTime() };
  }
  const s = Math.round(n);
  return { date: null, time: s >= 3600 ? `${Math.floor(s / 3600)}:${pad(Math.floor((s % 3600) / 60))}:${pad(s % 60)}` : `${Math.floor(s / 60)}:${pad(s % 60)}`, raw: n };
}

const levelOf = (label, score, th) => {
  const l = String(label || '').toLowerCase();
  if (/fall|critical|high/.test(l)) return 'High';
  if (/medium|moderate/.test(l)) return 'Medium';
  if (/low|normal|safe/.test(l)) return 'Normal';
  if (score == null) return null;
  return score / 100 >= th.high ? 'High' : score / 100 >= th.medium ? 'Medium' : 'Normal';
};

const kindOf = text => (/fall/i.test(text) ? 'fall' : /signal|lost|away|offline/i.test(text) ? 'away' : /high|unsafe|risk|critical|alert/i.test(text) ? 'high' : 'info');
const TAG = { fall: 'Fall', high: 'High', away: 'Away' };
const CARD = { fall: 'Fall detected', high: 'High risk', away: 'Signal lost' };

function mapEvent(e, i) {
  const t = readTime(first(e.timestamp, e.time, e.created_at, e.detected_at));
  const type = words(first(e.event_type, e.type, e.label, e.kind)) || 'Event';
  const kind = kindOf(`${type} ${e.severity || ''}`);
  const risk = toPct(first(e.risk_score, typeof e.risk === 'object' ? e.risk?.risk_score : e.risk, e.score));
  const dur = first(e.duration, e.duration_s, e.duration_seconds);
  return {
    id: String(first(e.id, e.event_id, `EVT-${t.raw || i}-${type}`)), kind, type, date: t.date, time: t.time, raw: t.raw,
    patient: first(e.patient_id, e.patient) || '—', room: first(e.room_id, e.room) || '—', bed: first(e.bed_id, e.bed) || '',
    risk, confidence: toPct(first(e.confidence, e.fall_confidence, e.detection_confidence)),
    status: words(first(e.status, e.state)) || 'Open', zone: first(e.zone, e.zone_name) || '', duration: dur == null ? '' : dur,
    outcome: first(e.outcome, e.note) || '',
  };
}

function mergeKeypoints(sk) {
  const pts = sk?.keypoints_normalized || sk?.keypoints;
  if (!pts?.length) return null;
  const conf = sk.keypoint_confidence || [];
  return pts.map((p, i) => (Array.isArray(p) ? [p[0], p[1], p[2] ?? conf[i] ?? 1] : [p.x, p.y, p.confidence ?? conf[i] ?? 1]));
}

function signalsOf(risk, signals) {
  const list = first(risk?.reasons, risk?.factors, risk?.contributing_factors);
  if (Array.isArray(list) && list.length) return list.map(x => words(typeof x === 'string' ? x : x?.name || x?.label)).filter(Boolean);
  return Object.entries(signals || {}).filter(([, v]) => v === true).map(([k]) => words(k));
}

function buildPose(st, th) {
  if (!st || (!st.risk && !st.risk_assessment && !st.skeleton)) return null;
  const risk = st.risk || st.risk_assessment || null;
  const score = toPct(risk?.risk_score);
  const keypoints = mergeKeypoints(st.skeleton);
  const confs = (st.skeleton?.keypoint_confidence || []).map(Number).filter(Number.isFinite);
  const prediction = words(first(risk?.prediction_label, risk?.prediction, risk?.label));
  const fallConfirmed = risk?.fall_confirmed === true || /fall/i.test(`${prediction || ''} ${risk?.risk_level || ''}`);
  const w = num(first(st.frame_width, st.skeleton?.frame_width)), h = num(first(st.frame_height, st.skeleton?.frame_height));
  return {
    keypoints, aspect: w && h ? w / h : null, patient: st.patient_id || '', room: st.room_id || '',
    detConf: confs.length ? Math.round((confs.reduce((a, b) => a + b, 0) / confs.length) * 100) : null,
    prediction, predConf: toPct(first(risk?.confidence, risk?.prediction_confidence, risk?.model_confidence)),
    score, level: fallConfirmed ? 'High' : levelOf(first(risk?.risk_level, risk?.level), score, th),
    signals: signalsOf(risk, st.signals), fps: first(st.fps, st.frame_rate) ?? '', frame: first(st.frame_number, st.frame) ?? '', fallConfirmed,
  };
}

export function buildViewModel(raw, th) {
  const pose = buildPose(raw.state, th);
  const liveId = pose?.patient || null, liveRoom = pose?.room || null;

  const events = raw.events.map(mapEvent).sort((a, b) => b.raw - a.raw);

  const patients = raw.patients.map(p => {
    const id = String(first(p.id, p.patient_id, p.name, '—'));
    const live = Boolean(liveId) && id === liveId;
    return { id, room: first(p.room_id, p.room) || '—', bed: first(p.bed_id, p.bed) || '', live,
      status: live ? 'Monitoring' : 'No signal', risk: live ? pose.score : null, level: live ? pose.level : 'Away' };
  });
  if (liveId && !patients.some(p => p.id === liveId)) patients.unshift({ id: liveId, room: liveRoom || '—', bed: '', live: true, status: 'Monitoring', risk: pose.score, level: pose.level });

  const rooms = raw.rooms.map(r => {
    const id = String(first(r.id, r.room_id, '—'));
    const pid = first(r.patient_id, patients.find(p => p.room === id)?.id) || '—';
    const live = Boolean(liveId) && (pid === liveId || id === liveRoom);
    return { id, patient: pid, zones: r.zones || null, status: live ? 'Monitoring' : 'No signal', risk: live ? pose.score : null, level: live ? pose.level : 'Away' };
  });
  if (liveRoom && !rooms.some(r => r.id === liveRoom)) rooms.unshift({ id: liveRoom, patient: liveId || '—', zones: null, status: 'Monitoring', risk: pose.score, level: pose.level });

  const alerts = events.filter(e => e.kind !== 'info').map(e => ({
    id: e.id, eventId: e.id, kind: e.kind, tag: TAG[e.kind], title: e.type, cardLabel: CARD[e.kind], patient: e.patient, room: e.room,
    confidence: e.confidence, risk: e.risk, detected: e.time, ago: e.date ? ago(e.date) : e.time, status: e.status,
  }));

  return { pose, patients, rooms, events, alerts };
}

/** Health cards derived from /health, the WebSocket and the age of the last pose frame. */
export function buildServices({ health, online, latency, ws, poseAt }) {
  const age = poseAt ? Date.now() - poseAt : null;
  const fresh = age != null && age < STALE_MS;
  const model = health?.yolo_model ? String(health.yolo_model).split(/[\\/]/).pop() : '';
  return [
    { id: 'camera', icon: 'camera', name: 'Pose pipeline', sub: age == null ? 'No keypoints received yet' : fresh ? 'Receiving keypoints' : `Last frame ${Math.round(age / 1000)}s ago`, badge: fresh ? 'Online' : 'No signal', tone: fresh ? 'ok' : 'warn' },
    { id: 'yolo', icon: 'cpu', name: 'YOLO model', sub: model || 'Not reported', badge: model ? 'Configured' : online ? 'Not reported' : 'Unknown', tone: model ? 'ok' : online ? 'warn' : 'bad' },
    { id: 'temporal', icon: 'cpu', name: 'Temporal model', sub: !online ? 'Not reported' : health.lstm_available ? 'LSTM available' : 'LSTM not available', badge: !online ? 'Unknown' : health.lstm_available ? 'Loaded' : 'Unavailable', tone: !online ? 'bad' : health.lstm_available ? 'ok' : 'warn' },
    { id: 'api', icon: 'cpu', name: 'FastAPI', sub: online ? `Latency ${latency}ms` : 'GET /health failed', badge: online ? 'Online' : 'Offline', tone: online ? 'ok' : 'bad' },
    { id: 'db', icon: 'db', name: 'Database', sub: 'Not reported by /health', badge: 'Not reported', tone: 'warn' },
    { id: 'ws', icon: 'cpu', name: 'WebSocket', sub: '/ws/monitor', badge: ws === 'connected' ? 'Connected' : ws === 'connecting' ? 'Connecting' : 'Disconnected', tone: ws === 'connected' ? 'ok' : ws === 'connecting' ? 'warn' : 'bad' },
  ];
}

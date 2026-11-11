import { useMemo, useState } from 'react';
import Icon from '../components/Icon';
import PoseCanvas from '../components/PoseCanvas';
import { Panel, PageHeading } from '../components/ui';

const ROLES = ['Admin', 'Nurse', 'Viewer'];

// rooms[].zones from the backend: {name: [x1, y1, x2, y2]} (0-1 of the frame; pixel values are scaled by 1280x720)
function toZones(zones) {
  if (!zones) return [];
  return Object.entries(zones).filter(([, a]) => Array.isArray(a) && a.length === 4 && a.every(Number.isFinite)).map(([name, a]) => {
    const [x1, y1, x2, y2] = a.some(v => v > 1.5) ? [a[0] / 1280, a[1] / 720, a[2] / 1280, a[3] / 720] : a;
    const kind = /bed/i.test(name) ? 'bed' : /floor/i.test(name) ? 'floor' : /unsafe|danger/i.test(name) ? 'unsafe' : 'safe';
    return { label: name.replace(/[_-]+/g, ' ').toUpperCase(), kind, box: [x1, y1, x2 - x1, y2 - y1] };
  });
}

function Num({ label, hint, value, onChange, disabled, placeholder }) {
  return (
    <label className="num"><span>{label}</span>
      <input value={value} placeholder={placeholder} disabled={disabled} onChange={e => onChange(e.target.value)} inputMode="decimal" />
      {hint && <small>{hint}</small>}</label>
  );
}

export default function Settings({ vm, settings, setSettings }) {
  const [role, setRole] = useState('Admin');
  const [roomId, setRoomId] = useState('');
  const [saved, setSaved] = useState(false);
  const [draft, setDraft] = useState(settings);
  const editable = role === 'Admin';
  const { ai, camera: cam } = draft;
  const setAi = k => v => { setSaved(false); setDraft(d => ({ ...d, ai: { ...d.ai, [k]: v } })); };
  const setCam = k => v => { setSaved(false); setDraft(d => ({ ...d, camera: { ...d.camera, [k]: v } })); };
  const m = parseFloat(ai.medium), h = parseFloat(ai.high), f = parseFloat(ai.fall);
  const valid = [m, h, f].every(n => !Number.isNaN(n)) && m < h && h < f;
  const room = vm.rooms.find(r => r.id === roomId) || vm.rooms[0];
  const zones = useMemo(() => toZones(room?.zones), [room]);
  const pose = vm.pose && room && vm.pose.room === room.id ? vm.pose : null;

  return (
    <>
      <PageHeading title="Settings" sub="AI thresholds, cameras and monitored zones"
        right={<div className="role-toggle"><span>Preview role</span>
          <button className="btn primary" onClick={() => setRole(ROLES[(ROLES.indexOf(role) + 1) % ROLES.length])}>{role}</button></div>} />
      <div className="settings-grid">
        <div className="stack">
          <Panel title="AI settings" sub="Model and risk engine configuration">
            <div className="form-grid">
              <Num label="YOLO confidence" hint="Configured as 0–1" value={ai.yolo} onChange={setAi('yolo')} disabled={!editable} />
              <Num label="Medium threshold" hint="Configured as 0–1" value={ai.medium} onChange={setAi('medium')} disabled={!editable} />
              <Num label="High-risk threshold" hint="Configured as 0–1" value={ai.high} onChange={setAi('high')} disabled={!editable} />
              <Num label="Fall threshold" hint="Configured as 0–1" value={ai.fall} onChange={setAi('fall')} disabled={!editable} />
              <Num label="Sequence length" hint="Frames" value={ai.seqLen} onChange={setAi('seqLen')} disabled={!editable} />
              <Num label="Sequence stride" hint="Frames" value={ai.stride} onChange={setAi('stride')} disabled={!editable} />
              <Num label="Minimum consecutive frames" hint="Frames" value={ai.minFrames} onChange={setAi('minFrames')} disabled={!editable} />
            </div>
            <div className={`validity ${valid ? 'ok' : 'bad'}`}>
              <Icon name={valid ? 'check' : 'alert'} size={16} />
              {valid ? 'Threshold order is valid: Medium < High risk < Fall' : 'Thresholds must satisfy Medium < High risk < Fall'}
            </div>
          </Panel>
          <Panel title="Camera settings" sub="Primary source configuration">
            <div className="form-grid">
              <label className="num"><span>Source</span><select disabled={!editable} value={cam.source} onChange={e => setCam('source')(e.target.value)}>
                <option>RTSP stream</option><option>USB webcam</option><option>Video file</option></select></label>
              <Num label="Camera ID" placeholder="CAM-01" value={cam.cameraId} onChange={setCam('cameraId')} disabled={!editable} />
              <label className="num"><span>Resolution</span><select disabled={!editable} value={cam.resolution} onChange={e => setCam('resolution')(e.target.value)}>
                <option>1920 × 1080</option><option>1280 × 720</option><option>640 × 480</option></select></label>
              <Num label="FPS" placeholder="24" value={cam.fps} onChange={setCam('fps')} disabled={!editable} />
            </div>
          </Panel>
        </div>
        <div className="stack">
          <Panel title="Zone configuration" sub="Define room safety zones">
            <label className="num"><span>Room</span>
              <select value={room?.id || ''} onChange={e => setRoomId(e.target.value)} disabled={!vm.rooms.length}>
                {vm.rooms.length ? vm.rooms.map(r => <option key={r.id}>{r.id}</option>) : <option>No rooms</option>}</select></label>
            <div className="zone-canvas"><PoseCanvas keypoints={pose?.keypoints} zones={zones} height={230} aspect={pose?.aspect || 16 / 9}
              tag={pose?.keypoints ? `${(pose.patient || 'PERSON').toUpperCase()}${pose.detConf != null ? ` · ${pose.detConf}%` : ''}` : null}
              emptyText={room ? 'No live pose for this room' : 'No rooms configured'} /></div>
            <div className="zone-legend"><span><i className="bed" />Bed zone</span><span><i className="safe" />Safe zone</span><span><i className="floor" />Floor zone</span></div>
            {room && !zones.length && <p className="muted tiny">This room has no zones in the backend (GET /rooms → zones).</p>}
            <button className="btn primary" disabled={!editable || !valid} onClick={() => { setSettings(draft); setSaved(true); }}>Save configuration</button>
            {saved && <div className="validity ok"><Icon name="check" size={16} />Configuration saved in this browser</div>}
            <p className="muted tiny">The backend has no settings endpoint yet, so these values are stored in this browser and drive the risk-level colours shown here.</p>
          </Panel>
        </div>
      </div>
    </>
  );
}

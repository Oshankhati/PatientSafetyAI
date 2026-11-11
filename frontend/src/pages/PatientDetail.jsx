import { useMemo, useState } from 'react';
import PoseCanvas from '../components/PoseCanvas';
import RecentEvents from '../components/RecentEvents';
import { RiskLine } from '../components/Charts';
import { Panel, MetaStrip, Pill, Select, Eyebrow } from '../components/ui';
import { timeLabel } from '../utils';

const RANGES = { 'Last 3 hours': 3 * 3600e3, 'Last 24 hours': 24 * 3600e3, 'Last 7 days': 7 * 24 * 3600e3 };

export default function PatientDetail({ vm, id, back, history, th, statusOf, updatedAt }) {
  const p = vm.patients.find(x => x.id === id);
  const [range, setRange] = useState('Last 3 hours');
  const all = history[id] || [];
  const points = useMemo(() => { const c = Date.now() - RANGES[range]; return all.filter(([t]) => t >= c); }, [all, range]);
  if (!p) return <div className="page-heading"><div><button className="crumb" onClick={back}>← Back to patients</button><h1>Patient not found</h1></div></div>;
  const pose = p.live ? vm.pose : null;
  return (
    <>
      <div className="page-heading">
        <div><button className="crumb" onClick={back}>← Back to patients</button><h1>{p.id}</h1><p>{p.room}{p.bed ? ` · Bed ${p.bed}` : ''}</p></div>
        <div className="score-line">Current risk score <b>{p.risk == null ? '—' : `${p.risk}%`}</b><Pill level={p.level} /></div>
      </div>
      <div className="detail-grid">
        <Panel>
          <div className="row-between"><Eyebrow>CURRENT STATUS</Eyebrow>{p.live && pose?.keypoints && <span className="live-tag"><i />Live</span>}</div>
          <h2 className="h-sm">{p.status}</h2>
          <PoseCanvas keypoints={pose?.keypoints} height={250} aspect={pose?.aspect || 16 / 9} emptyText={p.live ? 'Waiting for live pose keypoints…' : 'No live feed for this patient'}
            tag={pose?.keypoints ? `${p.id.toUpperCase()}${pose.detConf != null ? ` · ${pose.detConf}%` : ''}` : null} />
          <MetaStrip items={[[pose?.fps || '—', 'FPS'], [p.live ? p.id : '—', 'Tracked'], [p.room, 'Room']]} />
        </Panel>
        <Panel>
          <Eyebrow>PATIENT OVERVIEW</Eyebrow>
          <dl className="dl">
            <div><dt>Anonymous ID</dt><dd>{p.id}</dd></div><div><dt>Room</dt><dd>{p.room}</dd></div><div><dt>Bed</dt><dd>{p.bed || '—'}</dd></div>
            <div><dt>AI prediction</dt><dd>{pose?.prediction ? `${pose.prediction}${pose.predConf != null ? ` · ${pose.predConf}%` : ''}` : '—'}</dd></div>
            <div><dt>Risk engine</dt><dd className={p.level === 'High' ? 'warn' : ''}>{p.risk == null ? '—' : `${p.risk}% · ${p.level}`}</dd></div>
            <div><dt>First reading</dt><dd>{all.length ? timeLabel(new Date(all[0][0])) : '—'}</dd></div>
          </dl>
        </Panel>
      </div>
      <Panel title="Risk history" sub="Risk engine score over time"
        action={<Select label="Range" value={range} onChange={setRange} options={Object.keys(RANGES)} />}>
        <RiskLine points={points} height={260} high={th.high * 100} fall={th.fall * 100} />
      </Panel>
      <RecentEvents title="Event history" sub="Events associated with this patient" events={vm.events.filter(e => e.patient === id)} patients={vm.patients} filters={false} statusOf={statusOf} />
    </>
  );
}

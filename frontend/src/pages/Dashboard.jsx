import Icon from '../components/Icon';
import PoseCanvas from '../components/PoseCanvas';
import RecentEvents from '../components/RecentEvents';
import { Panel, PageHeading, MetaStrip, Pill, Eyebrow } from '../components/ui';
import { timeLabel } from '../utils';
import { isToday } from '../data/model';

export default function Dashboard({ vm, online, healthy, total, go, statusOf, updatedAt }) {
  const { patients, pose, events } = vm;
  const count = re => patients.filter(p => re.test(p.level)).length;
  const falls = events.filter(e => e.kind === 'fall' && (!e.date || isToday(e.date)));
  const lastFall = falls[0];
  const cards = [
    ['Total patients', patients.length, `${vm.rooms.length} monitored room${vm.rooms.length === 1 ? '' : 's'}`, 'users', ''],
    ['Normal', count(/normal/i), 'Within safe thresholds', 'check', 'ok'],
    ['High risk', count(/high|medium/i), 'Need closer attention', 'alert', 'risk'],
    ['Falls today', falls.length, lastFall ? `Latest ${lastFall.time}` : 'None detected', 'bell', 'fall'],
    ['System', online ? 'Online' : 'Offline', online ? `${healthy} of ${total} services healthy` : 'Waiting for backend', 'cpu', 'sys'],
  ];
  const high = pose?.level === 'High';
  return (
    <>
      <PageHeading title="Dashboard" sub="Overview of patient safety and monitoring activity" />
      <div className="stat-grid">
        {cards.map(([l, v, s, icon, kind]) => (
          <article className="stat-card" key={l}>
            <span className={`stat-icon ${kind}`}><Icon name={icon} size={20} /></span>
            <div><small>{l}</small><strong>{v}</strong><span>{s}</span></div>
          </article>
        ))}
      </div>

      <div className="dash-grid">
        <Panel className="live-card">
          <div className="live-head">
            <div><Eyebrow><span className={`live-dot ${pose?.keypoints ? '' : 'idle'}`} />LIVE</Eyebrow><h2>Live monitoring</h2><p>{pose ? `${pose.room || '—'} · ${pose.patient || '—'}` : 'No active feed'}</p></div>
            <button className="link" onClick={() => go('live')}>Open focused view ↗</button>
          </div>
          <PoseCanvas keypoints={pose?.keypoints} height={300} aspect={pose?.aspect || 16 / 9} tag={pose?.keypoints ? `${(pose.patient || 'PERSON').toUpperCase()}${pose.detConf != null ? ` · ${pose.detConf}%` : ''}` : null} />
          <MetaStrip items={[[pose?.fps || '—', 'FPS'], [pose?.patient || '—', 'Tracked'], [pose?.room || '—', 'Room'], [timeLabel(updatedAt), 'Timestamp']]} />
        </Panel>
        <Panel className="side-card"><Eyebrow>AI PREDICTION</Eyebrow><h3 className="big">{pose?.prediction || '—'}</h3>
          <div className="kv"><span>Confidence</span><b>{pose?.predConf == null ? '—' : `${pose.predConf}%`}</b></div></Panel>
        <Panel className="side-card"><Eyebrow>RISK SCORE</Eyebrow><h3 className={`big ${high ? 'warn' : ''}`}>{pose?.score == null ? '—' : `${pose.score}%`}</h3>
          {pose?.level && <Pill level={pose.level} />}
          {pose?.signals?.length > 0 && <ul className="factors">{pose.signals.map(s => <li key={s}>{s}</li>)}</ul>}</Panel>
      </div>

      <RecentEvents title="Recent events" sub="Latest detections from monitored rooms" events={events} patients={patients} statusOf={statusOf} onAll={() => go('events')} />
    </>
  );
}

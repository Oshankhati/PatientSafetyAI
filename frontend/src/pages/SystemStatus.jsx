import { useState } from 'react';
import Icon from '../components/Icon';
import { Panel, PageHeading, Eyebrow } from '../components/ui';
import { timeLabel } from '../utils';

export default function SystemStatus({ services, online, updatedAt }) {
  const [wsDown, setWsDown] = useState(true);
  const bad = services.filter(s => s.tone === 'bad').length;
  const warn = services.filter(s => s.tone === 'warn').length;
  const title = !online ? 'Backend not reachable' : bad ? 'Some services are down' : warn ? 'Operational, some services waiting' : 'All systems operational';
  return (
    <>
      <PageHeading title="System Status" sub="Live health of the monitoring pipeline" />
      <div className={`ok-banner ${!online || bad ? 'bad' : warn ? 'warn' : ''}`}><span className="ok-icon"><Icon name={online && !bad ? 'check' : 'alert'} size={24} stroke={2.4} /></span>
        <div><b>{title}</b><small>Last checked {timeLabel(updatedAt)} · Auto-refresh every 2 seconds</small></div></div>
      <div className="svc-grid">
        {services.map(s => (
          <article key={s.id} className="svc-card">
            <div className="row-between"><span className="svc-icon"><Icon name={s.icon} size={18} /></span><span className={`badge-ok ${s.tone}`}><i />{s.badge}</span></div>
            <h3>{s.name}</h3><small>{s.sub}</small>
            <div className="svc-foot"><span>Last check</span><b>{timeLabel(updatedAt)}</b></div>
          </article>
        ))}
      </div>
      <Panel title="Error state preview" sub="Component behavior if a dependency becomes unavailable">
        {wsDown ? (
          <div className="err-banner"><span className="err-icon">!</span>
            <div><b>WebSocket disconnected</b><small>Live data is unavailable. Patient states will show “No signal” until the connection is restored.</small></div>
            <button className="btn white" onClick={() => setWsDown(false)}>Retry connection</button></div>
        ) : <div className="ok-banner slim"><b>WebSocket reconnected</b><button className="link" onClick={() => setWsDown(true)}>Show error state again</button></div>}
      </Panel>
      <div className="state-grid">
        <Panel><Eyebrow>LOADING STATE</Eyebrow><div className="skel w100" /><div className="skel-row"><div className="skel" /><div className="skel" /></div><div className="skel w40" /></Panel>
        <Panel><Eyebrow>EMPTY STATE</Eyebrow><div className="empty-state"><Icon name="events" size={22} stroke={1.4} /><b>No events found</b><small>Try changing the selected filters.</small></div></Panel>
        <Panel><Eyebrow>FAILED SAVE</Eyebrow><b className="fail">Configuration not saved</b><small className="muted">Check the connection and try again.</small><div><button className="btn outline sm">Try again</button></div></Panel>
      </div>
    </>
  );
}

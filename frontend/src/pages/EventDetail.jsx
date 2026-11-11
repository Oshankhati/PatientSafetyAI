import Icon from '../components/Icon';
import { Panel, Eyebrow, StatusBadge } from '../components/ui';
import { timeLabel } from '../utils';

export default function EventDetail({ vm, id, back, statusOf, acks }) {
  const e = vm.events.find(x => x.id === id);
  if (!e) return <div className="page-heading"><div><button className="crumb" onClick={back}>← Back to events</button><h1>Event not found</h1></div></div>;
  const status = statusOf(e);
  const ack = acks[e.id];
  const dur = e.duration === '' ? '—' : /^\d+(\.\d+)?$/.test(String(e.duration)) ? `${e.duration} seconds` : e.duration;
  const rows = [
    ['Patient', e.patient], ['Room', `${e.room}${e.bed ? ` · Bed ${e.bed}` : ''}`],
    ['Timestamp', e.date ? e.date.toLocaleString('en-GB', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false }) : e.time],
    ['AI prediction', e.type], ['Model confidence', e.confidence == null ? '—' : `${e.confidence}%`], ['Risk score', e.risk == null ? '—' : `${e.risk}%`],
    ['Zone', e.zone || '—'], ['Duration', dur], ['Status', status],
  ];
  return (
    <>
      <div className="page-heading"><div>
        <button className="crumb" onClick={back}>← Back to events</button>
        <h1>Event {e.id}</h1><p>Recorded {e.date ? `at ${e.time}` : `at ${e.time} into the video`}</p></div></div>
      <div className="event-grid">
        <Panel>
          <div className="row-between"><div><Eyebrow>EVENT DETAILS</Eyebrow><h2 className="h-sm">{e.type}</h2></div><StatusBadge status={status} /></div>
          <dl className="dl two">{rows.map(([k, v]) => <div key={k}><dt>{k}</dt><dd>{v}</dd></div>)}</dl>
        </Panel>
        <div className="stack">
          <Panel><Eyebrow>SNAPSHOT</Eyebrow>
            <div className="snapshot"><Icon name="camera" size={28} stroke={1.4} /><b>Not stored</b><small>Raw-video storage is disabled</small></div></Panel>
          <Panel><Eyebrow>ACKNOWLEDGEMENT</Eyebrow>
            {ack ? (
              <>
                <div className="ack-user"><span className="avatar sm">AN</span><span><b>Alex Nurse</b><small>Acknowledged at {timeLabel(new Date(ack))}</small></span></div>
                {e.outcome && <div className="kv ruled"><span>Outcome</span><b>{e.outcome}</b></div>}
                <p className="muted tiny">Recorded in this browser only. The backend has no acknowledgement endpoint.</p>
              </>
            ) : status === 'Acknowledged' || status === 'Resolved' ? <p className="muted">{status} by the backend.</p> : <p className="muted">Not yet acknowledged.</p>}
          </Panel>
        </div>
      </div>
    </>
  );
}

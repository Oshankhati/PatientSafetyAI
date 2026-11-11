import { useMemo, useState } from 'react';
import Icon from '../components/Icon';
import { Panel, PageHeading, FilterBar, defaultFilters, applyFilters, Pill, Empty } from '../components/ui';

const isOpen = s => s === 'Open' || s === 'Investigating';

export default function Alerts({ vm, statusOf, onAck, openEvent, openPatient }) {
  const [f, setF] = useState(defaultFilters);
  const types = useMemo(() => [...new Set(vm.alerts.map(a => a.title))], [vm.alerts]);
  const open = vm.alerts.filter(a => isOpen(statusOf(a)));
  const top = open.filter(a => a.kind !== 'away').slice(0, 2);
  const rows = applyFilters(vm.alerts, f, a => ({ patient: a.patient, type: a.title, status: statusOf(a), date: vm.events.find(e => e.id === a.id)?.date }));
  return (
    <>
      <PageHeading title="Alerts" sub="Prioritized safety alerts requiring staff review" />
      {top.map(a => (
        <article key={a.id} className={`alert-card ${a.kind}`}>
          <span className="alert-icon"><Icon name="bell" size={20} /></span>
          <div className="alert-body">
            <small>{a.cardLabel.toUpperCase()}</small>
            <h3>{a.patient} · {a.room}</h3>
            <p>{[a.confidence != null && `Confidence ${a.confidence}%`, a.risk != null && `Risk score ${a.risk}%`, `Detected ${a.detected}`].filter(Boolean).join(' · ')}</p>
          </div>
          <button className="btn white" onClick={() => (a.kind === 'fall' ? openEvent(a.eventId) : openPatient(a.patient))}>{a.kind === 'fall' ? 'View event' : 'View patient'}</button>
          <button className={`btn ${a.kind === 'fall' ? 'danger' : 'primary'}`} onClick={() => onAck(a.id)}>Acknowledge</button>
        </article>
      ))}
      <Panel title="All alerts" sub="Ordered by severity and time" action={<span className="count-pill">{open.length} open</span>}>
        <FilterBar patients={vm.patients.map(p => p.id)} types={types} value={f} onChange={setF} />
        <ul className="alert-list">
          {rows.map(a => (
            <li key={a.id}>
              <button onClick={() => openEvent(a.eventId)}>
                <Pill level={a.tag}>{a.tag}</Pill>
                <span className="al-main"><b>{a.title}</b><small>{a.patient} · {a.room}</small></span>
                <span className="al-ago">{a.ago}</span><span className="al-status">{statusOf(a)}</span><Icon name="chevron" size={17} />
              </button>
            </li>
          ))}
        </ul>
        {!rows.length && <Empty>No alerts. Fall, high-risk and signal-lost events from the backend appear here.</Empty>}
      </Panel>
    </>
  );
}

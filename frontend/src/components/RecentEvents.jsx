import { useMemo, useState } from 'react';
import { Panel, FilterBar, defaultFilters, applyFilters, RiskPill, StatusBadge, Empty } from './ui';

/** Compact table used on Dashboard ("Recent events") and Patient detail ("Event history"). */
export default function RecentEvents({ title, sub, events, patients, onAll, filters = true, statusOf = e => e.status, onRow }) {
  const [f, setF] = useState(defaultFilters);
  const types = useMemo(() => [...new Set(events.map(e => e.type))], [events]);
  const rows = applyFilters(events, f, e => ({ patient: e.patient, type: e.type, status: statusOf(e), date: e.date })).slice(0, 5);
  return (
    <Panel title={title} sub={sub} action={onAll && <button className="btn outline" onClick={onAll}>View all events</button>}>
      {filters && <FilterBar patients={patients.map(p => p.id)} types={types} value={f} onChange={setF} />}
      <table className={`table ${onRow ? 'rows-hover' : ''}`}>
        <thead><tr><th>TIME</th><th>PATIENT</th><th>EVENT</th><th>RISK</th><th>STATUS</th></tr></thead>
        <tbody>
          {rows.map(e => <tr key={e.id} onClick={() => onRow?.(e.id)}><td>{e.time}</td><td>{e.patient}</td><td>{e.type}</td><td><RiskPill value={e.risk} /></td><td>{statusOf(e)}</td></tr>)}
        </tbody>
      </table>
      {!rows.length && <Empty>No events received from the backend{filters ? ' for the selected filters' : ''}.</Empty>}
    </Panel>
  );
}

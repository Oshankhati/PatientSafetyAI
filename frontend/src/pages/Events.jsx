import { useMemo, useState } from 'react';
import Icon from '../components/Icon';
import { Panel, PageHeading, FilterBar, defaultFilters, applyFilters, RiskPill, Empty } from '../components/ui';
import { csvDownload } from '../utils';

const PAGE = 10;

export default function Events({ vm, statusOf, openEvent }) {
  const [f, setF] = useState(defaultFilters);
  const [page, setPage] = useState(1);
  const types = useMemo(() => [...new Set(vm.events.map(e => e.type))], [vm.events]);
  const filtered = applyFilters(vm.events, f, e => ({ patient: e.patient, type: e.type, status: statusOf(e), date: e.date }));
  const pages = Math.max(1, Math.ceil(filtered.length / PAGE));
  const cur = Math.min(page, pages);
  const shown = filtered.slice((cur - 1) * PAGE, cur * PAGE);
  const nums = []; for (let n = Math.max(1, cur - 2); n <= Math.min(pages, Math.max(cur + 2, 5)); n++) nums.push(n);
  const exportCsv = () => csvDownload('events.csv', ['Event ID', 'Time', 'Patient', 'Room', 'Event type', 'Risk %', 'Confidence %', 'Status'],
    filtered.map(e => [e.id, e.date ? e.date.toISOString() : e.time, e.patient, e.room, e.type, e.risk, e.confidence, statusOf(e)]));
  return (
    <>
      <PageHeading title="Events" sub="Searchable history of model detections and system events"
        right={<button className="btn outline" onClick={exportCsv} disabled={!filtered.length}><Icon name="download" size={16} />Export CSV</button>} />
      <Panel>
        <FilterBar patients={vm.patients.map(p => p.id)} types={types} value={f} onChange={v => { setF(v); setPage(1); }} />
        <table className="table rows-hover">
          <thead><tr><th>EVENT ID</th><th>DATE / TIME</th><th>PATIENT</th><th>ROOM</th><th>EVENT TYPE</th><th>RISK</th><th>CONFIDENCE</th><th>STATUS</th></tr></thead>
          <tbody>
            {shown.map(e => (
              <tr key={e.id} onClick={() => openEvent(e.id)}>
                <td>{e.id}</td><td>{e.date ? `${e.date.toLocaleDateString('en-GB', { day: '2-digit', month: 'short' })}, ${e.time}` : e.time}</td><td>{e.patient}</td><td>{e.room}</td><td>{e.type}</td>
                <td><RiskPill value={e.risk} /></td><td>{e.confidence == null ? '—' : `${e.confidence}%`}</td><td>{statusOf(e)}</td>
              </tr>
            ))}
          </tbody>
        </table>
        {!shown.length && <Empty>No events received from the backend{vm.events.length ? ' for the selected filters' : ''}.</Empty>}
        <div className="pager">
          <span>Showing {shown.length ? `${(cur - 1) * PAGE + 1}–${(cur - 1) * PAGE + shown.length}` : '0'} of {filtered.length} events</span>
          <div>
            <button className="btn outline" disabled={cur === 1} onClick={() => setPage(cur - 1)}>Previous</button>
            {nums.map(n => <button key={n} className={`btn page ${cur === n ? 'on' : ''}`} onClick={() => setPage(n)}>{n}</button>)}
            <button className="btn outline" disabled={cur >= pages} onClick={() => setPage(cur + 1)}>Next</button>
          </div>
        </div>
      </Panel>
    </>
  );
}

import { useState } from 'react';
import Icon from './Icon';
import { levelClass, statusClass } from '../utils';

export const Pill = ({ level, children }) => (
  <span className={`pill ${levelClass(level)}`}><i />{children ?? level}</span>
);

export const RiskPill = ({ value }) => {
  if (value == null) return <span className="dash">—</span>;
  return value >= 70 ? <span className="pill high"><i />{value}%</span> : <span>{value}%</span>;
};

export const StatusBadge = ({ status }) => <span className={`status-badge ${statusClass(status)}`}>{status}</span>;

export const Panel = ({ title, sub, action, children, className = '' }) => (
  <section className={`panel ${className}`}>
    {(title || action) && (
      <div className="panel-head">
        <div>{title && <h2>{title}</h2>}{sub && <p>{sub}</p>}</div>
        {action}
      </div>
    )}
    {children}
  </section>
);

export const Eyebrow = ({ children }) => <div className="eyebrow">{children}</div>;

export const PageHeading = ({ title, sub, right }) => (
  <div className="page-heading"><div><h1>{title}</h1><p>{sub}</p></div>{right}</div>
);

export const Select = ({ label, value, onChange, options, className = '' }) => (
  <label className={`field ${className}`}>
    {label && <span>{label}</span>}
    <select value={value} onChange={e => onChange?.(e.target.value)}>{options.map(o => <option key={o}>{o}</option>)}</select>
  </label>
);

export function FilterBar({ patients, types = [], value, onChange }) {
  const [more, setMore] = useState(false);
  return (
    <div className="filter-bar">
      <Select label="Patient" value={value.patient} onChange={v => onChange({ ...value, patient: v })} options={['All patients', ...patients]} />
      <Select label="Event type" value={value.type} onChange={v => onChange({ ...value, type: v })} options={['All events', ...types]} />
      <Select label="Date" value={value.date} onChange={v => onChange({ ...value, date: v })} options={['All time', 'Today', 'Yesterday', 'Last 7 days']} />
      <button className="btn outline" onClick={() => setMore(m => !m)} aria-expanded={more}><Icon name="filter" size={16} />More filters</button>
      {more && (
        <Select label="Status" value={value.status} onChange={v => onChange({ ...value, status: v })} options={['Any status', 'Open', 'Acknowledged', 'Resolved', 'Investigating']} />
      )}
    </div>
  );
}

export const defaultFilters = { patient: 'All patients', type: 'All events', date: 'All time', status: 'Any status' };

const DAY = 86400000;
function dateOk(filter, date) {
  if (filter === 'All time' || !date) return true;            // events without an absolute timestamp are never hidden by date
  const now = new Date(); const start = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const t = date.getTime();
  if (filter === 'Today') return t >= start;
  if (filter === 'Yesterday') return t >= start - DAY && t < start;
  return t >= start - 6 * DAY;
}

export function applyFilters(rows, f, get) {
  return rows.filter(r => {
    const { patient, type, status, date } = get(r);
    return (f.patient === 'All patients' || patient === f.patient)
      && (f.type === 'All events' || type === f.type)
      && (f.status === 'Any status' || status === f.status)
      && dateOk(f.date, date);
  });
}

export const MetaStrip = ({ items }) => (
  <div className="meta-strip" style={{ gridTemplateColumns: `repeat(${items.length}, 1fr)` }}>
    {items.map(([v, l]) => <div key={l}><b>{v}</b><small>{l}</small></div>)}
  </div>
);

export const Empty = ({ children }) => <div className="empty">{children}</div>;

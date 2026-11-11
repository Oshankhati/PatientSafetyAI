import { useMemo, useState } from 'react';
import Icon from '../components/Icon';
import { BarChart, Donut, RiskLine } from '../components/Charts';
import { Panel, PageHeading, Select, Empty } from '../components/ui';
import { csvDownload } from '../utils';

const RANGES = { 'Last 24 hours': 1, 'Last 7 days': 7, 'Last 30 days': 30, 'All time': null };
const DAYS = ['S', 'M', 'T', 'W', 'T', 'F', 'S'];

export default function Analytics({ vm, history, th }) {
  const [range, setRange] = useState('Last 7 days');
  const d = useMemo(() => {
    const days = RANGES[range];
    const cutoff = days ? Date.now() - days * 86400000 : 0;
    const ev = vm.events.filter(e => !e.date || e.date.getTime() >= cutoff);
    const alerts = ev.filter(e => e.kind !== 'info');
    const falls = ev.filter(e => e.kind === 'fall');
    const risks = ev.map(e => e.risk).filter(v => v != null);
    const pts = Object.values(history).flat().filter(([t]) => t >= cutoff).sort((a, b) => a[0] - b[0]);
    const avg = pts.length ? Math.round(pts.reduce((s, [, v]) => s + v, 0) / pts.length) : risks.length ? Math.round(risks.reduce((a, b) => a + b, 0) / risks.length) : null;
    const hi = ev.filter(e => e.risk != null && e.risk >= th.high * 100);
    // falls per day, last 7 calendar days (needs absolute timestamps)
    const week = [...Array(7)].map((_, i) => { const dt = new Date(); dt.setDate(dt.getDate() - (6 - i)); return dt; });
    const perDay = week.map(dt => [DAYS[dt.getDay()], falls.filter(e => e.date && e.date.toDateString() === dt.toDateString()).length]);
    const byRoom = Object.entries(alerts.reduce((m, e) => ({ ...m, [e.room]: (m[e.room] || 0) + 1 }), {})).sort((a, b) => b[1] - a[1]).slice(0, 6);
    const lv = n => vm.patients.filter(p => p.level === n).length, tot = vm.patients.length;
    const dist = tot ? [['Normal', Math.round((lv('Normal') / tot) * 100), '#2a8f5e'], ['High risk', Math.round(((lv('High') + lv('Medium')) / tot) * 100), '#b4501f'], ['Fall', vm.pose?.fallConfirmed ? Math.round(100 / tot) : 0, '#b71c24'], ['No signal', Math.round((lv('Away') / tot) * 100), '#9aaeb3']].filter(x => x[1] > 0) : [];
    return { ev, alerts, falls, avg, hi, perDay, byRoom, dist, pts };
  }, [vm, history, range, th]);

  const exportCsv = () => csvDownload('analytics-events.csv', ['Event ID', 'Time', 'Patient', 'Room', 'Event type', 'Risk %'], d.ev.map(e => [e.id, e.date ? e.date.toISOString() : e.time, e.patient, e.room, e.type, e.risk]));
  const kpis = [
    ['Monitored patients', vm.patients.length, `${vm.rooms.length} room${vm.rooms.length === 1 ? '' : 's'}`],
    ['Total alerts', d.alerts.length, range.toLowerCase()],
    ['Falls', d.falls.length, `${d.falls.filter(e => e.date && new Date().toDateString() === e.date.toDateString()).length} today`],
    ['Average risk', d.avg == null ? '—' : `${d.avg}%`, d.pts.length ? 'from live readings' : 'from event scores'],
    ['High-risk events', d.hi.length, d.ev.length ? `${Math.round((d.hi.length / d.ev.length) * 100)}% of events` : '—'],
  ];
  const noDays = d.perDay.every(x => x[1] === 0);
  return (
    <>
      <PageHeading title="Analytics" sub="Patient safety trends and operational insights"
        right={<div className="head-actions"><Select label="Date range" value={range} onChange={setRange} options={Object.keys(RANGES)} />
          <button className="btn outline" onClick={exportCsv} disabled={!d.ev.length}><Icon name="download" size={16} />Export CSV</button></div>} />
      <div className="kpi-grid">{kpis.map(([l, v, s]) => <article className="kpi" key={l}><small>{l}</small><strong>{v}</strong><span>{s}</span></article>)}</div>
      <div className="two-col">
        <Panel title="Falls per day" sub="Confirmed and detected events">{noDays && !d.falls.length ? <Empty>No falls recorded in this range.</Empty> : <BarChart data={d.perDay} />}</Panel>
        <Panel title="Risk distribution" sub="Current patient states">{d.dist.length ? <Donut data={d.dist} total={vm.patients.length} /> : <Empty>No patients returned by the backend.</Empty>}</Panel>
        <Panel title="Alerts by room" sub={range}>{d.byRoom.length ? <BarChart data={d.byRoom} /> : <Empty>No alerts in this range.</Empty>}</Panel>
        <Panel title="Average risk over time" sub="All monitored patients"><RiskLine points={d.pts} height={230} high={th.high * 100} fall={th.fall * 100} /></Panel>
      </div>
    </>
  );
}

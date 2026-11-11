import { useState } from 'react';
import { Panel, PageHeading, Select, Pill, Empty } from '../components/ui';

export default function Patients({ vm, openPatient }) {
  const [status, setStatus] = useState('All statuses');
  const statuses = [...new Set(vm.patients.map(p => p.status))];
  const rows = vm.patients.filter(p => status === 'All statuses' || p.status === status);
  return (
    <>
      <PageHeading title="Patients" sub="Anonymous patients currently assigned to monitored rooms"
        right={<Select label="Status" value={status} onChange={setStatus} options={['All statuses', ...statuses]} />} />
      <Panel>
        <table className="table rows-hover">
          <thead><tr><th>PATIENT</th><th>ROOM / BED</th><th>STATUS</th><th>RISK SCORE</th><th>RISK LEVEL</th><th /></tr></thead>
          <tbody>
            {rows.map(p => (
              <tr key={p.id} onClick={() => openPatient(p.id)}>
                <td>{p.id}</td><td>{p.room}{p.bed ? ` · ${p.bed}` : ''}</td><td>{p.status}</td>
                <td>{p.risk == null ? '—' : `${p.risk}%`}</td><td><Pill level={p.level} /></td>
                <td className="right"><button className="link" onClick={e => { e.stopPropagation(); openPatient(p.id); }}>View <span>›</span></button></td>
              </tr>
            ))}
          </tbody>
        </table>
        {!rows.length && <Empty>No patients returned by the backend (GET /patients).</Empty>}
      </Panel>
    </>
  );
}

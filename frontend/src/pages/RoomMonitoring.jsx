import Icon from '../components/Icon';
import { PageHeading, Pill, Empty } from '../components/ui';

export default function RoomMonitoring({ vm, openPatient }) {
  return (
    <>
      <PageHeading title="Room Monitoring" sub="Status of monitored rooms and assigned patients" />
      <div className="room-grid">
        {vm.rooms.map(r => (
          <article key={r.id} className="room-card" onClick={() => r.patient !== '—' && openPatient(r.patient)}>
            <div className="row-between"><small className="cap">ROOM</small><Pill level={r.level} /></div>
            <h3>{r.id}</h3>
            <div className="assigned"><Icon name="users" size={18} stroke={1.5} /><span><small>Assigned patient</small><b>{r.patient}</b></span></div>
            <div className="room-foot"><span>{r.level === 'Away' ? 'No signal' : r.status}</span><b>{r.risk == null ? '—' : `${r.risk}%`}</b></div>
            {r.level === 'Away' && <div className="room-note">No live signal for this room</div>}
          </article>
        ))}
      </div>
      {!vm.rooms.length && <Empty>No rooms returned by the backend (GET /rooms).</Empty>}
    </>
  );
}

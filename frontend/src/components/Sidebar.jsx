import Icon from './Icon';

const ITEMS = [['dashboard', 'Dashboard', 'grid'], ['patients', 'Patients', 'users'], ['live', 'Live Monitoring', 'camera'], ['alerts', 'Alerts', 'bell'], ['events', 'Events', 'events'], ['rooms', 'Room Monitoring', 'room'], ['analytics', 'Analytics', 'chart'], ['settings', 'Settings', 'settings']];

export default function Sidebar({ active, onNavigate, alertCount, online, healthy = 0, total = 0 }) {
  return (
    <aside className="sidebar">
      <div className="brand">
        <span className="brand-mark"><Icon name="plus" size={26} stroke={2.6} /></span>
        <span><b>PatientSafetyAI</b><small>Clinical Monitoring</small></span>
      </div>
      <nav className="main-nav">
        {ITEMS.map(([id, label, icon]) => (
          <button key={id} className={`nav-item ${active === id ? 'active' : ''}`} onClick={() => onNavigate(id)}>
            <span className="nav-icon"><Icon name={icon} /></span><span>{label}</span>
            {id === 'alerts' && alertCount > 0 && <i className="nav-badge">{alertCount}</i>}
          </button>
        ))}
      </nav>
      <div className="sidebar-bottom">
        <div className="presentation">Presentation controls</div>
        <button className={`side-row ${active === 'status' ? 'active' : ''}`} onClick={() => onNavigate('status')}>
          <span className={`online-dot ${online ? '' : 'offline'}`} />
          <span><b>{online ? 'System Online' : 'System Offline'}</b><small>{online ? `${healthy} of ${total} services healthy` : 'Waiting for backend'}</small></span>
          <Icon name="chevron" size={15} />
        </button>
        <div className="side-row user">
          <span className="avatar">AN</span><span><b>Alex Nurse</b><small>Nurse</small></span>
          <button className="icon-btn" title="Sign out" aria-label="Sign out"><Icon name="logout" size={17} /></button>
        </div>
      </div>
    </aside>
  );
}

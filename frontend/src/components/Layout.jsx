import Sidebar from './Sidebar';
import Topbar from './Topbar';

export default function Layout({ active, onNavigate, alertCount, online, healthy, total, notifications, children, banner }) {
  return (
    <div className="app-shell">
      <Sidebar active={active} onNavigate={onNavigate} alertCount={alertCount} online={online} healthy={healthy} total={total} />
      <main className="main-area">
        <Topbar unread={alertCount} notifications={notifications} onOpen={onNavigate} />
        <section className="page-content">{banner}{children}</section>
      </main>
    </div>
  );
}

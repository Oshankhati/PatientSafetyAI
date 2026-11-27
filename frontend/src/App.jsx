import { useCallback, useEffect, useMemo, useState } from 'react';
import Layout from './components/Layout';
import Icon from './components/Icon';
import useMonitoring from './hooks/useMonitoring';
import { buildServices, buildViewModel } from './data/model';
import { DEFAULT_SETTINGS } from './data/defaults';
import Dashboard from './pages/Dashboard';
import Patients from './pages/Patients';
import PatientDetail from './pages/PatientDetail';
import LiveMonitoring from './pages/LiveMonitoring';
import Alerts from './pages/Alerts';
import Events from './pages/Events';
import EventDetail from './pages/EventDetail';
import RoomMonitoring from './pages/RoomMonitoring';
import Analytics from './pages/Analytics';
import Settings from './pages/Settings';
import SystemStatus from './pages/SystemStatus';
import Login from './pages/Login';
import Signup from './pages/Signup';
import { apiRequest, acknowledgeEvent, loadSettings, saveSettings } from './services/api';

function readSession(key) {
  try { return JSON.parse(sessionStorage.getItem(key)); } catch { return null; }
}

function DashboardApp({ staff, accessToken, onLogout }) {
  const raw = useMonitoring(accessToken, onLogout);
  const [settings, setSettings] = useState(DEFAULT_SETTINGS);
  const [settingsError, setSettingsError] = useState('');
  const [acks, setAcks] = useState({});
  const [view, setView] = useState({ page: 'dashboard' });

  useEffect(() => {
    let active = true;
    loadSettings(accessToken).then(value => {
      if (active) setSettings({ ...DEFAULT_SETTINGS, ...value });
    }).catch(error => {
      if (error.status === 401) onLogout();
      else if (active) setSettingsError(error.message);
    });
    return () => { active = false; };
  }, [accessToken, onLogout]);

  const th = useMemo(() => {
    const n = (v, d) => (Number.isFinite(parseFloat(v)) ? parseFloat(v) : d);
    return { medium: n(settings.ai.medium, 0.4), high: n(settings.ai.high, 0.7), fall: n(settings.ai.fall, 0.9) };
  }, [settings]);

  const vm = useMemo(() => buildViewModel(raw, th), [raw.state, raw.events, raw.patients, raw.rooms, th]); // eslint-disable-line react-hooks/exhaustive-deps
  const services = useMemo(() => buildServices({ health: raw.health, online: raw.online, latency: raw.latency, ws: raw.ws, poseAt: raw.poseAt }),
    [raw.health, raw.online, raw.latency, raw.ws, raw.poseAt, raw.updated]); // eslint-disable-line react-hooks/exhaustive-deps
  const healthy = services.filter(s => s.tone === 'ok').length;

  const statusOf = useCallback(
    x => (acks[x.id] || raw.acknowledgements?.[x.id] ? 'Acknowledged' : x.status),
    [acks, raw.acknowledgements],
  );
  const onAck = async id => {
    try {
      const acknowledgement = await acknowledgeEvent(accessToken, id);
      setAcks(current => ({ ...current, [id]: acknowledgement }));
    } catch (error) {
      if (error.status === 401) onLogout();
      else setSettingsError(`Could not acknowledge alert: ${error.message}`);
    }
  };
  const onSaveSettings = async nextSettings => {
    try {
      const saved = await saveSettings(accessToken, nextSettings);
      setSettings(saved);
      setSettingsError('');
      return true;
    } catch (error) {
      if (error.status === 401) onLogout();
      else setSettingsError(`Could not save settings: ${error.message}`);
      return false;
    }
  };
  const go = (page, extra = {}) => { setView({ page, ...extra }); window.scrollTo(0, 0); };
  const openPatient = id => go('patients', { patient: id });
  const openEvent = id => go('events', { event: id });

  const openAlerts = vm.alerts.filter(a => ['Open', 'Investigating'].includes(statusOf(a)));
  const notifications = openAlerts.slice(0, 6).map(a => ({ id: a.id, tag: a.tag, title: `${a.title} — ${a.patient}`, ago: a.ago, go: 'alerts' }));
  const { page } = view;

  const pages = {
    dashboard: <Dashboard vm={vm} online={raw.online} healthy={healthy} total={services.length} go={go} statusOf={statusOf} updatedAt={raw.poseAt ? new Date(raw.poseAt) : null} />,
    patients: view.patient
      ? <PatientDetail vm={vm} id={view.patient} back={() => go('patients')} history={raw.history} th={th} statusOf={statusOf} />
      : <Patients vm={vm} openPatient={openPatient} />,
    live: <LiveMonitoring vm={vm} th={th} updatedAt={raw.poseAt ? new Date(raw.poseAt) : null} accessToken={accessToken} onLogout={onLogout} />,
    alerts: <Alerts vm={vm} statusOf={statusOf} onAck={onAck} openEvent={openEvent} openPatient={openPatient} />,
    events: view.event
    ? <EventDetail vm={vm} id={view.event} back={() => go('events')} statusOf={statusOf} acks={{ ...raw.acknowledgements, ...acks }} />
      : <Events vm={vm} statusOf={statusOf} openEvent={openEvent} />,
    rooms: <RoomMonitoring vm={vm} openPatient={openPatient} />,
    analytics: <Analytics vm={vm} history={raw.history} th={th} />,
    settings: <Settings vm={vm} settings={settings} onSave={onSaveSettings} error={settingsError} />,
    status: <SystemStatus services={services} online={raw.online} updatedAt={raw.updated} />,
  };

  const banner = (
    <>
      {!raw.online && raw.updated && (
        <div className="conn-banner">Cannot reach the backend (GET /health failed). Start it with <code>uvicorn backend.main:app --port 8000</code>; the page reconnects automatically.</div>
      )}
      {settingsError && (
        <div className="conn-banner" role="alert">{settingsError}</div>
      )}
      {vm.pose?.fallConfirmed && page !== 'live' && (
        <div className="banner fall-banner"><Icon name="alert" size={18} />
          <div><b>Fall detected{vm.pose.patient ? ` — ${vm.pose.patient}` : ''}{vm.pose.room ? ` · ${vm.pose.room}` : ''}</b><span>The model reports a confirmed fall right now.</span></div>
          <button className="btn white" onClick={() => go('live')}>Open live view</button></div>
      )}
    </>
  );

  return (
    // <Layout active={page} onNavigate={go} alertCount={openAlerts.length} online={raw.online} healthy={healthy} total={services.length} notifications={notifications} banner={banner}>
    <Layout
  active={page}
  onNavigate={go}
  alertCount={openAlerts.length}
  online={raw.online}
  healthy={healthy}
  total={services.length}
  notifications={notifications}
  banner={banner}
  staff={staff}
  onLogout={onLogout}
>
      {pages[page]}
    </Layout>
  );
}

export default function App() {
  const [staff, setStaff] = useState(() => readSession('psai.staff'));
  const [accessToken, setAccessToken] = useState(() => readSession('psai.accessToken'));
  const [showSignup, setShowSignup] = useState(false);
  const [authReady, setAuthReady] = useState(false);

  useEffect(() => {
    if (!staff || !accessToken) {
      setStaff(null);
      setAccessToken(null);
      setAuthReady(true);
      return;
    }
    let active = true;
    apiRequest('/auth/me', { token: accessToken })
      .then(({ data }) => {
        if (active) {
          setStaff(data.staff);
          sessionStorage.setItem('psai.staff', JSON.stringify(data.staff));
        }
      })
      .catch(() => {
        if (active) {
          setStaff(null);
          setAccessToken(null);
          sessionStorage.removeItem('psai.staff');
          sessionStorage.removeItem('psai.accessToken');
        }
      })
      .finally(() => { if (active) setAuthReady(true); });
    return () => { active = false; };
  }, []);

  const handleLogin = (staffData, token) => {
    setStaff(staffData);
    setAccessToken(token);
    sessionStorage.setItem('psai.staff', JSON.stringify(staffData));
    sessionStorage.setItem('psai.accessToken', JSON.stringify(token));
    setAuthReady(true);
    setShowSignup(false);
  };

  const handleLogout = useCallback(() => {
    setStaff(null);
    setAccessToken(null);
    sessionStorage.removeItem('psai.staff');
    sessionStorage.removeItem('psai.accessToken');
    setAuthReady(true);
  }, []);

  if (!authReady) return <main className="login-page">Checking your session…</main>;

  if (!staff) {
    if (showSignup) {
      return (
        <Signup
          onSignup={() => setShowSignup(false)}
          onBackToLogin={() => setShowSignup(false)}
        />
      );
    }

    return (
      <Login
        onLogin={handleLogin}
        onSignup={() => setShowSignup(true)}
      />
    );
  }

  return <DashboardApp staff={staff} accessToken={accessToken} onLogout={handleLogout} />;
}

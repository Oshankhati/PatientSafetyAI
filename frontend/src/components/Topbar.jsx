
import { useEffect, useState } from 'react';
import Icon from './Icon';
import { clock } from '../utils';
import { Pill } from './ui';

export default function Topbar({ unread, notifications, onOpen, staff }) {
  const [open, setOpen] = useState(false);
  const [now, setNow] = useState(new Date());

  useEffect(() => {
    const id = setInterval(() => setNow(new Date()), 15000);
    return () => clearInterval(id);
  }, []);

  const fullName = staff?.full_name || 'Staff Member';
  const role = staff?.role || 'Nurse';

  const initials = fullName
    .trim()
    .split(/\s+/)
    .map(part => part[0])
    .join('')
    .slice(0, 2)
    .toUpperCase();

  return (
    <header className="topbar">
      <div className="notif-wrap">
        <button
          className="bell"
          aria-label="Notifications"
          aria-expanded={open}
          onClick={() => setOpen(o => !o)}
        >
          <Icon name="bell" size={20} />
          {unread > 0 && (
            <span>{unread > 99 ? '99+' : unread}</span>
          )}
        </button>

        {open && (
          <div className="notif-panel" role="dialog" aria-label="Notifications">
            <div className="notif-head">
              <div>
                <b>Notifications</b>
                <small>{unread} open</small>
              </div>
              <button
                className="icon-btn"
                onClick={() => setOpen(false)}
                aria-label="Close"
              >
                <Icon name="x" size={16} />
              </button>
            </div>

            {notifications.length ? (
              notifications.map(n => (
                <button
                  key={n.id}
                  className="notif-row"
                  onClick={() => {
                    setOpen(false);
                    onOpen(n.go);
                  }}
                >
                  <Pill level={n.tag}>{n.tag}</Pill>
                  <span>
                    <b>{n.title}</b>
                    <small>{n.ago}</small>
                  </span>
                </button>
              ))
            ) : (
              <div className="empty">No notifications</div>
            )}

            <div className="notif-foot">
              Select a notification to view details
            </div>
          </div>
        )}
      </div>

      <div className="date-text">{clock(now)}</div>

      <div className="profile">
        <span className="avatar">{initials}</span>
        <span>
          <b>{fullName}</b>
          <small>{role}</small>
        </span>
        <Icon name="chevron" size={15} />
      </div>
    </header>
  );
}

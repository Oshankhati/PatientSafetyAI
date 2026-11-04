export const pct = v => (v == null ? '—' : `${Math.round(Number(v) <= 1 ? Number(v) * 100 : Number(v))}%`);

export function levelClass(level) {
  const l = String(level || '').toLowerCase();
  if (/high|critical|fall/.test(l)) return 'high';
  if (/medium/.test(l)) return 'medium';
  if (/normal|low/.test(l)) return 'normal';
  if (/away|no signal/.test(l)) return 'away';
  return 'neutral';
}

export const statusClass = s => String(s || '').toLowerCase().replace(/\s+/g, '-');

export function clock(d = new Date()) {
  const date = d.toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' });
  const time = d.toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false });
  return `${date}, ${time}`;
}

/** Smooth SVG path (monotone-ish Catmull-Rom) through points [[x,y],...] */
export function smoothPath(pts) {
  if (pts.length < 2) return '';
  let d = `M${pts[0][0]},${pts[0][1]}`;
  for (let i = 0; i < pts.length - 1; i++) {
    const p0 = pts[i - 1] || pts[i], p1 = pts[i], p2 = pts[i + 1], p3 = pts[i + 2] || p2;
    const c1 = [p1[0] + (p2[0] - p0[0]) / 6, p1[1] + (p2[1] - p0[1]) / 6];
    const c2 = [p2[0] - (p3[0] - p1[0]) / 6, p2[1] - (p3[1] - p1[1]) / 6];
    d += ` C${c1[0]},${c1[1]} ${c2[0]},${c2[1]} ${p2[0]},${p2[1]}`;
  }
  return d;
}

export function timeLabel(d) {
  return d ? d.toLocaleTimeString('en-GB', { hour12: false }) : '—';
}
export function ago(d, now = Date.now()) {
  if (!d) return '—';
  const s = Math.max(0, Math.round((now - d.getTime()) / 1000));
  if (s < 10) return 'just now';
  if (s < 60) return `${s} sec ago`;
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  if (s < 86400) return `${Math.round(s / 3600)} h ago`;
  return `${Math.round(s / 86400)} d ago`;
}
export const sameDay = (a, b) => a.toDateString() === b.toDateString();
export function csvDownload(name, head, rows) {
  const csv = [head, ...rows].map(r => r.map(c => `"${String(c ?? '').replace(/"/g, '""')}"`).join(',')).join('\n');
  const a = Object.assign(document.createElement('a'), { href: URL.createObjectURL(new Blob([csv], { type: 'text/csv' })), download: name });
  a.click(); URL.revokeObjectURL(a.href);
}


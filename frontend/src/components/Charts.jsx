import { useEffect, useRef, useState } from 'react';
import { smoothPath } from '../utils';

// Measures its container so SVG text keeps its natural size (no stretching).
function useWidth(fallback = 700) {
  const ref = useRef(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    if (!ref.current) return undefined;
    const ro = new ResizeObserver(([e]) => setW(Math.max(280, Math.round(e.contentRect.width))));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w];
}

export function BarChart({ data, height = 230, labelTop = true }) {
  const W = 560, padL = 14, padB = 28, padT = 26, H = height;
  const max = Math.max(...data.map(d => d[1]), 1);
  const slot = (W - padL * 2) / data.length, bw = Math.min(46, slot * 0.58);
  return (
    <svg className="chart" viewBox={`0 0 ${W} ${H}`} role="img">
      {data.map(([l, v], i) => {
        const h = Math.max(v ? ((H - padB - padT) * v) / max : 0, 0);
        const x = padL + slot * i + (slot - bw) / 2, y = H - padB - h;
        return (
          <g key={i}>
            {v === 0 ? <rect x={x} y={H - padB - 1.5} width={bw} height={2} fill="#2a9a95" rx="1" /> : <rect x={x} y={y} width={bw} height={h} rx="4" fill="#3a9f99" />}
            {labelTop && <text x={x + bw / 2} y={y - 6} textAnchor="middle" className="ct">{v}</text>}
            <text x={x + bw / 2} y={H - 8} textAnchor="middle" className="ca">{l}</text>
          </g>
        );
      })}
    </svg>
  );
}

export function Donut({ data, total }) {
  const R = 64, r = 46, C = 80, sum = data.reduce((s, d) => s + d[1], 0);
  let a0 = -Math.PI / 2;
  const arcs = data.map(([name, v, color]) => {
    const a1 = a0 + (v / sum) * Math.PI * 2, large = a1 - a0 > Math.PI ? 1 : 0;
    const p = (rad, a) => [C + rad * Math.cos(a), C + rad * Math.sin(a)];
    const [x0, y0] = p(R, a0), [x1, y1] = p(R, a1), [x2, y2] = p(r, a1), [x3, y3] = p(r, a0);
    const d = `M${x0},${y0} A${R},${R} 0 ${large} 1 ${x1},${y1} L${x2},${y2} A${r},${r} 0 ${large} 0 ${x3},${y3}Z`;
    a0 = a1; return <path key={name} d={d} fill={color} />;
  });
  return (
    <div className="donut-wrap">
      <svg viewBox="0 0 160 160" className="donut">
        {arcs}
        <text x="80" y="82" textAnchor="middle" className="dn">{total}</text>
        <text x="80" y="98" textAnchor="middle" className="dl">patients</text>
      </svg>
      <ul className="legend">
        {data.map(([n, v, c]) => <li key={n}><i style={{ background: c }} /><span>{n}</span><b>{v}%</b></li>)}
      </ul>
    </div>
  );
}

const hhmm = t => new Date(t).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false });

/** points: [[epochMs, percent], ...] — real readings recorded from the live risk score. */
export function RiskLine({ points, height = 250, high = 70, fall = 90 }) {
  const [ref, W] = useWidth();
  const H = height, L = 38, R = 8, T = 22, B = 26;
  if (!points || points.length < 2) {
    return <div ref={ref} className="chart-empty" style={{ height }}>{points?.length ? 'Collecting readings…' : 'No risk readings yet. They appear here as the backend reports live risk scores.'}</div>;
  }
  const t0 = points[0][0], t1 = points[points.length - 1][0], span = Math.max(1, t1 - t0);
  const x = t => L + ((W - L - R) * (t - t0)) / span;
  const y = v => T + ((H - T - B) * (100 - v)) / 100;
  const pts = points.map(([t, v]) => [x(t), y(v)]);
  const line = smoothPath(pts);
  const area = `${line} L${pts[pts.length - 1][0]},${y(0)} L${pts[0][0]},${y(0)}Z`;
  const ticks = [0, 1 / 3, 2 / 3, 1].map((f, i, a) => [t0 + span * f, i === a.length - 1 ? 'Now' : hhmm(t0 + span * f)]);
  return (
    <div ref={ref} className="chart-box"><svg className="chart" width={W} height={H} viewBox={`0 0 ${W} ${H}`} role="img">
      {[0, 25, 50, 75, 100].map(g => (
        <g key={g}><line x1={L} x2={W - R} y1={y(g)} y2={y(g)} stroke="#e6edef" /><text x={L - 8} y={y(g) + 3} textAnchor="end" className="ca">{g}%</text></g>
      ))}
      <path d={area} fill="#168d8a" opacity=".1" />
      <line x1={L} x2={W - R} y1={y(fall)} y2={y(fall)} stroke="#c8605f" strokeDasharray="6 4" />
      <line x1={L} x2={W - R} y1={y(high)} y2={y(high)} stroke="#d49a6a" strokeDasharray="6 4" />
      <path d={line} fill="none" stroke="#168d8a" strokeWidth="2.4" />
      <text x={W - R - 4} y={9} textAnchor="end" className="ca">- - - High {(high / 100).toFixed(2)}   - - - Fall {(fall / 100).toFixed(2)}</text>
      {ticks.map(([t, l], i) => <text key={i} x={x(t)} y={H - 6} textAnchor={i === 3 ? 'end' : i === 0 ? 'start' : 'middle'} className="ca">{l}</text>)}
    </svg></div>
  );
}

import { useEffect, useRef } from 'react';

// COCO-17 limbs; face points are skipped, neck + spine are drawn from shoulder/hip midpoints.
const LIMBS = [[5, 7], [7, 9], [6, 8], [8, 10], [11, 13], [13, 15], [12, 14], [14, 16]];
const COLORS = { bed: ['rgba(40,160,170,.22)', '#2fb7c0'], safe: ['rgba(40,170,110,.22)', '#2fb77c'], floor: ['rgba(210,120,50,.22)', '#d9763a'], unsafe: ['rgba(200,60,60,.2)', '#d85a5a'] };

/**
 * keypoints: [[x, y, confidence], ...] normalised 0-1 to the camera frame (as sent by /monitor/state).
 * The frame is drawn letter-boxed at `aspect` so the body keeps its real proportions.
 * zones: [{label, kind, box:[x, y, w, h]}] normalised to the same frame.
 */
export default function PoseCanvas({ keypoints, zones = [], showBox = true, tag = null, height = 340, aspect = 16 / 9, emptyText = 'Waiting for live pose keypoints…' }) {
  const ref = useRef(null);

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return undefined;
    const draw = () => {
      const dpr = window.devicePixelRatio || 1;
      const W = canvas.clientWidth, H = canvas.clientHeight;
      canvas.width = W * dpr; canvas.height = H * dpr;
      const ctx = canvas.getContext('2d');
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.fillStyle = '#0d2a30'; ctx.fillRect(0, 0, W, H);
      ctx.strokeStyle = 'rgba(80,150,150,.16)'; ctx.lineWidth = 1;
      for (let x = 0; x <= W; x += 28) { ctx.beginPath(); ctx.moveTo(x + .5, 0); ctx.lineTo(x + .5, H); ctx.stroke(); }
      for (let y = 0; y <= H; y += 28) { ctx.beginPath(); ctx.moveTo(0, y + .5); ctx.lineTo(W, y + .5); ctx.stroke(); }

      const fh = Math.min(H, W / aspect), fw = fh * aspect, fx = (W - fw) / 2, fy = (H - fh) / 2;
      const X = v => fx + v * fw, Y = v => fy + v * fh;

      ctx.font = '700 10px DM Sans, Arial'; ctx.textAlign = 'center';
      zones.forEach(z => {
        const [bx, by, bw, bh] = z.box; const [fill, line] = COLORS[z.kind] || COLORS.safe;
        ctx.fillStyle = fill; ctx.strokeStyle = line; ctx.lineWidth = 1.5;
        ctx.fillRect(X(bx), Y(by), bw * fw, bh * fh); ctx.strokeRect(X(bx), Y(by), bw * fw, bh * fh);
        ctx.fillStyle = line; ctx.fillText(z.label, X(bx + bw / 2), Y(by + bh / 2) + 3);
      });

      const pts = (keypoints || []).map(k => ({ x: k[0], y: k[1], c: k[2] ?? 1 }));
      const ok = i => pts[i] && Number.isFinite(pts[i].x) && Number.isFinite(pts[i].y) && pts[i].c > 0.25 && !(pts[i].x === 0 && pts[i].y === 0);
      if (!pts.length || ![0, 5, 6, 11, 12].some(ok)) {
        ctx.fillStyle = '#9ec3c6'; ctx.font = '14px DM Sans, Arial'; ctx.textAlign = 'center';
        ctx.fillText(emptyText, W / 2, H / 2); return;
      }
      const P = pts.map(p => ({ x: X(p.x), y: Y(p.y) }));
      const mid = (a, b) => (ok(a) && ok(b) ? { x: (P[a].x + P[b].x) / 2, y: (P[a].y + P[b].y) / 2 } : null);
      const neck = mid(5, 6), hip = mid(11, 12);
      const segs = [];
      LIMBS.forEach(([a, b]) => ok(a) && ok(b) && segs.push([P[a], P[b]]));
      if (neck && ok(0)) segs.push([P[0], neck]);
      if (neck && hip) segs.push([neck, hip]);
      if (neck) [5, 6].forEach(i => ok(i) && segs.push([neck, P[i]]));
      if (hip) [11, 12].forEach(i => ok(i) && segs.push([hip, P[i]]));

      ctx.lineCap = 'round'; ctx.strokeStyle = '#7ff0e0'; ctx.lineWidth = 2.5;
      segs.forEach(([a, b]) => { ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); });
      const dots = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].filter(ok).map(i => P[i]);
      [neck, hip].forEach(p => p && dots.push(p));
      dots.forEach(p => { ctx.beginPath(); ctx.fillStyle = '#c9fff7'; ctx.strokeStyle = '#27c9c0'; ctx.lineWidth = 2; ctx.arc(p.x, p.y, 4.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke(); });

      if (showBox) {
        const used = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16].filter(ok).map(i => P[i]);
        const xs = used.map(p => p.x), ys = used.map(p => p.y), pad = 14;
        ctx.setLineDash([5, 4]); ctx.strokeStyle = '#3fd0c2'; ctx.lineWidth = 1.3;
        ctx.strokeRect(Math.min(...xs) - pad, Math.min(...ys) - pad, Math.max(...xs) - Math.min(...xs) + pad * 2, Math.max(...ys) - Math.min(...ys) + pad * 2);
        ctx.setLineDash([]);
      }
    };
    draw();
    const ro = new ResizeObserver(draw); ro.observe(canvas);
    return () => ro.disconnect();
  }, [keypoints, zones, showBox, aspect, emptyText]);

  return (
    <div className="pose-wrap" style={{ height }}>
      <canvas ref={ref} className="pose-canvas" />
      {tag !== false && <><span className="pose-tag left">{tag || 'NO PERSON TRACKED'}</span><span className="pose-tag right">SKELETON OUTPUT ONLY</span></>}
    </div>
  );
}

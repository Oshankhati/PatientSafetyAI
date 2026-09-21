import { useEffect, useMemo, useRef, useState } from "react";

const API = import.meta.env.VITE_API_URL || "http://127.0.0.1:8001";

const SKELETON = [
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
  [5, 6],
  [5, 7],
  [7, 9],
  [6, 8],
  [8, 10],
  [5, 11],
  [6, 12],
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
];

function levelClass(level) {
  if (!level) return "pill";
  if (level.includes("FALL")) return "pill danger";
  if (level.includes("HIGH")) return "pill warn";
  if (level.includes("MEDIUM") || level.includes("PRE")) return "pill mid";
  return "pill ok";
}

function SkeletonView({ skeleton }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    ctx.fillStyle = "#0b1520";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "#2a425c";
    ctx.setLineDash([4, 6]);
    ctx.strokeRect(24, 16, w - 48, h - 32);
    ctx.setLineDash([]);
    if (!skeleton?.keypoints_normalized) {
      ctx.fillStyle = "#8aa0b8";
      ctx.font = "14px Segoe UI, sans-serif";
      ctx.fillText("Waiting for COCO-17 keypoints (no camera image stored).", 36, h / 2);
      return;
    }
    const pts = skeleton.keypoints_normalized;
    const conf = skeleton.keypoint_confidence || pts.map(() => 1);
    const toXY = (i) => [pts[i][0] * w, pts[i][1] * h];
    ctx.lineWidth = 3;
    ctx.strokeStyle = "#f0a202";
    SKELETON.forEach(([a, b]) => {
      if (conf[a] < 0.3 || conf[b] < 0.3) return;
      const [x1, y1] = toXY(a);
      const [x2, y2] = toXY(b);
      if ((x1 === 0 && y1 === 0) || (x2 === 0 && y2 === 0)) return;
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    });
    pts.forEach((pt, i) => {
      if (conf[i] < 0.3) return;
      ctx.beginPath();
      ctx.fillStyle = "#3ecfff";
      ctx.arc(pt[0] * w, pt[1] * h, 4, 0, Math.PI * 2);
      ctx.fill();
    });
  }, [skeleton]);

  return <canvas ref={canvasRef} width={640} height={360} className="skel" aria-label="Skeleton monitor" />;
}

export default function App() {
  const [health, setHealth] = useState(null);
  const [state, setState] = useState(null);
  const [events, setEvents] = useState([]);
  const [patients, setPatients] = useState([]);
  const [rooms, setRooms] = useState([]);
  const [error, setError] = useState("");

  async function refresh() {
    try {
      const [h, r, e, p, rm] = await Promise.all([
        fetch(`${API}/health`).then((x) => x.json()),
        fetch(`${API}/monitor/state`).then((x) => x.json()),
        fetch(`${API}/events/latest?n=12`).then((x) => x.json()),
        fetch(`${API}/patients`).then((x) => x.json()),
        fetch(`${API}/rooms`).then((x) => x.json()),
      ]);
      setHealth(h);
      setState(r);
      setEvents(e.events || []);
      setPatients(p.patients || []);
      setRooms(rm.rooms || []);
      setError("");
    } catch (err) {
      setError(`Cannot reach API at ${API}. Start: py -3.12 -m uvicorn backend.main:app --host 127.0.0.1 --port 8001`);
    }
  }

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 1200);
    return () => clearInterval(id);
  }, []);

  const riskBody = state?.risk;
  const patient = patients[0] || { id: "Patient 01", room_id: "ICU-101", status: "MONITORING" };

  const liveNote = useMemo(() => {
    if (!health) return "Connecting…";
    return health.lstm_available
      ? "LSTM baseline loaded (domain-shifted on live YOLO)."
      : "Pose/risk heuristics only — LSTM weights not loaded.";
  }, [health]);

  return (
    <div className="page">
      <header>
        <div>
          <h1>PatientSafetyAI</h1>
          <p className="sub">Privacy-conscious fall-risk monitoring · academic prototype</p>
        </div>
        <button type="button" onClick={refresh}>
          Refresh
        </button>
      </header>

      {error ? <div className="banner">{error}</div> : null}

      <section className="grid">
        <article className="card">
          <h2>Patient</h2>
          <p>
            <strong>{state?.patient_id || patient.id}</strong>
          </p>
          <p>Room: {state?.room_id || patient.room_id}</p>
          <p>Status: {patient.status || "MONITORING"}</p>
          <p>Zone: {state?.zone || "—"}</p>
          <p className="muted">{liveNote}</p>
        </article>

        <article className="card">
          <h2>Risk</h2>
          <p className="score">{riskBody ? Number(riskBody.risk_score).toFixed(2) : "—"}</p>
          <span className={levelClass(riskBody?.risk_level)}>{riskBody?.risk_level || "NO DATA"}</span>
          <p>Prediction: {riskBody?.prediction_label || "—"}</p>
          <p className="muted">LSTM p(fall): {riskBody?.lstm_fall_probability ?? "n/a"}</p>
        </article>

        <article className="card">
          <h2>System</h2>
          <p>API: {health?.status || "down"}</p>
          <p>YOLO: {health?.yolo_model || "—"}</p>
          <p>LSTM: {health?.lstm_available ? "available" : "unavailable"}</p>
          <p className="muted">Not a medical device. Dashboard draws keypoints, not stored video.</p>
        </article>
      </section>

      <section className="card">
        <h2>Live monitor</h2>
        <p className="muted">
          COCO-17 skeleton from /predict/frame. Publish with{" "}
          <code>python demo.py --source … --publish-api http://127.0.0.1:8000</code>
        </p>
        <div className="monitor-box">
          <SkeletonView skeleton={state?.skeleton} />
          <div className="monitor-meta">
            {riskBody ? (
              <>
                <div className={levelClass(riskBody.risk_level)}>{riskBody.prediction_label}</div>
                <p>Confirmed fall: {String(riskBody.fall_confirmed)}</p>
                <p>Consecutive high frames: {riskBody.consecutive_high}</p>
                <p>Frame: {state?.frame_number ?? "—"}</p>
              </>
            ) : (
              <p>Waiting for pose traffic…</p>
            )}
          </div>
        </div>
      </section>

      <section className="card">
        <h2>Recent events</h2>
        {events.length === 0 ? (
          <p className="muted">No alerts yet. Fall alerts require temporal confirmation.</p>
        ) : (
          <ul className="events">
            {events
              .slice()
              .reverse()
              .map((ev, i) => (
                <li key={`${ev.timestamp}-${i}`}>
                  <span className="time">{ev.timestamp?.slice(11, 19) || "--"}</span>
                  <span>{ev.event_type}</span>
                  <span className="muted">
                    score {ev.risk_score} · {ev.zone}
                  </span>
                </li>
              ))}
          </ul>
        )}
      </section>

      <section className="card">
        <h2>Rooms</h2>
        <ul>
          {rooms.map((room) => (
            <li key={room.id}>
              {room.id} {room.name ? `· ${room.name}` : ""}
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

# PatientSafetyAI – frontend (live data only)

React 18 + Vite. No sample data: every screen is filled from the FastAPI backend.

    # terminal 1 (backend)
    uvicorn backend.main:app --port 8000
    # terminal 2 (this project)
    npm install
    npm run dev          # http://localhost:5175

Vite proxies /health /monitor /risk /events /patients /rooms and the WebSocket /ws/monitor to 127.0.0.1:8000
(change TARGET in vite.config.js; or set VITE_API_URL / VITE_WS_URL in a .env file).

## Where each screen gets its data
| Screen | Source |
|---|---|
| Live skeleton, risk score, prediction, fall banner | GET /monitor/state + WebSocket /ws/monitor (instant on every processed frame) |
| Events, Alerts, Notifications, Analytics | GET /events/latest?n=100 |
| Patients, Rooms, zone overlay in Settings | GET /patients, GET /rooms (rooms[].zones) |
| System Status | GET /health + WebSocket state + age of the last pose frame |
| Risk-history charts | live risk scores recorded in the browser while the page is open |

## Not available from the backend yet (handled in the browser)
- Acknowledge (alerts) and Settings > Save are stored in this browser only; the backend has no endpoints for them.
- Database health is shown as "Not reported" (/health does not return it).

## Notes
- Event / risk field names are read tolerantly in src/data/model.js; adjust there if your pipeline uses other names.
- Verified with a mocked backend (idle, fall, no frames). Not yet run against your real backend.

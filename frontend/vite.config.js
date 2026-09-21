import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "0.0.0.0",
    port: 5175,
    strictPort: true,
    proxy: {
      "/health": "http://127.0.0.1:8001",
      "/risk": "http://127.0.0.1:8001",
      "/monitor": "http://127.0.0.1:8001",
      "/events": "http://127.0.0.1:8001",
      "/patients": "http://127.0.0.1:8001",
      "/rooms": "http://127.0.0.1:8001",
      "/predict": "http://127.0.0.1:8001",
      "/ws": { target: "ws://127.0.0.1:8001", ws: true },
    },
  },
  preview: {
    host: "0.0.0.0",
    port: 4173,
    strictPort: false,
  },
});

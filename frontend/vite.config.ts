import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const api = process.env.FEATHERLM_API ?? "http://127.0.0.1:8000";
const proxied = ["/v1", "/training", "/health", "/metrics", "/system", "/settings", "/benchmarks", "/models", "/teach"];

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { proxy: Object.fromEntries(proxied.map((p) => [p, { target: api, changeOrigin: false }])) },
  build: { outDir: "dist", chunkSizeWarningLimit: 1200 },
});

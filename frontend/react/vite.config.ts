import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The bundle is served by FastAPI from `frontend/dist` under the `/ui` mount,
// so assets must be requested with that prefix in production.
export default defineConfig({
  plugins: [react()],
  base: '/ui/',
  build: {
    outDir: '../dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      // Pipeline endpoints live at the backend root, not under /ui.
      '^/(start_pipeline|result|embed|ask_llm|login|signup|logout|users)': {
        target: process.env.BACKEND_URL ?? 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})

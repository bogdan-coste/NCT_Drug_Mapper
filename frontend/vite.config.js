import { fileURLToPath, URL } from 'node:url'
import vue from '@vitejs/plugin-vue'
import { defineConfig } from 'vite'

// Backend (FastAPI) origin used for the dev-server proxy.
const BACKEND = process.env.BACKEND_URL || 'http://localhost:8000'

export default defineConfig(({ command }) => ({
  plugins: [vue()],
  // Dev server serves at "/", the production bundle is served by FastAPI under "/ui/".
  base: command === 'build' ? '/ui/' : '/',
  resolve: {
    alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) },
  },
  build: {
    // Emitted to frontend/dist, which FastAPI serves.
    outDir: 'dist',
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      '/start_pipeline': BACKEND,
      '/start_dc_pipeline': BACKEND,
      '/result': BACKEND,
      '/embed': BACKEND,
      '/ask_llm': BACKEND,
      '/health': BACKEND,
    },
  },
}))

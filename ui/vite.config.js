import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    host: true,
    port: 5173,
    proxy: {
      // Go API
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      // Python agent
      '/py': {
        target: 'http://localhost:8001',
        changeOrigin: true,
        rewrite: (p) => p.replace(/^\/py/, ''), // so /py/agent/* -> /agent/* at FastAPI
      },
    },
  },
})

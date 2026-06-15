import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  // Built app is served by the FastAPI server mounted at /avatar (same-origin so
  // the nav tab works over Tailscale without a separate port). `npm run dev`
  // still serves at root on :5173.
  base: '/avatar/',
  plugins: [react()],
  server: {
    port: 5173,
  },
})

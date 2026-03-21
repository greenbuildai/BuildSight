import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/upload_video': 'http://localhost:8000',
      '/upload_video_raw': 'http://localhost:8000',
      '/upload_init': 'http://localhost:8000',
      '/upload_chunk': 'http://localhost:8000',
      '/upload_complete': 'http://localhost:8000',
      '/video_feed': 'http://localhost:8000',
      '/config': 'http://localhost:8000',
      '/uploads': 'http://localhost:8000',
      '/ws': {
        target: 'ws://localhost:8000',
        ws: true
      }
    }
  }
})

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    // Disable CSS minification: Vite 8's LightningCSS bundler has issues
    // with @keyframes declarations in certain CSS structures in this project.
    cssMinify: false,
  },
})



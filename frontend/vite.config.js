import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/api': {
        target: 'https://h847mxpg-5001.inc1.devtunnels.ms/',
        changeOrigin: true,
      },
      '/result': {
        target: 'https://h847mxpg-5001.inc1.devtunnels.ms/',
        changeOrigin: true,
      },
      '/uploads': {
        target: 'https://h847mxpg-5001.inc1.devtunnels.ms/',
        changeOrigin: true,
      }
    }
  }
})

import { defineConfig } from 'vite'

const backendTarget =
  (globalThis as unknown as { process?: { env?: Record<string, string | undefined> } }).process?.env
    ?.VITE_SOLARLAB_SERVER ?? 'http://127.0.0.1:8000'

const v2Routes = [
  '/schema',
  '/configuration-schema',
  '/configs',
  '/configuration-preview',
  '/materials',
  '/experiments',
  '/jobs',
  '/runs',
  '/results',
  '/artifacts',
] as const

export default defineConfig({
  server: {
    port: 5173,
    proxy: {
      ...Object.fromEntries(
        v2Routes.map((route) => [
          route,
          {
            target: backendTarget,
            changeOrigin: true,
          },
        ]),
      ),
      '/steady-service': {
        target: backendTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/steady-service/, ''),
      },
    },
  },
})

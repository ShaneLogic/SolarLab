/** Preview the existing build against an explicitly running research job service.
 * Usage: node scripts/ionic-jv-browser.mjs --service http://127.0.0.1:8000 --port 5173
 * Starts no numerical service or job; Ctrl-C closes only this preview server. */
import { preview } from 'vite'
import { parseArgs } from 'node:util'
import { fileURLToPath } from 'node:url'

const { values } = parseArgs({ options: { service: { type: 'string' }, port: { type: 'string', default: '5173' } } })
if (!values.service) throw Error('Pass --service with the explicit installed job-service URL')
const service = new URL(values.service), port = Number(values.port)
if (!['http:', 'https:'].includes(service.protocol) || service.username || service.password || service.search || service.hash
  || service.pathname !== '/' || !Number.isInteger(port) || port < 1 || port > 65535) throw Error('Invalid service URL or port')
const root = fileURLToPath(new URL('..', import.meta.url))
const server = await preview({ root, configFile: false, preview: { host: '127.0.0.1', port, strictPort: true,
  proxy: { '/steady-service': { target: service.origin, changeOrigin: true, rewrite: path => path.replace(/^\/steady-service/, '') } } } })
console.log(JSON.stringify({ browser_url: `http://127.0.0.1:${port}/?view=ionic-jv`, service: service.origin, preview_pid: process.pid, root }))
let closing = false
for (const signal of ['SIGINT', 'SIGTERM']) process.on(signal, () => {
  if (closing) return
  closing = true
  void server.close().then(() => {
    console.log(JSON.stringify({ preview_closed: true, preview_pid: process.pid }))
  }, error => { console.error(error); process.exitCode = 1 })
})

import { readFileSync } from 'node:fs'
import { gunzipSync } from 'node:zlib'
import { createHash } from 'node:crypto'
import { resolve } from 'node:path'

/** Exact captured HTTP bytes from the accepted original attempts; no numerical test work. */
export function savedSteady2DFixture() {
  const packed = readFileSync(resolve('src/test-helpers/steady-2d-records.json.gz'))
  const fixture = JSON.parse(gunzipSync(packed).toString('utf8'))
  for (const record of Object.values(fixture.responses)) {
    if (createHash('sha256').update(Buffer.from(record.base64, 'base64')).digest('hex') !== record.sha256) throw Error('Fixture content hash differs')
  }
  return fixture
}
export function fixtureFetch(fixture, methods = []) {
  return async (input, init = {}) => {
    methods.push(init.method ?? 'GET')
    if ((init.method ?? 'GET') !== 'GET') throw Error('Saved fixture never submits a numerical job')
    const url = new URL(input), path = url.pathname.replace(/^\/steady-service/, '') + url.search
    const record = fixture.responses[path]
    if (!record) throw Error('Missing captured GET ' + path)
    return new Response(Uint8Array.from(Buffer.from(record.base64, 'base64')), { status: record.status, headers: record.headers })
  }
}

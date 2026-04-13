import type { JobStartResponse, JobStreamHandlers, ProgressEvent } from './types'

export async function startJob(
  kind: 'jv' | 'impedance' | 'degradation',
  device: unknown,
  params: Record<string, unknown>,
  configPath: string | null = null,
): Promise<string> {
  const resp = await fetch('http://127.0.0.1:8000/api/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ kind, device, params, config_path: configPath }),
  })
  if (!resp.ok) {
    throw new Error(`POST /api/jobs failed: ${resp.status} ${resp.statusText}`)
  }
  const body = (await resp.json()) as JobStartResponse
  return body.job_id
}

export function streamJobEvents<TResult>(
  jobId: string,
  handlers: JobStreamHandlers<TResult>,
): () => void {
  const source = new EventSource(`http://127.0.0.1:8000/api/jobs/${jobId}/events`)
  source.addEventListener('progress', (e: MessageEvent) => {
    try {
      handlers.onProgress(JSON.parse(e.data) as ProgressEvent)
    } catch (err) {
      console.error('failed to parse progress event', err)
    }
  })
  source.addEventListener('result', (e: MessageEvent) => {
    try {
      handlers.onResult(JSON.parse(e.data) as TResult)
    } catch (err) {
      handlers.onError(`failed to parse result: ${String(err)}`)
    }
  })
  source.addEventListener('error', (e: MessageEvent) => {
    // Native EventSource 'error' fires on connection close with no data —
    // only surface as a user error when the server sent an SSE `event: error`
    // frame (which carries JSON).
    if (!e.data) return
    try {
      const payload = JSON.parse(e.data) as { message?: string }
      handlers.onError(payload.message ?? 'stream error')
    } catch {
      handlers.onError('stream error')
    }
  })
  source.addEventListener('done', () => {
    handlers.onDone()
    source.close()
  })
  return () => source.close()
}

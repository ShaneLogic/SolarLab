export interface SavedResponse { status: number; headers: Record<string, string>; base64: string; sha256: string }
export interface SavedFixture {
  schema: string
  targets: { storeId: string; runId: string; attemptId: string; accepted: number; state: 'succeeded' | 'failed' }[]
  responses: Record<string, SavedResponse>
}
export function savedIonicJVFixture(): SavedFixture
export function fixtureFetch(fixture: SavedFixture, methods?: string[]): typeof fetch

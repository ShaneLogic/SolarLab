/** Read-only stored-run preparation; no submission, cancellation or UI selection.
 * Contracts: solarlab_server/app.py, events.py and solarlab/io/run_store.py.
 */
export type StoredInteger = number | bigint;
export type StoredObject = Record<string, unknown>;
export type RunState = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';

export interface RunSnapshot extends StoredObject {
  run_id: string;
  current_attempt_id: string;
  state: RunState;
  generation: StoredInteger;
  request: StoredObject;
}

export interface ArtifactReference {
  artifact_id: string;
  manifest_sha256: string;
}

export interface TerminalRecord extends StoredObject {
  state: 'succeeded' | 'failed' | 'cancelled';
  result: StoredObject;
  artifacts: ArtifactReference[];
}

export interface AttemptSnapshot extends StoredObject {
  run_id: string;
  attempt_id: string;
  state: RunState;
  generation: StoredInteger;
  result: StoredObject | null;
  terminal: TerminalRecord | null;
}

export interface StoredEvent extends StoredObject {
  sequence: StoredInteger;
  run_id: string;
  attempt_id: string;
  kind: 'queued' | 'running' | 'progress' | 'terminal';
  payload: StoredObject;
}

export interface EventPage extends StoredObject {
  store_id: string;
  events: StoredEvent[];
  next_cursor: StoredInteger;
}

export interface ArtifactRecord extends StoredObject, ArtifactReference {
  run_id: string;
  attempt_id: string;
  sha256: string;
  size_bytes: StoredInteger;
  phase: 'committed';
  metadata: StoredObject;
}

/** Keep the original JSON for exact export, including integer/float spelling.
 * Unsafe JSON integer literals are bigint in value; opaque fields are not
 * reinterpreted as a scientific schema. No JSON.stringify round trip is needed.
 */
export interface StoredDocument<T> { value: T; json: string }
export interface AttemptIdentity { runId: string; attemptId: string }
export interface ReplayCursor extends AttemptIdentity {
  endpoint: string;
  storeId: string;
  sequence: string;
}
export interface StreamEnd {
  reason: 'terminal' | 'terminal_acknowledged' | 'disconnected' | 'unsubscribed';
  cursor: ReplayCursor;
  transportError?: unknown;
}
export interface RunSubscription {
  readonly cursor: ReplayCursor;
  readonly done: Promise<StreamEnd>;
  unsubscribe(): void;
}

type ErrorCode = 'http' | 'malformed' | 'identity_mismatch' | 'protocol'
  | 'integrity' | 'limit_exceeded' | 'verification_unavailable' | 'consumer';
export class RunClientError extends Error {
  readonly code: ErrorCode;
  readonly status?: number;
  readonly detail?: unknown;

  constructor(code: ErrorCode, message: string, options?: ErrorOptions & { status?: number; detail?: unknown }) {
    super(message, options);
    this.name = 'RunClientError';
    this.code = code;
    this.status = options?.status;
    this.detail = options?.detail;
  }
}

const MAX_SEQUENCE = 9223372036854775807n;
const states = ['queued', 'running', 'succeeded', 'failed', 'cancelled'];
const terminalStates = ['succeeded', 'failed', 'cancelled'];
const digestPattern = /^[a-f0-9]{64}$/;

function requireValue(condition: unknown, message: string, code: ErrorCode = 'malformed'): asserts condition {
  if (!condition) throw new RunClientError(code, message);
}

function object(value: unknown): asserts value is StoredObject {
  requireValue(value !== null && typeof value === 'object' && !Array.isArray(value), 'Expected a stored JSON object');
}

function identifier(value: unknown): asserts value is string {
  requireValue(typeof value === 'string' && /^[A-Za-z0-9_][A-Za-z0-9_.:-]{0,127}$/.test(value), 'Invalid stored identifier');
}

function integer(value: unknown, maximum?: bigint): bigint {
  requireValue(typeof value === 'bigint' || (typeof value === 'number' && Number.isSafeInteger(value)), 'Expected an exact integer');
  const result = BigInt(value);
  requireValue(result >= 0n && (maximum === undefined || result <= maximum), 'Integer outside the stored domain');
  return result;
}

function sequence(value: string): bigint {
  requireValue(typeof value === 'string' && /^[0-9]{1,19}$/.test(value), 'Invalid replay sequence');
  return integer(BigInt(value), MAX_SEQUENCE);
}

function same(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) return true;
  if (!left || !right || typeof left !== 'object' || typeof right !== 'object') return false;
  if (Array.isArray(left) !== Array.isArray(right)) return false;
  const a = left as StoredObject, b = right as StoredObject;
  const keys = Object.keys(a);
  return keys.length === Object.keys(b).length
    && keys.every(key => Object.hasOwn(b, key) && same(a[key], b[key]));
}

function parseJson(json: string): unknown {
  try {
    // Native source-aware revivers retain SQLite nanoseconds/64-bit cursors and
    // arbitrary opaque integer metadata. Older runtimes fail closed when exact
    // recovery is needed; they must not silently alias two persisted IDs.
    return JSON.parse(json, (_key: string, value: unknown, context?: { source?: string }) => {
      if (typeof value !== 'number') return value;
      if (context?.source && /^-?[0-9]+$/.test(context.source) && !Number.isSafeInteger(value)) {
        return BigInt(context.source);
      }
      requireValue(context?.source || Number.isSafeInteger(value) || !Number.isInteger(value),
        'This runtime cannot recover an exact JSON integer', 'verification_unavailable');
      requireValue(Number.isFinite(value), 'Non-finite JSON number');
      return value;
    });
  } catch (cause) {
    if (cause instanceof RunClientError) throw cause;
    throw new RunClientError('malformed', 'Invalid stored JSON', { cause });
  }
}

function reference(value: unknown): asserts value is ArtifactReference {
  object(value);
  identifier(value.artifact_id);
  requireValue(typeof value.manifest_sha256 === 'string' && digestPattern.test(value.manifest_sha256), 'Invalid artifact manifest digest');
}

function terminal(value: unknown): asserts value is TerminalRecord {
  object(value);
  requireValue(terminalStates.includes(String(value.state)), 'Invalid terminal state');
  object(value.result);
  requireValue(Array.isArray(value.artifacts), 'Missing terminal artifact references');
  value.artifacts.forEach(reference);
  requireValue(new Set(value.artifacts.map(item => item.artifact_id)).size === value.artifacts.length, 'Duplicate terminal artifact reference');
}

function event(value: unknown, runId: string, attemptId?: string): asserts value is StoredEvent {
  object(value);
  requireValue(value.run_id === runId && (attemptId === undefined || value.attempt_id === attemptId), 'Foreign run/attempt event', 'identity_mismatch');
  identifier(value.attempt_id);
  requireValue(integer(value.sequence, MAX_SEQUENCE) > 0n, 'Event sequence must be positive');
  integer(value.generation);
  integer(value.created_ns);
  requireValue(['queued', 'running', 'progress', 'terminal'].includes(String(value.kind)), 'Unknown persisted event kind', 'protocol');
  if (value.kind === 'progress') {
    requireValue(typeof value.event_key === 'string' && value.event_key.startsWith('user:'), 'Invalid progress event key');
    identifier(value.event_key.slice(5));
  } else requireValue(value.event_key === value.kind, 'Invalid lifecycle event key');
  object(value.payload);
  if (value.kind === 'terminal') terminal(value.payload);
}

async function bytes(response: Response, maximum: number): Promise<Uint8Array<ArrayBuffer>> {
  const reader = response.body?.getReader();
  requireValue(reader, 'Missing response body');
  const chunks: Uint8Array[] = [];
  let length = 0;
  try {
    while (true) {
      const item = await reader.read();
      if (item.done) break;
      length += item.value.byteLength;
      requireValue(length <= maximum, 'Response exceeds the client byte limit', 'limit_exceeded');
      chunks.push(item.value);
    }
  } finally {
    await reader.cancel().catch(() => undefined);
    reader.releaseLock();
  }
  const result = new Uint8Array(length);
  let offset = 0;
  for (const chunk of chunks) { result.set(chunk, offset); offset += chunk.byteLength; }
  return result;
}

function decode(data: Uint8Array): string {
  try { return new TextDecoder('utf-8', { fatal: true }).decode(data); }
  catch (cause) { throw new RunClientError('malformed', 'Invalid UTF-8 response', { cause }); }
}

/** A caller-selected reader, bound to one expected RunStore, never a default API.
 * maxResponseBytes bounds each JSON/artifact response and each SSE frame, not
 * a physical output or solver limit. Cross-origin deployment/CORS is separate.
 */
export class StoredRunClient {
  readonly endpoint: string;
  readonly storeId: string;
  readonly maxResponseBytes: number;

  constructor(endpoint: string | URL, storeId: string, maxResponseBytes = 16 * 1024 ** 2) {
    const url = new URL(endpoint);
    requireValue(['http:', 'https:'].includes(url.protocol) && !url.username && !url.password && !url.search && !url.hash, 'Expected an explicit HTTP(S) service endpoint');
    identifier(storeId);
    requireValue(Number.isSafeInteger(maxResponseBytes) && maxResponseBytes > 0, 'Invalid client transport byte limit');
    this.endpoint = url.href.replace(/\/$/, '') + '/';
    this.storeId = storeId;
    this.maxResponseBytes = maxResponseBytes;
  }

  private path(runId: string, attemptId?: string): string {
    identifier(runId);
    if (attemptId !== undefined) identifier(attemptId);
    return `runs/${encodeURIComponent(runId)}` + (attemptId === undefined ? '' : `/attempts/${encodeURIComponent(attemptId)}`);
  }

  private bind(response: Response): void {
    requireValue(response.headers.get('X-RunStore-ID') === this.storeId, 'Response is not bound to the requested store', 'identity_mismatch');
  }

  private async successful(response: Response): Promise<void> {
    if (response.ok) return;
    let detail: unknown;
    try { detail = parseJson(decode(await bytes(response, this.maxResponseBytes))); }
    catch { detail = undefined; }
    throw new RunClientError('http', `Stored run service: HTTP ${response.status}`, { status: response.status, detail });
  }

  private async document(path: string, signal?: AbortSignal): Promise<StoredDocument<StoredObject>> {
    const response = await fetch(new URL(path, this.endpoint), { signal, redirect: 'error', cache: 'no-store', headers: { Accept: 'application/json' } });
    try {
      await this.successful(response); this.bind(response);
      requireValue(response.status === 200 && response.headers.get('Content-Type')?.split(';')[0] === 'application/json', 'Expected a JSON record response', 'protocol');
    } catch (error) { await response.body?.cancel().catch(() => undefined); throw error; }
    const json = decode(await bytes(response, this.maxResponseBytes));
    const value = parseJson(json);
    object(value);
    return { value, json };
  }

  async getRun(runId: string, signal?: AbortSignal): Promise<StoredDocument<RunSnapshot>> {
    const result = await this.document(this.path(runId), signal);
    const value = result.value;
    requireValue(value.run_id === runId, 'Foreign run snapshot', 'identity_mismatch');
    identifier(value.current_attempt_id);
    requireValue(states.includes(String(value.state)), 'Invalid run state');
    integer(value.generation); integer(value.created_ns);
    object(value.request); object(value.request.identity); object(value.request.input_metadata);
    requireValue(value.request.identity_kind === 'opaque_upstream', 'Unknown execution identity representation');
    for (const key of ['H_input', 'H_physics', 'H_execution']) {
      requireValue(typeof value.request.identity[key] === 'string' && (value.request.identity[key] as string).trim(), 'Missing opaque upstream identity');
    }
    return result as StoredDocument<RunSnapshot>;
  }

  async getAttempt(target: AttemptIdentity, signal?: AbortSignal): Promise<StoredDocument<AttemptSnapshot>> {
    const { runId, attemptId } = target;
    const result = await this.document(this.path(runId, attemptId), signal);
    const value = result.value;
    requireValue(value.run_id === runId && value.attempt_id === attemptId, 'Foreign attempt snapshot', 'identity_mismatch');
    requireValue(states.includes(String(value.state)), 'Invalid attempt state');
    integer(value.generation); integer(value.created_ns);
    requireValue(integer(value.attempt_number) > 0n, 'Invalid attempt number');
    if (value.owner_id !== null) identifier(value.owner_id);
    if (value.started_ns !== null) integer(value.started_ns);
    if (terminalStates.includes(String(value.state))) {
      terminal(value.terminal); object(value.result); integer(value.finished_ns);
      requireValue(value.terminal.state === value.state && same(value.terminal.result, value.result), 'Contradictory terminal snapshot', 'protocol');
    } else {
      requireValue(value.terminal === null && value.result === null && value.finished_ns === null, 'Pending attempt contains a terminal result', 'protocol');
    }
    return result as StoredDocument<AttemptSnapshot>;
  }

  /** A run page may span attempts; it is not an attempt-stream replay cursor. */
  async getEvents(runId: string, after = '0', limit?: number, signal?: AbortSignal): Promise<StoredDocument<EventPage>> {
    let previous = sequence(after);
    if (limit !== undefined) requireValue(Number.isSafeInteger(limit) && limit >= 1 && limit <= 1000, 'Invalid event page limit');
    const query = new URLSearchParams({ after });
    if (limit !== undefined) query.set('limit', String(limit));
    const result = await this.document(`${this.path(runId)}/events?${query}`, signal);
    requireValue(result.value.store_id === this.storeId, 'Foreign event-page store', 'identity_mismatch');
    requireValue(Array.isArray(result.value.events), 'Missing event page');
    if (limit !== undefined) requireValue(result.value.events.length <= limit, 'Oversized event page', 'protocol');
    for (const item of result.value.events) {
      event(item, runId);
      const current = integer(item.sequence, MAX_SEQUENCE);
      requireValue(current > previous, 'Event page is not strictly increasing', 'protocol');
      previous = current; // Global SQLite IDs need not be contiguous for this run.
    }
    requireValue(integer(result.value.next_cursor, MAX_SEQUENCE) === previous, 'Incorrect next page cursor', 'protocol');
    return result as StoredDocument<EventPage>;
  }

  /** The service verifies its manifest; the client matches the committed reference
   * and independently checks the delivered content size and SHA-256. NPY/raw
   * bytes are returned intact, without a float-array or scientific conversion.
   */
  async getArtifact(target: AttemptIdentity, expected: ArtifactReference, signal?: AbortSignal): Promise<{ record: StoredDocument<ArtifactRecord>; bytes: Uint8Array<ArrayBuffer> }> {
    const { runId, attemptId } = target;
    this.path(runId, attemptId); reference(expected);
    const { artifact_id: artifactId, manifest_sha256: manifestSha256 } = expected;
    const path = `artifacts/${encodeURIComponent(artifactId)}`;
    const record = await this.document(path, signal);
    const value = record.value;
    requireValue(value.run_id === runId && value.attempt_id === attemptId && value.artifact_id === artifactId, 'Foreign artifact identity', 'identity_mismatch');
    reference(value);
    requireValue(value.manifest_sha256 === manifestSha256 && value.phase === 'committed', 'Artifact is not the committed reference', 'integrity');
    requireValue(typeof value.sha256 === 'string' && digestPattern.test(value.sha256), 'Invalid content digest');
    object(value.metadata);
    const size = integer(value.size_bytes);
    requireValue(size <= BigInt(this.maxResponseBytes), 'Artifact exceeds the client byte limit', 'limit_exceeded');
    const subtle = globalThis.crypto?.subtle;
    requireValue(subtle, 'Content hashing is unavailable', 'verification_unavailable');
    const response = await fetch(new URL(`${path}/content`, this.endpoint), { signal, redirect: 'error', cache: 'no-store' });
    try {
      await this.successful(response); this.bind(response);
      requireValue(response.status === 200 && response.headers.get('Content-Type')?.split(';')[0] === 'application/octet-stream', 'Expected an exact artifact byte response', 'protocol');
    } catch (error) { await response.body?.cancel().catch(() => undefined); throw error; }
    const data = await bytes(response, this.maxResponseBytes);
    let digest: ArrayBuffer;
    try { digest = await subtle.digest('SHA-256', data); }
    catch (cause) { throw new RunClientError('verification_unavailable', 'Content hashing failed', { cause }); }
    const hash = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
    requireValue(BigInt(data.byteLength) === size && hash === value.sha256 && response.headers.get('ETag') === `"${hash}"`, 'Artifact content integrity mismatch', 'integrity');
    return { record: record as StoredDocument<ArtifactRecord>, bytes: data };
  }

  /** One bounded-memory connection to one exact attempt. Reconnect explicitly by
   * calling subscribe again with the prior subscription.cursor. Only successful
   * onEvent completion advances that cursor; disconnect never cancels the run.
   * Native fetch exposes status/identity headers that EventSource hides.
   */
  subscribe(target: AttemptIdentity, onEvent: (record: StoredDocument<StoredEvent>) => void | Promise<void>, resume?: ReplayCursor): RunSubscription {
    const path = this.path(target.runId, target.attemptId);
    const identity = Object.freeze({ endpoint: this.endpoint, storeId: this.storeId, runId: target.runId, attemptId: target.attemptId });
    let cursor: ReplayCursor = Object.freeze({ ...identity, sequence: '0' });
    if (resume !== undefined) {
      requireValue(Object.entries(identity).every(([key, value]) => resume[key as keyof ReplayCursor] === value), 'Replay cursor belongs to another source/store/run/attempt', 'identity_mismatch');
      sequence(resume.sequence);
      cursor = Object.freeze({ ...identity, sequence: resume.sequence });
    }
    const abort = new AbortController();
    let unsubscribed = false;
    const end = (reason: StreamEnd['reason'], transportError?: unknown): StreamEnd => ({ reason, cursor, ...(transportError === undefined ? {} : { transportError }) });
    const done = (async (): Promise<StreamEnd> => {
      let response: Response;
      try {
        response = await fetch(new URL(`${path}/events?after=0`, this.endpoint), {
          signal: abort.signal, redirect: 'error', cache: 'no-store',
          headers: { Accept: 'text/event-stream', 'Last-Event-ID': cursor.sequence },
        });
      } catch (error) { return end(unsubscribed ? 'unsubscribed' : 'disconnected', error); }
      try {
        await this.successful(response); this.bind(response);
        if (response.status === 204) {
          requireValue(sequence(cursor.sequence) > 0n, 'Terminal acknowledgement without a replay cursor', 'protocol');
          return end(unsubscribed ? 'unsubscribed' : 'terminal_acknowledged');
        }
        requireValue(response.status === 200 && response.headers.get('Content-Type')?.split(';')[0] === 'text/event-stream', 'Expected an attempt event stream', 'protocol');
      } catch (error) { await response.body?.cancel().catch(() => undefined); throw error; }
      const reader = response.body?.getReader();
      requireValue(reader, 'Missing stream body', 'protocol');
      const decoder = new TextDecoder('utf-8', { fatal: true });
      let buffer = '', lastJson: string | undefined;
      let terminalSeen = false;
      try {
        while (!unsubscribed) {
          let item: ReadableStreamReadResult<Uint8Array>;
          try { item = await reader.read(); }
          catch (error) { return end(unsubscribed ? 'unsubscribed' : 'disconnected', error); }
          try { buffer += decoder.decode(item.value, { stream: !item.done }); }
          catch (cause) { throw new RunClientError('malformed', 'Invalid stream UTF-8', { cause }); }
          let boundary: number;
          while ((boundary = buffer.indexOf('\n\n')) !== -1 && !unsubscribed) {
            const block = buffer.slice(0, boundary);
            buffer = buffer.slice(boundary + 2);
            requireValue(new TextEncoder().encode(block).byteLength <= this.maxResponseBytes, 'Event frame exceeds the client byte limit', 'limit_exceeded');
            // Deliberately the server's LF/id/event/single-data-line contract,
            // not a general EventSource implementation or a legacy done shim.
            const fields = new Map<string, string>();
            for (const line of block.split('\n')) {
              if (!line || line.startsWith(':')) continue;
              const match = /^(id|event|data): (.*)$/.exec(line);
              requireValue(match && !fields.has(match[1]), 'Malformed persisted-event frame', 'protocol');
              fields.set(match[1], match[2]);
            }
            if (fields.size === 0) continue;
            requireValue(fields.size === 3, 'Incomplete persisted-event frame', 'protocol');
            const json = fields.get('data')!;
            const value = parseJson(json); event(value, identity.runId, identity.attemptId);
            const current = integer(value.sequence, MAX_SEQUENCE);
            requireValue(sequence(fields.get('id')!) === current && fields.get('event') === value.kind, 'SSE ID/kind differs from its stored record', 'protocol');
            if (current === sequence(cursor.sequence) && json === lastJson) continue;
            requireValue(!terminalSeen && current > sequence(cursor.sequence), 'Conflicting duplicate, regressing cursor or event after terminal', 'protocol');
            try { await onEvent({ value, json }); }
            catch (cause) { throw new RunClientError('consumer', 'Event consumer failed; cursor was not advanced', { cause }); }
            cursor = Object.freeze({ ...identity, sequence: current.toString() });
            lastJson = json;
            terminalSeen = value.kind === 'terminal';
          }
          requireValue(new TextEncoder().encode(buffer).byteLength <= this.maxResponseBytes, 'Event frame exceeds the client byte limit', 'limit_exceeded');
          if (item.done) {
            // An interrupted partial frame is never delivered or acknowledged.
            if (buffer.length !== 0) return end('disconnected');
            return end(unsubscribed ? 'unsubscribed' : terminalSeen ? 'terminal' : 'disconnected');
          }
        }
        return end('unsubscribed');
      } finally {
        await reader.cancel().catch(() => undefined);
        reader.releaseLock();
      }
    })();
    return { get cursor() { return cursor; }, done,
      unsubscribe() { unsubscribed = true; abort.abort(); } };
  }
}

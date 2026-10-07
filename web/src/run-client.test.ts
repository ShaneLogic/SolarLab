// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { StoredRunClient } from './run-client';
import type { ReplayCursor, RunSubscription, StoredEvent } from './run-client';
import { webcrypto } from './test-helpers/schema-fixture.mjs';

const endpoint = 'http://127.0.0.1:8123/stored/';
const storeId = 'store-a';
const target = { runId: 'run-a', attemptId: 'attempt-a' };
const client = () => new StoredRunClient(endpoint, storeId);
const manifest = 'b'.repeat(64);
const emptySha = 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855';
const artifactReference = { artifact_id: 'artifact-a', manifest_sha256: manifest };
const opaque = {
  identity_kind: 'opaque_upstream',
  identity: { H_input: 'input-source', H_physics: 'physical-source', H_execution: 'actual-execution', extra: null },
  input_metadata: { zero: 0, negative_zero: -0, null_value: null, enabled: false, qualification: 'unqualified', layers: [{ value: 0 }] },
};

function run(extra = {}) {
  return { run_id: target.runId, current_attempt_id: target.attemptId, state: 'running', generation: 1, created_ns: 0, request: opaque, ...extra };
}
function attempt(extra = {}) {
  return { run_id: target.runId, attempt_id: target.attemptId, attempt_number: 1,
    state: 'running', generation: 1, created_ns: 0, started_ns: 0, finished_ns: null,
    owner_id: 'fixture', result: null, terminal: null, ...extra };
}
function terminal(state = 'failed') {
  return { state, result: { reason: 'fixture-failure', tau: null, zero: 0, qualification: 'not_evaluated' }, artifacts: [] };
}
function event(sequence: number, kind = 'progress', extra = {}) {
  return { sequence, run_id: target.runId, attempt_id: target.attemptId, generation: kind === 'queued' ? 0 : 1,
    kind, event_key: kind === 'progress' ? `user:p${sequence}` : kind, created_ns: 0,
    payload: kind === 'terminal' ? terminal() : { zero: 0, missing: null }, ...extra };
}
function artifact(extra = {}) {
  return { ...artifactReference, run_id: target.runId, attempt_id: target.attemptId, generation: 1,
    phase: 'committed', sha256: emptySha, size_bytes: 0, format: 'bytes',
    name: 'fixture', relative_path: 'objects/fixture.bin', created_ns: 0,
    metadata: { null_value: null, zero: 0 }, ...extra };
}
function jsonResponse(value: unknown, headers = {}, status = 200) {
  return new Response(JSON.stringify(value), { status, headers: { 'Content-Type': 'application/json', 'X-RunStore-ID': storeId, ...headers } });
}
function frame(record = event(1)): string {
  return `id: ${record.sequence}\nevent: ${record.kind}\ndata: ${JSON.stringify(record)}\n\n`;
}
function stream(text: string, headers = {}) {
  return new Response(text, { headers: { 'Content-Type': 'text/event-stream; charset=utf-8', 'X-RunStore-ID': storeId, ...headers } });
}
function respond(...responses: Response[]) {
  const mock = vi.fn();
  for (const response of responses) mock.mockResolvedValueOnce(response);
  vi.stubGlobal('fetch', mock);
  return mock;
}
function cursor(sequence = '3'): ReplayCursor {
  return { endpoint, storeId, ...target, sequence };
}

beforeEach(() => { vi.stubGlobal('crypto', webcrypto); });
afterEach(() => { vi.unstubAllGlobals(); });

describe('explicit stored-run reads', () => {
  it('keeps opaque metadata, exact large integers, signed zero, null and omission', async () => {
    const raw = JSON.stringify(run())
      .replace('"created_ns":0', '"created_ns":18446744073709551617')
      .replace('"negative_zero":0', '"negative_zero":-0.0');
    const fetch = respond(new Response(raw, { headers: { 'Content-Type': 'application/json', 'X-RunStore-ID': storeId } }));
    const result = await client().getRun(target.runId);
    expect(result.json).toBe(raw);
    expect(result.value.created_ns).toBe(18446744073709551617n);
    expect(result.value.request).toEqual(opaque);
    const input = result.value.request.input_metadata as Record<string, unknown>;
    expect(Object.is(input.negative_zero, -0)).toBe(true);
    expect(input.null_value).toBeNull();
    expect(Object.hasOwn(input, 'omitted')).toBe(false);
    expect(fetch).toHaveBeenCalledWith(new URL('runs/run-a', endpoint), expect.objectContaining({ redirect: 'error', cache: 'no-store' }));
    expect(fetch.mock.calls[0][1].method).toBeUndefined();
  });

  it('keeps a pending snapshot distinct from a failed terminal and result', async () => {
    const final = terminal();
    respond(jsonResponse(attempt()), jsonResponse(attempt({ state: 'failed', result: final.result, terminal: final, finished_ns: 0 })));
    expect((await client().getAttempt(target)).value.result).toBeNull();
    const result = (await client().getAttempt(target)).value;
    expect(result.state).toBe('failed');
    expect(result.result).toEqual(final.result);
    expect(result.terminal).toEqual(final);
  });

  it.each([
    { terminal: undefined },
    { result: {} },
    { finished_ns: 1 },
    { state: 'unknown' },
    { owner_id: undefined },
  ])('rejects malformed pending records %j', async extra => {
    respond(jsonResponse(attempt(extra)));
    await expect(client().getAttempt(target)).rejects.toHaveProperty('code');
  });

  it('rejects contradictory terminal results without selecting either one', async () => {
    respond(jsonResponse(attempt({ state: 'failed', result: {}, terminal: terminal(), finished_ns: 0 })));
    await expect(client().getAttempt(target)).rejects.toMatchObject({ code: 'protocol' });
  });

  it.each(['store-b', ''])('rejects a foreign or absent response store %j', async header => {
    respond(jsonResponse(run(), { 'X-RunStore-ID': header }));
    await expect(client().getRun(target.runId)).rejects.toMatchObject({ code: 'identity_mismatch' });
  });

  it('rejects foreign snapshot IDs', async () => {
    respond(jsonResponse(run({ run_id: 'other' })), jsonResponse(attempt({ attempt_id: 'other' })));
    await expect(client().getRun(target.runId)).rejects.toMatchObject({ code: 'identity_mismatch' });
    await expect(client().getAttempt(target)).rejects.toMatchObject({ code: 'identity_mismatch' });
  });

  it('retains an HTTP failure detail without reclassifying it as a run outcome', async () => {
    const detail = { detail: { code: 'artifact_integrity', reason: null, count: 0 } };
    respond(jsonResponse(detail, {}, 409));
    await expect(client().getRun(target.runId)).rejects.toMatchObject({ code: 'http', status: 409, detail });
  });

  it.each(['{', 'null', '[]', '{"state":"running"}'])('rejects malformed JSON/record %s', async text => {
    respond(new Response(text, { headers: { 'Content-Type': 'application/json', 'X-RunStore-ID': storeId } }));
    await expect(client().getRun(target.runId)).rejects.toHaveProperty('code');
  });

  it('rejects invalid UTF-8 and a wrong content type', async () => {
    respond(new Response(new Uint8Array([255]), { headers: { 'Content-Type': 'application/json', 'X-RunStore-ID': storeId } }),
      jsonResponse(run(), { 'Content-Type': 'text/html' }));
    await expect(client().getRun(target.runId)).rejects.toMatchObject({ code: 'malformed' });
    await expect(client().getRun(target.runId)).rejects.toMatchObject({ code: 'protocol' });
  });

  it('refuses unsafe integer parsing on a runtime without source-aware revivers', async () => {
    respond(new Response(JSON.stringify(run()).replace('"created_ns":0', '"created_ns":9007199254740993'),
      { headers: { 'Content-Type': 'application/json', 'X-RunStore-ID': storeId } }));
    const nativeParse = JSON.parse;
    const spy = vi.spyOn(JSON, 'parse').mockImplementation((text, reviver) => nativeParse(text,
      reviver ? function (key, value) { return reviver.call(this, key, value); } : undefined));
    try { await expect(client().getRun(target.runId)).rejects.toMatchObject({ code: 'verification_unavailable' }); }
    finally { spy.mockRestore(); }
  });

  it('bounds received JSON bytes and rejects identifier/path injection', async () => {
    respond(jsonResponse(run()));
    await expect(new StoredRunClient(endpoint, storeId, 8).getRun(target.runId)).rejects.toMatchObject({ code: 'limit_exceeded' });
    await expect(client().getRun('../run')).rejects.toMatchObject({ code: 'malformed' });
    expect(() => new StoredRunClient('https://user:secret@host.invalid', storeId)).toThrow();
  });
});

describe('run event pages', () => {
  it('accepts gaps and multiple attempts while retaining the exact cursor and empty-page zero', async () => {
    const records = [event(1, 'terminal', { attempt_id: 'old-attempt' }), event(8, 'queued')];
    respond(jsonResponse({ store_id: storeId, events: records, next_cursor: 8 }),
      jsonResponse({ store_id: storeId, events: [], next_cursor: 0 }));
    expect((await client().getEvents(target.runId, '0', 2)).value.events).toEqual(records);
    expect((await client().getEvents(target.runId)).value.next_cursor).toBe(0);
  });

  it.each([
    { store_id: 'foreign', events: [], next_cursor: 0 },
    { store_id: storeId, events: [event(1, 'queued', { run_id: 'foreign' })], next_cursor: 1 },
    { store_id: storeId, events: [event(1), event(1)], next_cursor: 1 },
    { store_id: storeId, events: [event(3), event(2)], next_cursor: 2 },
    { store_id: storeId, events: [event(3)], next_cursor: 4 },
    { store_id: storeId, events: [], next_cursor: null },
  ])('reports foreign or malformed pages %j', async page => {
    respond(jsonResponse(page));
    await expect(client().getEvents(target.runId)).rejects.toHaveProperty('code');
  });
});

describe('attempt streaming and replay', () => {
  it.each(['failed', 'succeeded', 'cancelled'])('delivers one persisted %s terminal with its unaltered result', async state => {
    const records = [event(1, 'queued'), event(3, 'running'), event(7), event(9, 'terminal', { payload: terminal(state) })];
    respond(stream(': keepalive\n\n' + records.map(record => frame(record)).join('')));
    const received: StoredEvent[] = [];
    const subscription = client().subscribe(target, record => { received.push(record.value); });
    expect((await subscription.done).reason).toBe('terminal');
    expect(received).toEqual(records);
    expect(subscription.cursor).toEqual(cursor('9'));
  });

  it('reconnects with Last-Event-ID precedence and recognises terminal HTTP 204 without a synthetic result', async () => {
    const fetch = respond(stream(frame(event(4))), stream(frame(event(9, 'terminal'))),
      new Response(null, { status: 204, headers: { 'X-RunStore-ID': storeId } }));
    const handler = vi.fn();
    const first = client().subscribe(target, handler);
    expect((await first.done).reason).toBe('disconnected');
    const second = client().subscribe(target, handler, first.cursor);
    expect((await second.done).reason).toBe('terminal');
    const acknowledged = client().subscribe(target, handler, second.cursor);
    expect((await acknowledged.done).reason).toBe('terminal_acknowledged');
    expect(handler).toHaveBeenCalledTimes(2);
    expect(fetch.mock.calls[1][1].headers['Last-Event-ID']).toBe('4');
    expect(String(fetch.mock.calls[1][0])).toBe(endpoint + 'runs/run-a/attempts/attempt-a/events?after=0');
  });

  it.each(['endpoint', 'storeId', 'runId', 'attemptId'])('refuses a foreign %s replay cursor before opening transport', key => {
    const fetch = respond();
    expect(() => client().subscribe(target, vi.fn(), { ...cursor(), [key]: 'foreign' })).toThrow();
    expect(fetch).not.toHaveBeenCalled();
  });

  it('checks stream store, record run/attempt, SSE ID and kind independently', async () => {
    for (const response of [
      stream(frame(), { 'X-RunStore-ID': 'foreign' }),
      stream(frame(event(1, 'progress', { run_id: 'foreign' }))),
      stream(frame(event(1, 'progress', { attempt_id: 'foreign' }))),
      stream(frame().replace('id: 1', 'id: 2')),
      stream(frame().replace('event: progress', 'event: terminal')),
    ]) {
      respond(response);
      const handler = vi.fn();
      await expect(client().subscribe(target, handler).done).rejects.toHaveProperty('code');
      expect(handler).not.toHaveBeenCalled();
    }
  });

  it('does not duplicate an identical last frame and reports conflicting/backward replay', async () => {
    const one = frame(event(1));
    respond(stream(one + one + frame(event(6, 'terminal'))));
    const handler = vi.fn();
    await client().subscribe(target, handler).done;
    expect(handler).toHaveBeenCalledTimes(2);
    for (const text of [one + frame(event(1, 'progress', { payload: { changed: true } })), one + frame(event(3)) + one]) {
      respond(stream(text));
      await expect(client().subscribe(target, vi.fn()).done).rejects.toMatchObject({ code: 'protocol' });
    }
  });

  it('preserves IDs above Number.MAX_SAFE_INTEGER without aliasing adjacent events', async () => {
    const large = (value: string, kind = 'progress') => frame(event(1, kind))
      .replace('id: 1', `id: ${value}`).replace('"sequence":1', `"sequence":${value}`);
    respond(stream(large('9007199254740993') + large('9007199254740994', 'terminal')));
    const values: StoredEvent[] = [];
    const subscription = client().subscribe(target, record => { values.push(record.value); });
    await subscription.done;
    expect(values.map(value => value.sequence)).toEqual([9007199254740993n, 9007199254740994n]);
    expect(subscription.cursor.sequence).toBe('9007199254740994');
  });

  it('rejects extra post-terminal or legacy error/result/done events', async () => {
    for (const text of [frame(event(1, 'terminal')) + frame(event(2)),
      ...['error', 'result', 'done'].map(kind => frame(event(1, kind)))]) {
      respond(stream(text));
      await expect(client().subscribe(target, vi.fn()).done).rejects.toMatchObject({ code: 'protocol' });
    }
  });

  it('reports network errors as disconnects and HTTP/protocol errors as errors', async () => {
    const failure = new TypeError('offline');
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(failure));
    expect(await client().subscribe(target, vi.fn()).done).toMatchObject({ reason: 'disconnected', transportError: failure });
    respond(jsonResponse({ detail: 'unavailable' }, {}, 503),
      new Response(null, { status: 204, headers: { 'X-RunStore-ID': storeId } }));
    await expect(client().subscribe(target, vi.fn()).done).rejects.toMatchObject({ code: 'http', status: 503 });
    await expect(client().subscribe(target, vi.fn()).done).rejects.toMatchObject({ code: 'protocol' });
  });

  it('supports split frames and keeps an interrupted partial frame unacknowledged', async () => {
    const text = frame(event(5)) + frame(event(6)).slice(0, -1);
    const chunks = new ReadableStream<Uint8Array>({ start(controller) {
      for (const part of [text.slice(0, 7), text.slice(7, 27), text.slice(27)]) controller.enqueue(new TextEncoder().encode(part));
      controller.close();
    } });
    respond(new Response(chunks, { headers: { 'Content-Type': 'text/event-stream', 'X-RunStore-ID': storeId } }));
    const handler = vi.fn(), subscription = client().subscribe(target, handler);
    expect((await subscription.done).reason).toBe('disconnected');
    expect(subscription.cursor.sequence).toBe('5');
    expect(handler).toHaveBeenCalledTimes(1);
  });

  it('rejects completed malformed frames and oversized pending frames', async () => {
    respond(stream('id: 1\ndata: {}\n\n'), stream('x'.repeat(65)));
    await expect(client().subscribe(target, vi.fn()).done).rejects.toMatchObject({ code: 'protocol' });
    await expect(new StoredRunClient(endpoint, storeId, 64).subscribe(target, vi.fn()).done).rejects.toMatchObject({ code: 'limit_exceeded' });
  });

  it('unsubscribes only the transport, stops further delivery and permits a new subscription', async () => {
    const fetch = respond(stream(frame(event(1)) + frame(event(4, 'terminal'))), stream(frame(event(4, 'terminal'))));
    const seen: StoredEvent[] = [];
    let subscription: RunSubscription;
    subscription = client().subscribe(target, record => { seen.push(record.value); subscription.unsubscribe(); });
    expect((await subscription.done).reason).toBe('unsubscribed');
    expect(seen).toHaveLength(1);
    expect(fetch.mock.calls[0][1].signal.aborted).toBe(true);
    expect((await client().subscribe(target, vi.fn(), subscription.cursor).done).reason).toBe('terminal');
    expect(fetch.mock.calls.every(([, init]) => init.method === undefined)).toBe(true);
  });

  it('acknowledges only a successfully completed consumer callback', async () => {
    respond(stream(frame(event(1)) + frame(event(4))));
    const failure = new Error('consumer write failed');
    const subscription = client().subscribe(target, async record => { if (record.value.sequence === 4) throw failure; });
    await expect(subscription.done).rejects.toMatchObject({ code: 'consumer', cause: failure });
    expect(subscription.cursor.sequence).toBe('1');
  });
});

describe('validated artifact downloads', () => {
  function content(data: Uint8Array = new Uint8Array(), extra = {}) {
    return new Response(data as Uint8Array<ArrayBuffer>, { headers: { 'Content-Type': 'application/octet-stream', 'X-RunStore-ID': storeId, ETag: `"${emptySha}"`, ...extra } });
  }

  it('accepts an actual zero-byte artifact and preserves metadata rather than treating zero as missing', async () => {
    respond(jsonResponse(artifact()), content());
    const result = await client().getArtifact(target, artifactReference);
    expect(result.bytes.byteLength).toBe(0);
    expect(result.record.value.metadata).toEqual({ null_value: null, zero: 0 });
    expect(Object.hasOwn(result.record.value.metadata, 'missing')).toBe(false);
  });

  it('preserves uninterpreted byte words and verifies their actual digest', async () => {
    const data = new Uint8Array([0, 128, 255, 0, 1, 2, 3]);
    const sha = Array.from(new Uint8Array(await webcrypto.subtle.digest('SHA-256', data)), x => x.toString(16).padStart(2, '0')).join('');
    respond(jsonResponse(artifact({ size_bytes: data.length, sha256: sha })), content(data, { ETag: `"${sha}"` }));
    expect((await client().getArtifact(target, artifactReference)).bytes).toEqual(data);
  });

  it.each([
    { artifact_id: 'foreign' }, { run_id: 'foreign' }, { attempt_id: 'foreign' },
    { manifest_sha256: 'c'.repeat(64) }, { phase: 'published' },
  ])('refuses a foreign or uncommitted artifact %j before content fetch', async extra => {
    const fetch = respond(jsonResponse(artifact(extra)));
    await expect(client().getArtifact(target, artifactReference)).rejects.toHaveProperty('code');
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('detects byte corruption, wrong size and a lying ETag independently', async () => {
    for (const [record, body] of [
      [artifact(), content(new Uint8Array([1]))],
      [artifact({ size_bytes: 1 }), content()],
      [artifact(), content(undefined, { ETag: '"foreign"' })],
      [artifact({ sha256: 'c'.repeat(64) }), content()],
    ] as const) {
      respond(jsonResponse(record), body);
      await expect(client().getArtifact(target, artifactReference)).rejects.toMatchObject({ code: 'integrity' });
    }
  });

  it('checks the content response store independently of metadata and propagates integrity HTTP errors', async () => {
    respond(jsonResponse(artifact()), content(undefined, { 'X-RunStore-ID': 'other' }),
      jsonResponse({ detail: { code: 'artifact_integrity', message: 'missing' } }, {}, 409));
    await expect(client().getArtifact(target, artifactReference)).rejects.toMatchObject({ code: 'identity_mismatch' });
    await expect(client().getArtifact(target, artifactReference)).rejects.toMatchObject({ code: 'http', status: 409,
      detail: { detail: { code: 'artifact_integrity', message: 'missing' } } });
  });

  it('fails closed when hashing is unavailable and refuses declared oversized content', async () => {
    respond(jsonResponse(artifact()), jsonResponse(artifact({ size_bytes: 17 * 1024 ** 2 })));
    vi.stubGlobal('crypto', undefined);
    await expect(client().getArtifact(target, artifactReference)).rejects.toMatchObject({ code: 'verification_unavailable' });
    await expect(client().getArtifact(target, artifactReference)).rejects.toMatchObject({ code: 'limit_exceeded' });
  });
});

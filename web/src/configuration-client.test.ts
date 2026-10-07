// @vitest-environment node
import { afterEach, describe, expect, it, vi } from 'vitest';
import type { DeviceInput, TandemInput } from './generated/configuration-inputs';
import {
  ConfigurationPreviewError, previewDeviceConfiguration, previewTandemConfiguration,
} from './configuration-client';

const endpoint = 'http://127.0.0.1:9000/configuration-preview/device';
// Transport-only DTOs. Actual material configurations run through the real
// resolver and HTTP service in test_configuration_preview.py.
function device(): DeviceInput {
  return {
    schema_version: 'solarlab.device-preparation.v1', id: 'wire-device',
    source_format: 'canonical', layers: [], description: undefined,
    settings: { phi_left: -0, Phi: 0, work_function_left_eV: null },
    simulation_hints: null,
  };
}

function tandem(): TandemInput {
  return {
    schema_version: 'solarlab.tandem-preparation.v1', source_schema_version: 1,
    id: 'wire-tandem', device_type: 'tandem_2T_monolithic',
    top_cell_reference: 'provided-top', bottom_cell_reference: 'provided-bottom',
    top_cell: device(), bottom_cell: { ...device(), id: 'bottom' },
    junction_model: 'ideal_ohmic', light_direction: 'top_first', junction_stack: [],
  };
}

function envelope(input: DeviceInput | TandemInput = device(), kind: 'device' | 'tandem' = 'device') {
  return {
    schema: 'solarlab.configuration-preview.v1', kind,
    status: 'prepared_pending_dependencies', can_execute: false,
    input,
    resolved: {
      schema: `solarlab.resolved-${kind}-preparation.v1`, id: input.id,
      can_execute: false, capability_gaps: ['transport-fixture-only'],
    },
    identity: {
      scope: 'configuration_content', content_sha256: 'a'.repeat(64),
      default_catalog_sha256: 'b'.repeat(64), resource_library_sha256: 'c'.repeat(64),
    },
  };
}

function response(json = JSON.stringify(envelope()), status = 200, type = 'application/json') {
  return new Response(json, { status, headers: { 'Content-Type': type } });
}

function respond(json = JSON.stringify(envelope()), status = 200, type = 'application/json') {
  const fetch = vi.fn().mockResolvedValue(response(json, status, type));
  vi.stubGlobal('fetch', fetch);
  return fetch;
}

afterEach(() => { vi.unstubAllGlobals(); });

describe('prepared configuration transport', () => {
  it('posts generated input to the explicit endpoint with numeric signed zero, null and omission', async () => {
    const input = device();
    const fetch = respond();
    const signal = new AbortController().signal;
    await previewDeviceConfiguration(new URL(endpoint), input, signal);
    const [url, init] = fetch.mock.calls[0];
    expect(String(url)).toBe(endpoint);
    expect(init).toMatchObject({ method: 'POST', headers: { 'Content-Type': 'application/json' }, signal });
    expect(init.body).toContain('"phi_left":-0.0');
    const sent = JSON.parse(init.body);
    expect(Object.is(sent.settings.phi_left, -0)).toBe(true);
    expect(typeof sent.settings.phi_left).toBe('number');
    expect(sent.settings.Phi).toBe(0);
    expect(sent.settings.work_function_left_eV).toBeNull();
    expect(sent.simulation_hints).toBeNull();
    expect(sent).not.toHaveProperty('description');
    expect(Object.is(input.settings?.phi_left, -0)).toBe(true);
    expect(Object.hasOwn(input, 'description')).toBe(true);
  });

  it('submits the generated tandem graph without loading its reference strings', async () => {
    const input = tandem();
    const fetch = respond(JSON.stringify(envelope(input, 'tandem')));
    const result = await previewTandemConfiguration(endpoint.replace('/device', '/tandem'), input);
    const sent = JSON.parse(fetch.mock.calls[0][1].body);
    expect(sent.top_cell_reference).toBe('provided-top');
    expect(Object.is(sent.top_cell.settings.phi_left, -0)).toBe(true);
    expect(sent.bottom_cell.settings.work_function_left_eV).toBeNull();
    expect(result.value.kind).toBe('tandem');
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('snapshots input before waiting and isolates the deeply frozen result', async () => {
    const input = device();
    const originalId = input.id;
    let finish!: (value: Response) => void;
    const fetch = vi.fn((_endpoint: unknown, _init: RequestInit) => new Promise<Response>(resolve => { finish = resolve; }));
    vi.stubGlobal('fetch', fetch);
    const waiting = previewDeviceConfiguration(endpoint, input);
    input.id = 'edited-later';
    input.settings!.Phi = 123;
    finish(response());
    const result = await waiting;
    expect(JSON.parse(fetch.mock.calls[0][1]!.body as string).settings.Phi).toBe(0);
    expect(result.value.input.id).toBe(originalId);
    expect(Object.isFrozen(result)).toBe(true);
    expect(Object.isFrozen(result.value.input.settings)).toBe(true);
    expect(Object.isFrozen(result.value.resolved.capability_gaps)).toBe(true);
    expect(() => { (result.value.input as Record<string, unknown>).id = 'changed'; }).toThrow(TypeError);
    expect(input.id).toBe('edited-later');
  });

  it('retains original JSON tokens and opaque provenance without claiming execution identity', async () => {
    const raw = JSON.stringify(envelope()).replace('"phi_left":0', '"phi_left":-0.0')
      .replace('"capability_gaps":', '"original_integer":9007199254740993,"clear":null,"capability_gaps":');
    respond(raw);
    const result = await previewDeviceConfiguration(endpoint, device());
    expect(result.json).toBe(raw);
    expect(result.value.resolved.original_integer).toBe(9007199254740993n);
    expect(Object.is((result.value.input.settings as Record<string, unknown>).phi_left, -0)).toBe(true);
    expect(result.value.resolved.clear).toBeNull();
    expect(result.value.resolved).not.toHaveProperty('absent');
    expect(result.value.identity.scope).toBe('configuration_content');
    expect(result.value.identity).not.toHaveProperty('H_execution');
    expect(result.value.status).toBe('prepared_pending_dependencies');
    expect(result.value.can_execute).toBe(false);
  });

  it('preserves structured field errors, unresolved status and original error JSON', async () => {
    const raw = '{"detail":{"code":"configuration_validation","status":"unresolved","can_execute":false,"field_errors":[{"loc":["layers",0,"parameters","mu_n"],"input":null,"msg":"invalid"}]}}';
    respond(raw, 422);
    const error = await previewDeviceConfiguration(endpoint, device()).catch(e => e);
    expect(error).toBeInstanceOf(ConfigurationPreviewError);
    expect(error).toMatchObject({ code: 'http', status: 422, json: raw, data: JSON.parse(raw) });
    expect(Object.isFrozen(error.data.detail.field_errors[0].loc)).toBe(true);
  });

  it.each([503, 415, 413])('retains HTTP %i even when its body is not JSON', async status => {
    respond('upstream unavailable', status, 'text/plain');
    await expect(previewDeviceConfiguration(endpoint, device())).rejects.toMatchObject({
      code: 'http', status, json: 'upstream unavailable', data: undefined,
    });
  });

  it.each([
    ['schema', 'other'], ['kind', 'tandem'], ['status', 'succeeded'], ['can_execute', true],
    ['input.id', 'foreign'], ['input.schema_version', 'unrecognized'],
    ['resolved.id', 'foreign'], ['resolved.schema', 'unrecognized'], ['resolved.can_execute', true],
    ['identity.scope', 'H_execution'], ['identity.content_sha256', 'invalid'],
    ['identity.default_catalog_sha256', null], ['identity.resource_library_sha256', 'A'.repeat(64)],
  ] as Array<[string, unknown]>)('rejects mismatched response %s', async (path, replacement) => {
    const data = envelope() as unknown as Record<string, unknown>;
    const names = path.split('.');
    const parent = names.length === 2 ? data[names[0]] as Record<string, unknown> : data;
    parent[names.at(-1)!] = replacement;
    respond(JSON.stringify(data));
    await expect(previewDeviceConfiguration(endpoint, device())).rejects.toMatchObject({ code: 'protocol' });
  });

  it('rejects an unexpected success status or content type', async () => {
    respond(JSON.stringify(envelope()), 202);
    await expect(previewDeviceConfiguration(endpoint, device())).rejects.toMatchObject({ code: 'protocol' });
    respond(JSON.stringify(envelope()), 200, 'text/plain');
    await expect(previewDeviceConfiguration(endpoint, device())).rejects.toMatchObject({ code: 'protocol' });
  });

  it.each(['{', '{"x":NaN}', '{"x":1e999}'])('rejects malformed or nonfinite response %s', async raw => {
    respond(raw);
    await expect(previewDeviceConfiguration(endpoint, device())).rejects.toMatchObject({ code: 'malformed' });
  });

  it('rejects invalid UTF-8 instead of replacing bytes silently', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(new Uint8Array([255]))));
    await expect(previewDeviceConfiguration(endpoint, device())).rejects.toMatchObject({ code: 'malformed' });
  });

  it('fails clearly when this runtime cannot recover an unsafe JSON integer', async () => {
    const nativeParse = JSON.parse;
    vi.stubGlobal('JSON', { ...JSON, stringify: JSON.stringify, parse: (text: string, reviver: (key: string, value: unknown) => unknown) =>
      nativeParse(text, (key, value) => reviver(key, value)) });
    respond('{"unknown":9007199254740993}');
    await expect(previewDeviceConfiguration(endpoint, device())).rejects.toMatchObject({ code: 'precision_unavailable' });
  });

  it('propagates fetch and response-body transport failures, including abort', async () => {
    const failure = new TypeError('connection refused');
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(failure));
    await expect(previewDeviceConfiguration(endpoint, device())).rejects.toBe(failure);
    const aborted = new DOMException('aborted while receiving', 'AbortError');
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ arrayBuffer: () => Promise.reject(aborted) }));
    await expect(previewDeviceConfiguration(endpoint, device())).rejects.toBe(aborted);
  });

  it.each([NaN, Infinity, new Date(0), 1n, () => 1, [undefined], new Array(1)])(
    'rejects unrepresentable input before fetch', async value => {
      const fetch = respond();
      const input = { ...device(), unknown: value } as DeviceInput;
      await expect(previewDeviceConfiguration(endpoint, input)).rejects.toMatchObject({ code: 'input' });
      expect(fetch).not.toHaveBeenCalled();
    },
  );

  it('rejects cycles and symbol fields without starting transport', async () => {
    const fetch = respond();
    const input = device() as DeviceInput & { self?: unknown };
    input.self = input;
    await expect(previewDeviceConfiguration(endpoint, input)).rejects.toMatchObject({ code: 'input' });
    delete input.self;
    Object.defineProperty(input, Symbol('hidden'), { value: 0 });
    await expect(previewDeviceConfiguration(endpoint, input)).rejects.toMatchObject({ code: 'input' });
    expect(fetch).not.toHaveBeenCalled();
  });
});

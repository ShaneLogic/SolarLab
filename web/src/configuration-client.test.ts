// @vitest-environment node
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { DeviceInput, TandemInput, JVExperimentInput } from './generated/configuration-inputs';
import { configurationSchemaSha256 } from './generated/configuration-schema';
import {
  ConfigurationPreviewError, previewDeviceConfiguration, previewTandemConfiguration, spatialExperimentInputFromPreview,
  previewJVExperiment, jvExperimentInputFromPreview, previewLegacyWire,
} from './configuration-client';
import type { LegacyWirePreviewRequest } from './configuration-client';
import { webcrypto } from './test-helpers/schema-fixture.mjs';

const endpoint = 'http://127.0.0.1:9000/configuration-preview/device';

describe('exact experiment input reopen', () => {
  function saved(integer: string, digest = configurationSchemaSha256) {
    return `{"schema":"solarlab.configuration-preview.v1","kind":"experiment","can_execute":false,"identity":{"configuration_schema_sha256":"${digest}"},"input":{"schema_version":"solarlab.experiment-preparation.v1","id":"saved","device":{"id":"device","settings":{"phi_left":-0.0},"extension":{"integer":${integer},"word":"0.00000000000000000001 m","empty":null}},"experiment":{"kind":"jv_2d"}}}`;
  }
  it('recovers exactly representable large SCAPS numbers and preserves signed zero/words/null', () => {
    const json = saved('100000000000000000000');
    const result = spatialExperimentInputFromPreview(json);
    expect(Reflect.get(result.device, 'extension')).toStrictEqual({ integer: 1e20, word: '0.00000000000000000001 m', empty: null });
    expect(Object.is(result.device.settings!.phi_left, -0)).toBe(true);
    expect(json).toContain('100000000000000000000');
  });
  it('rejects a nonrepresentable integer with its path instead of truncating it', () => {
    expect(() => spatialExperimentInputFromPreview(saved('100000000000000000001')))
      .toThrow('not exactly representable at ["device","extension","integer"]');
  });
  it('keeps schema mismatch visible when reopening', () => {
    expect(() => spatialExperimentInputFromPreview(saved('0', '0'.repeat(64)))).toThrow('schema identity');
  });
});
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

describe('source-bound legacy YAML preview transport', () => {
  beforeEach(() => { vi.stubGlobal('crypto', webcrypto); });
  const legacyEndpoint = 'http://127.0.0.1:9000/configuration-preview/legacy-wire';
  function wireRequest(): LegacyWirePreviewRequest {
    return { schema_version: 'solarlab.legacy-wire-preview-request.v1', input: device(),
      source: { id: 'source.yaml', sha256: 'd'.repeat(64) }, references: {} };
  }
  async function wireEnvelope(request = wireRequest(), text = '\uFEFF# constructed UTF-8 transport fixture\r\nvalue: -0.0\r\n') {
    const bytes = new TextEncoder().encode(text);
    const sha256 = Array.from(new Uint8Array(await webcrypto.subtle.digest('SHA-256', bytes)),
      byte => byte.toString(16).padStart(2, '0')).join('');
    return {
      schema: 'solarlab.legacy-wire-preview.v1', kind: 'legacy-wire', status: 'prepared_data_only', can_execute: false,
      input: request.input,
      identity: { scope: 'legacy_wire_preparation', source: { ...request.source }, references: { ...request.references },
        default_catalog_sha256: 'b'.repeat(64), resource_library_sha256: 'c'.repeat(64),
        configuration_schema_sha256: configurationSchemaSha256 },
      report: { schema: 'solarlab.legacy-wire-preparation.v1', status: 'prepared_data_only', supported: true,
        can_execute: false, strict_roundtrip_passed: true, source: { ...request.source },
        reference_sources: { ...request.references }, emitted_references: {},
        emitted: { id: request.source.id, sha256, bytes: bytes.byteLength },
        default_catalog_sha256: 'b'.repeat(64), unsupported: [] as { path: string; code: string; message: string }[] },
      documents: [{ role: 'source', reference: null as string | null, source_id: request.source.id,
        sha256, size_bytes: bytes.byteLength, media_type: 'application/yaml', utf8: text }],
    };
  }

  it('sends canonical input and exact source bindings without normalizing -0, null or omission', async () => {
    const request = wireRequest();
    const envelope = await wireEnvelope(request);
    const fetch = respond(JSON.stringify(envelope));
    const abort = new AbortController();
    const result = await previewLegacyWire(legacyEndpoint, request, abort.signal);
    const [url, options] = fetch.mock.calls[0];
    expect(url).toBe(legacyEndpoint);
    expect(options.signal).toBe(abort.signal);
    expect(options.headers['X-Solarlab-Configuration-Schema']).toBe(configurationSchemaSha256);
    const sent = JSON.parse(options.body);
    expect(Object.is(sent.input.settings.phi_left, -0)).toBe(true);
    expect(sent.input.settings.work_function_left_eV).toBeNull();
    expect(sent.input).not.toHaveProperty('description');
    expect(sent.source).toEqual(request.source);
    expect(result.value.documents[0].utf8).toBe(envelope.documents[0].utf8);
    expect(Object.isFrozen(result.value.report)).toBe(true);
  });

  it('uses the submitted identity snapshot when the caller later changes its selection', async () => {
    const request = { ...wireRequest(), source: { ...wireRequest().source }, input: device() };
    const envelope = await wireEnvelope(request);
    let release!: (response: Response) => void;
    vi.stubGlobal('fetch', vi.fn(() => new Promise<Response>(resolve => { release = resolve; })));
    const pending = previewLegacyWire(legacyEndpoint, request);
    request.source.id = 'changed-after-send';
    request.input.id = 'changed-after-send';
    // The response represents the earlier submitted input, not the mutated object.
    envelope.input = { ...device() };
    release(response(JSON.stringify(envelope)));
    await expect(pending).resolves.toMatchObject({ value: { input: { id: 'wire-device' } } });
  });

  it.each(['utf8', 'sha256', 'size_bytes'])('rejects changed downloadable %s', async key => {
    const envelope = await wireEnvelope();
    const document = envelope.documents[0] as Record<string, unknown>;
    document[key] = key === 'utf8' ? 'different bytes' : key === 'sha256' ? 'f'.repeat(64) : 0;
    envelope.report.emitted = { id: document.source_id as string, sha256: document.sha256 as string, bytes: document.size_bytes as number };
    respond(JSON.stringify(envelope));
    await expect(previewLegacyWire(legacyEndpoint, wireRequest())).rejects.toMatchObject({ code: 'integrity' });
  });

  it.each(['source', 'schema', 'input', 'execution', 'missing', 'reference'])('rejects incorrect %s response binding', async problem => {
    const envelope = await wireEnvelope();
    if (problem === 'source') envelope.identity.source.sha256 = 'e'.repeat(64);
    else if (problem === 'schema') envelope.identity.configuration_schema_sha256 = 'e'.repeat(64);
    else if (problem === 'input') envelope.input.id = 'foreign';
    else if (problem === 'execution') envelope.can_execute = true;
    else if (problem === 'missing') envelope.documents = [];
    else { envelope.documents[0].role = 'reference'; envelope.documents[0].reference = 'foreign.yaml'; }
    respond(JSON.stringify(envelope));
    await expect(previewLegacyWire(legacyEndpoint, wireRequest())).rejects.toMatchObject({ code: 'protocol' });
  });

  it('retains unsupported field diagnostics and refuses any partial YAML', async () => {
    const envelope = await wireEnvelope(), original = envelope.documents;
    envelope.status = envelope.report.status = 'unsupported';
    envelope.report.supported = false;
    envelope.report.unsupported = [{ path: '$.materials', code: 'material_inheritance', message: 'Named material graph cannot be flattened' }];
    envelope.documents = [];
    respond(JSON.stringify(envelope));
    const result = await previewLegacyWire(legacyEndpoint, wireRequest());
    expect(result.value.report.unsupported[0].path).toBe('$.materials');
    expect(result.value.documents).toEqual([]);
    envelope.documents = original;
    respond(JSON.stringify(envelope));
    await expect(previewLegacyWire(legacyEndpoint, wireRequest())).rejects.toMatchObject({ code: 'protocol' });
  });

  it.each(['id', 'sha256', 'bytes', 'reference_sources'])('rejects self-consistent bytes with mismatched report %s', async key => {
    const envelope = await wireEnvelope();
    if (key === 'reference_sources') envelope.report.reference_sources = { unexpected: { id: 'foreign', sha256: 'a'.repeat(64) } };
    else (envelope.report.emitted as Record<string, unknown>)[key] = key === 'bytes' ? 0 : 'f'.repeat(64);
    respond(JSON.stringify(envelope));
    await expect(previewLegacyWire(legacyEndpoint, wireRequest())).rejects.toMatchObject({ code: 'protocol' });
  });

  it('does not accept a download when hashing is unavailable or fails', async () => {
    respond(JSON.stringify(await wireEnvelope()));
    vi.stubGlobal('crypto', undefined);
    await expect(previewLegacyWire(legacyEndpoint, wireRequest())).rejects.toMatchObject({ code: 'verification_unavailable' });
    const cause = new Error('digest failed');
    respond(JSON.stringify(await wireEnvelope()));
    vi.stubGlobal('crypto', { subtle: { digest: () => Promise.reject(cause) } });
    await expect(previewLegacyWire(legacyEndpoint, wireRequest())).rejects.toMatchObject({ code: 'verification_unavailable', cause });
  });

  it('retains a source-binding HTTP error without publishing a candidate', async () => {
    respond('{"detail":{"status":"unresolved","can_execute":false,"field_errors":[{"loc":["source","sha256"],"msg":"wrong source"}]}}', 409);
    await expect(previewLegacyWire(legacyEndpoint, wireRequest())).rejects.toMatchObject({
      code: 'http', status: 409, data: { detail: { field_errors: [{ loc: ['source', 'sha256'] }] } },
    });
  });
});


describe('discriminated J-V preparation transport', () => {
  function jv(input: JVExperimentInput) {
    return { ...envelope(), kind: 'experiment', input,
      resolved: { schema: 'solarlab.resolved-jv-experiment-preparation.v1', id: input.id, can_execute: false },
      identity: { ...envelope().identity, configuration_schema_sha256: configurationSchemaSha256 } };
  }
  it('validates both J-V branches and rejects a same-ID response for a different branch', async () => {
    const raw: JVExperimentInput = { schema_version: 'solarlab.experiment-preparation.v1', id: 'same', device: device(), experiment: { kind: 'jv', V_max: null, v_rate: '40 mV/s' } };
    const fetch = respond(JSON.stringify(jv(raw)));
    const result = await previewJVExperiment('/configuration-preview/experiment', raw);
    expect(result.value.resolved.schema).toBe('solarlab.resolved-jv-experiment-preparation.v1');
    expect(fetch.mock.calls[0][1].headers['X-Solarlab-Configuration-Schema']).toBe(configurationSchemaSha256);
    expect(fetch.mock.calls[0][1].body).toContain('"v_rate":"40 mV/s"');
    const dark: JVExperimentInput = { ...raw, experiment: { kind: 'dark_jv' } };
    respond(JSON.stringify(jv(dark)));
    await expect(previewJVExperiment('/configuration-preview/experiment', raw)).rejects.toThrow('branch does not match');
    respond(JSON.stringify(jv(dark)));
    await expect(previewJVExperiment('/configuration-preview/experiment', dark)).resolves.toMatchObject({ value: { input: { experiment: { kind: 'dark_jv' } } } });
  });
  it('reopens exact J-V input and gives a typed error for a missing or wrong branch', () => {
    const raw: JVExperimentInput = { schema_version: 'solarlab.experiment-preparation.v1', id: 'saved', device: device(), experiment: { kind: 'dark_jv', V_max: -0 } };
    const json = JSON.stringify(jv(raw)).replace('"V_max":0', '"V_max":-0.0');
    expect(Object.is(jvExperimentInputFromPreview(json).experiment.V_max, -0)).toBe(true);
    expect(() => spatialExperimentInputFromPreview(json)).toThrow('spatial experiment');
    const missing = jv(raw); Reflect.deleteProperty(missing.input, 'experiment');
    expect(() => jvExperimentInputFromPreview(JSON.stringify(missing))).toThrow(ConfigurationPreviewError);
  });
});

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

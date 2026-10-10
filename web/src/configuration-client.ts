/** Explicit prepared-configuration transport; no default selection or execution. */
import type { DeviceInput, TandemInput, SpatialExperimentInput, JVExperimentInput, SweepInput } from './generated/configuration-inputs';
import { configurationSchemaSha256 } from './generated/configuration-schema';

type Kind = 'device' | 'tandem' | 'experiment' | 'sweep';
type ErrorCode = 'input' | 'http' | 'malformed' | 'protocol' | 'precision_unavailable' | 'verification_unavailable' | 'integrity';

export class ConfigurationPreviewError extends Error {
  readonly code: ErrorCode;
  readonly status?: number;
  readonly data?: unknown;
  readonly json?: string;

  constructor(code: ErrorCode, message: string, options?: ErrorOptions & {
    status?: number; data?: unknown; json?: string;
  }) {
    super(message, options);
    this.name = 'ConfigurationPreviewError';
    this.code = code;
    this.status = options?.status;
    this.data = options?.data;
    this.json = options?.json;
  }
}

export interface ConfigurationPreview {
  readonly schema: 'solarlab.configuration-preview.v1';
  readonly kind: Kind;
  readonly status: 'prepared_pending_dependencies';
  readonly can_execute: false;
  /** Opaque prepared mappings: these are not generated execution/result DTOs.
   * Unsafe JSON integers are bigint; json retains every original wire token. */
  readonly input: Readonly<Record<string, unknown>>;
  readonly resolved: Readonly<Record<string, unknown>>;
  readonly identity: {
    readonly scope: 'configuration_content';
    readonly content_sha256: string;
    readonly default_catalog_sha256: string;
    readonly resource_library_sha256: string;
    readonly configuration_schema_sha256?: string;
  };
}

export interface ConfigurationPreviewDocument {
  readonly value: ConfigurationPreview;
  readonly json: string;
}

export interface LegacyWireSourceIdentity {
  readonly id: string;
  readonly sha256: string;
}

export interface LegacyWirePreviewRequest {
  readonly schema_version: 'solarlab.legacy-wire-preview-request.v1';
  readonly input: DeviceInput | TandemInput;
  readonly source: LegacyWireSourceIdentity;
  readonly references?: Readonly<Record<string, LegacyWireSourceIdentity>>;
}

export interface LegacyWireContent {
  readonly role: 'source' | 'reference';
  readonly reference: string | null;
  readonly source_id: string;
  readonly sha256: string;
  readonly size_bytes: number;
  readonly media_type: 'application/yaml';
  /** Exact UTF-8 text; no parse/reformat step occurs before download. */
  readonly utf8: string;
}

export interface LegacyWirePreviewDocument {
  readonly json: string;
  readonly value: {
    readonly schema: 'solarlab.legacy-wire-preview.v1';
    readonly kind: 'legacy-wire';
    readonly status: 'prepared_data_only' | 'unsupported';
    readonly can_execute: false;
    readonly input: Readonly<Record<string, unknown>>;
    readonly identity: Readonly<Record<string, unknown>>;
    readonly report: Readonly<Record<string, unknown>> & {
      readonly supported: boolean;
      readonly unsupported: readonly { readonly path: string; readonly code: string; readonly message: string }[];
    };
    readonly documents: readonly LegacyWireContent[];
  };
}

function requireValue(value: unknown, message: string, code: ErrorCode = 'protocol'): asserts value {
  if (!value) throw new ConfigurationPreviewError(code, message);
}

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** Snapshot the generated DTO before awaiting fetch. Numeric -0 stays numeric
 * -0.0 on the wire; optional undefined properties are omitted, never nulled. */
function encode(value: unknown, ancestors = new Set<object>()): string {
  if (value === null) return 'null';
  if (typeof value === 'string' || typeof value === 'boolean') return JSON.stringify(value);
  if (typeof value === 'number') {
    requireValue(Number.isFinite(value), 'Configuration numbers must be finite', 'input');
    return Object.is(value, -0) ? '-0.0' : JSON.stringify(value);
  }
  requireValue(value !== null && typeof value === 'object', 'Unsupported configuration input value', 'input');
  requireValue(!ancestors.has(value), 'Cyclic configuration input', 'input');
  requireValue(Array.isArray(value) || Object.getPrototypeOf(value) === Object.prototype
    || Object.getPrototypeOf(value) === null, 'Expected plain configuration DTO data', 'input');
  requireValue(Object.getOwnPropertySymbols(value).length === 0, 'Symbol fields are not JSON input', 'input');
  ancestors.add(value);
  try {
    if (Array.isArray(value)) {
      return '[' + Array.from(value, item => encode(item, ancestors)).join(',') + ']';
    }
    return '{' + Object.entries(value).filter(([, item]) => item !== undefined)
      .map(([key, item]) => JSON.stringify(key) + ':' + encode(item, ancestors)).join(',') + '}';
  } finally { ancestors.delete(value); }
}

function parse(json: string): unknown {
  try {
    return JSON.parse(json, (_key: string, value: unknown, context?: { source?: string }) => {
      if (typeof value !== 'number') return value;
      requireValue(Number.isFinite(value), 'Non-finite preview response number', 'malformed');
      if (context?.source && /^-?[0-9]+$/.test(context.source) && !Number.isSafeInteger(value)) {
        return BigInt(context.source);
      }
      requireValue(context?.source || Number.isSafeInteger(value) || !Number.isInteger(value),
        'This runtime cannot recover exact preview JSON integers', 'precision_unavailable');
      return value;
    });
  } catch (cause) {
    if (cause instanceof ConfigurationPreviewError) throw cause;
    throw new ConfigurationPreviewError('malformed', 'Preview is not valid JSON', { cause });
  }
}

function freeze(value: unknown): void {
  if (value && typeof value === 'object') {
    for (const child of Object.values(value)) freeze(child);
    Object.freeze(value);
  }
}

async function requestPreview(
  endpoint: string | URL, body: string, withSchema: boolean, signal?: AbortSignal,
): Promise<{ value: unknown; json: string }> {
  const response = await fetch(endpoint, {
    method: 'POST', headers: { 'Content-Type': 'application/json',
      ...(withSchema ? { 'X-Solarlab-Configuration-Schema': configurationSchemaSha256 } : {}) }, body, signal,
  });
  const bytes = await response.arrayBuffer();
  let json: string;
  try { json = new TextDecoder('utf-8', { fatal: true }).decode(bytes); }
  catch (cause) {
    throw new ConfigurationPreviewError('malformed', 'Preview is not valid UTF-8', { status: response.status, cause });
  }
  if (!response.ok) {
    let data: unknown;
    try { data = parse(json); } catch { data = undefined; }
    freeze(data);
    throw new ConfigurationPreviewError('http', `Configuration preview: HTTP ${response.status}`, {
      status: response.status, data, json,
    });
  }
  requireValue(response.status === 200
    && response.headers.get('Content-Type')?.split(';')[0].trim().toLowerCase() === 'application/json',
  'Expected a configuration preview JSON response');
  return { value: parse(json), json };
}

async function preview(
  endpoint: string | URL, kind: Kind, input: DeviceInput | TandemInput | SpatialExperimentInput | JVExperimentInput | SweepInput, signal?: AbortSignal,
): Promise<ConfigurationPreviewDocument> {
  const body = encode(input);
  const requestedId = input.id;
  const experimentKind = 'experiment' in input ? input.experiment.kind : undefined;
  const resolvedKind = kind === 'experiment' ? ['jv', 'dark_jv'].includes(experimentKind!) ? 'jv-experiment' : 'spatial-experiment' : kind;
  const { value, json } = await requestPreview(endpoint, body, ['experiment', 'sweep'].includes(kind), signal);
  requireValue(object(value) && value.schema === 'solarlab.configuration-preview.v1'
    && value.kind === kind && value.status === 'prepared_pending_dependencies' && value.can_execute === false,
  'Unexpected configuration preview kind or execution status');
  requireValue(object(value.input) && value.input.id === requestedId
    && value.input.schema_version === `solarlab.${kind}-preparation.v1`, 'Preview input identity mismatch');
  requireValue(object(value.resolved) && value.resolved.id === requestedId
    && value.resolved.schema === `solarlab.resolved-${resolvedKind}-preparation.v1`
    && value.resolved.can_execute === false, 'Unexpected resolved configuration identity or execution status');
  requireValue(object(value.identity) && value.identity.scope === 'configuration_content', 'Expected configuration content identity');
  for (const field of ['content_sha256', 'default_catalog_sha256', 'resource_library_sha256']) {
    requireValue(typeof value.identity[field] === 'string' && /^[a-f0-9]{64}$/.test(value.identity[field]),
      `Invalid preview ${field}`);
  }
  if (kind === 'experiment' || kind === 'sweep') requireValue(value.identity.configuration_schema_sha256 === configurationSchemaSha256,
    'Experiment preview schema identity does not match this client');
  if (kind === 'experiment') requireValue(object(value.input.experiment) && value.input.experiment.kind === experimentKind,
    'Experiment preview branch does not match the submitted input');
  freeze(value);
  return Object.freeze({ value: value as unknown as ConfigurationPreview, json });
}

/** Use the existing lossless JSON transport and verify every downloadable byte
 * snapshot. Source IDs refer only to documents supplied to the server context. */
export async function previewLegacyWire(
  endpoint: string | URL, request: LegacyWirePreviewRequest, signal?: AbortSignal,
): Promise<LegacyWirePreviewDocument> {
  const body = encode(request);
  const requestedId = request.input.id, requestedSchema = request.input.schema_version;
  const source = { ...request.source };
  const references = Object.fromEntries(Object.entries(request.references ?? {}).map(([name, binding]) => [name, { ...binding }]));
  const subcells = request.input.schema_version === 'solarlab.tandem-preparation.v1'
    ? { top_cell: request.input.top_cell.id, bottom_cell: request.input.bottom_cell.id } : {};
  const matches = (value: unknown, expected: LegacyWireSourceIdentity) => object(value)
    && value.id === expected.id && value.sha256 === expected.sha256;
  const referenceBindingsMatch = (value: unknown) => object(value)
    && Object.keys(value).length === Object.keys(references).length
    && Object.entries(references).every(([name, binding]) => matches(value[name], binding));
  const { value, json } = await requestPreview(endpoint, body, true, signal);
  requireValue(object(value) && value.schema === 'solarlab.legacy-wire-preview.v1' && value.kind === 'legacy-wire'
    && value.can_execute === false, 'Unexpected legacy export preview or execution status');
  requireValue(object(value.input) && value.input.id === requestedId && value.input.schema_version === requestedSchema,
    'Legacy preview input identity mismatch');
  for (const [side, id] of Object.entries(subcells)) requireValue(object(value.input[side]) && value.input[side].id === id,
    'Legacy preview subcell instance identity mismatch');
  requireValue(object(value.identity) && value.identity.scope === 'legacy_wire_preparation'
    && matches(value.identity.source, source) && referenceBindingsMatch(value.identity.references), 'Legacy preview source identity mismatch');
  for (const key of ['default_catalog_sha256', 'resource_library_sha256']) requireValue(
    typeof value.identity[key] === 'string' && /^[a-f0-9]{64}$/.test(value.identity[key]), `Invalid legacy preview ${key}`);
  requireValue(value.identity.configuration_schema_sha256 === configurationSchemaSha256, 'Legacy preview schema identity mismatch');
  requireValue(object(value.report) && value.report.schema === 'solarlab.legacy-wire-preparation.v1'
    && value.report.can_execute === false && typeof value.report.supported === 'boolean'
    && value.report.status === value.status && matches(value.report.source, source)
    && referenceBindingsMatch(value.report.reference_sources)
    && value.report.default_catalog_sha256 === value.identity.default_catalog_sha256
    && Array.isArray(value.report.unsupported) && Array.isArray(value.documents), 'Invalid legacy dry-run report');
  for (const issue of value.report.unsupported) requireValue(object(issue) && typeof issue.path === 'string'
    && typeof issue.code === 'string' && typeof issue.message === 'string', 'Invalid unsupported field diagnostic');
  if (!value.report.supported) {
    requireValue(value.status === 'unsupported' && value.documents.length === 0 && value.report.unsupported.length > 0,
      'A rejected legacy edit must not publish partial documents');
  } else {
    requireValue(value.status === 'prepared_data_only' && value.report.strict_roundtrip_passed === true
      && value.report.unsupported.length === 0 && value.documents.length === 1 + Object.keys(references).length,
    'Legacy export is not a complete lossless preparation');
    const emittedReferences = value.report.emitted_references;
    requireValue(object(value.report.emitted) && object(emittedReferences)
      && Object.keys(emittedReferences).length === Object.keys(references).length,
    'Missing emitted document provenance');
    const subtle = globalThis.crypto?.subtle;
    requireValue(subtle, 'Content hashing is unavailable', 'verification_unavailable');
    const seen = new Set<string>();
    for (const document of value.documents) {
      requireValue(object(document) && (document.role === 'source' || document.role === 'reference')
        && typeof document.utf8 === 'string' && document.media_type === 'application/yaml'
        && typeof document.sha256 === 'string' && /^[a-f0-9]{64}$/.test(document.sha256)
        && Number.isSafeInteger(document.size_bytes) && Number(document.size_bytes) >= 0, 'Invalid legacy document');
      const main = document.role === 'source';
      const reference = document.reference;
      requireValue(main ? reference === null : typeof reference === 'string' && Object.hasOwn(references, reference),
        'Unexpected legacy document reference');
      const binding = main ? source : references[reference as string];
      const key = main ? 'source' : `reference:${reference}`;
      requireValue(!seen.has(key) && document.source_id === binding.id, 'Duplicate or foreign legacy document');
      seen.add(key);
      const emitted = main ? value.report.emitted : emittedReferences[reference as string];
      requireValue(object(emitted) && emitted.id === document.source_id && emitted.sha256 === document.sha256
        && emitted.bytes === document.size_bytes, 'Download disagrees with the migration report provenance');
      const bytes = new TextEncoder().encode(document.utf8);
      let digest: ArrayBuffer;
      try { digest = await subtle.digest('SHA-256', bytes); }
      catch (cause) { throw new ConfigurationPreviewError('verification_unavailable', 'Content hashing failed', { cause }); }
      const hash = Array.from(new Uint8Array(digest), byte => byte.toString(16).padStart(2, '0')).join('');
      requireValue(bytes.byteLength === document.size_bytes && hash === document.sha256, 'Legacy YAML content integrity mismatch', 'integrity');
      signal?.throwIfAborted();
    }
    requireValue(seen.has('source'), 'Missing main legacy document');
  }
  signal?.throwIfAborted();
  freeze(value);
  return Object.freeze({ value: value as unknown as LegacyWirePreviewDocument['value'], json });
}

export function previewDeviceConfiguration(
  endpoint: string | URL, input: DeviceInput, signal?: AbortSignal,
): Promise<ConfigurationPreviewDocument> {
  return preview(endpoint, 'device', input, signal);
}

export function previewTandemConfiguration(
  endpoint: string | URL, input: TandemInput, signal?: AbortSignal,
): Promise<ConfigurationPreviewDocument> {
  return preview(endpoint, 'tandem', input, signal);
}

export function previewSpatialExperiment(
  endpoint: string | URL, input: SpatialExperimentInput, signal?: AbortSignal,
): Promise<ConfigurationPreviewDocument> {
  return preview(endpoint, 'experiment', input, signal);
}

export function previewExperimentConfiguration(
  endpoint: string | URL, input: SpatialExperimentInput | JVExperimentInput, signal?: AbortSignal,
): Promise<ConfigurationPreviewDocument> {
  return preview(endpoint, 'experiment', input, signal);
}

export function previewJVExperiment(
  endpoint: string | URL, input: JVExperimentInput, signal?: AbortSignal,
): Promise<ConfigurationPreviewDocument> {
  return previewExperimentConfiguration(endpoint, input, signal);
}

export function previewSweepConfiguration(endpoint: string | URL, input: SweepInput, signal?: AbortSignal): Promise<ConfigurationPreviewDocument> {
  return preview(endpoint, 'sweep', input, signal);
}

function editable(value: unknown, path: (string | number)[]): unknown {
  if (typeof value === 'bigint') {
    const number = Number(value);
    requireValue(Number.isFinite(number) && BigInt(number) === value,
      `Saved input integer is not exactly representable at ${JSON.stringify(path)}`, 'precision_unavailable');
    return number;
  }
  if (Array.isArray(value)) return value.map((item, index) => editable(item, [...path, index]));
  if (object(value)) return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, editable(item, [...path, key])]));
  return value;
}

/** Reopen editable input from the original response text. JSON integers that
 * originated as exactly representable numbers can exceed MAX_SAFE_INTEGER.
 * Recover only those values; never round other integers into a generated DTO. */
function experimentInputFromPreview(json: string): SpatialExperimentInput | JVExperimentInput {
  const doc = parse(json);
  requireValue(object(doc) && doc.schema === 'solarlab.configuration-preview.v1'
    && doc.kind === 'experiment' && doc.can_execute === false && object(doc.input)
    && doc.input.schema_version === 'solarlab.experiment-preparation.v1', 'Expected an experiment preparation preview');
  requireValue(object(doc.identity) && doc.identity.configuration_schema_sha256 === configurationSchemaSha256,
    'Saved experiment schema identity does not match this client');
  requireValue(object(doc.input.experiment) && typeof doc.input.experiment.kind === 'string',
    'Saved preview is missing its experiment branch');
  const input = editable(doc.input, []);
  encode(input); // Check representability without using this text as an export.
  return input as SpatialExperimentInput | JVExperimentInput;
}

export function spatialExperimentInputFromPreview(json: string): SpatialExperimentInput {
  const input = experimentInputFromPreview(json);
  requireValue(['jv_2d', 'voc_grain_sweep'].includes(input.experiment.kind), 'Expected a spatial experiment input');
  return input as SpatialExperimentInput;
}

export function jvExperimentInputFromPreview(json: string): JVExperimentInput {
  const input = experimentInputFromPreview(json);
  requireValue(['jv', 'dark_jv'].includes(input.experiment.kind), 'Expected a one-dimensional J-V experiment input');
  return input as JVExperimentInput;
}

function savedSweep(json: string): Record<string, unknown> {
  const doc = parse(json);
  requireValue(object(doc) && doc.schema === 'solarlab.configuration-preview.v1' && doc.kind === 'sweep'
    && doc.can_execute === false && object(doc.input) && doc.input.schema_version === 'solarlab.sweep-preparation.v1', 'Expected a sweep preparation preview');
  requireValue(object(doc.identity) && doc.identity.configuration_schema_sha256 === configurationSchemaSha256, 'Saved sweep schema identity does not match this client');
  return doc;
}

export function sweepInputFromPreview(json: string): SweepInput {
  const input = editable(savedSweep(json).input, []);
  encode(input);
  return input as SweepInput;
}

/** Materialize only the selected returned input; this is not run admission. */
export function sweepPointInputFromPreview(json: string, pointId: string): DeviceInput | JVExperimentInput {
  const doc = savedSweep(json);
  requireValue(object(doc.resolved) && Array.isArray(doc.resolved.points), 'Saved sweep has no point declarations');
  const points = doc.resolved.points.filter(value => object(value) && value.id === pointId);
  requireValue(points.length === 1 && object(points[0]) && points[0].can_execute === false
    && object(points[0].applied_input), 'Select one declared non-executing point with an applied input');
  const input = editable(points[0].applied_input, []);
  requireValue(object(input) && ['solarlab.device-preparation.v1', 'solarlab.experiment-preparation.v1'].includes(String(input.schema_version)), 'Unknown sweep point input kind');
  encode(input);
  return input as unknown as DeviceInput | JVExperimentInput;
}

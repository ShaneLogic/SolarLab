/** Explicit prepared-configuration transport; no default selection or execution. */
import type { DeviceInput, TandemInput, SpatialExperimentInput } from './generated/configuration-inputs';
import { configurationSchemaSha256 } from './generated/configuration-schema';

type Kind = 'device' | 'tandem' | 'experiment';
type ErrorCode = 'input' | 'http' | 'malformed' | 'protocol' | 'precision_unavailable';

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

async function preview(
  endpoint: string | URL, kind: Kind, input: DeviceInput | TandemInput | SpatialExperimentInput, signal?: AbortSignal,
): Promise<ConfigurationPreviewDocument> {
  const body = encode(input);
  const requestedId = input.id;
  const response = await fetch(endpoint, {
    method: 'POST', headers: { 'Content-Type': 'application/json',
      ...(kind === 'experiment' ? { 'X-Solarlab-Configuration-Schema': configurationSchemaSha256 } : {}) }, body, signal,
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
  const value = parse(json);
  requireValue(object(value) && value.schema === 'solarlab.configuration-preview.v1'
    && value.kind === kind && value.status === 'prepared_pending_dependencies' && value.can_execute === false,
  'Unexpected configuration preview kind or execution status');
  requireValue(object(value.input) && value.input.id === requestedId
    && value.input.schema_version === `solarlab.${kind}-preparation.v1`, 'Preview input identity mismatch');
  requireValue(object(value.resolved) && value.resolved.id === requestedId
    && value.resolved.schema === `solarlab.resolved-${kind === 'experiment' ? 'spatial-experiment' : kind}-preparation.v1`
    && value.resolved.can_execute === false, 'Unexpected resolved configuration identity or execution status');
  requireValue(object(value.identity) && value.identity.scope === 'configuration_content', 'Expected configuration content identity');
  for (const field of ['content_sha256', 'default_catalog_sha256', 'resource_library_sha256']) {
    requireValue(typeof value.identity[field] === 'string' && /^[a-f0-9]{64}$/.test(value.identity[field]),
      `Invalid preview ${field}`);
  }
  if (kind === 'experiment') requireValue(value.identity.configuration_schema_sha256 === configurationSchemaSha256,
    'Experiment preview schema identity does not match this client');
  freeze(value);
  return Object.freeze({ value: value as unknown as ConfigurationPreview, json });
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

/** Reopen editable input from the original response text. JSON integers that
 * originated as exactly representable numbers can exceed MAX_SAFE_INTEGER.
 * Recover only those values; never round other integers into a generated DTO. */
export function spatialExperimentInputFromPreview(json: string): SpatialExperimentInput {
  const doc = parse(json);
  requireValue(object(doc) && doc.schema === 'solarlab.configuration-preview.v1'
    && doc.kind === 'experiment' && doc.can_execute === false && object(doc.input)
    && doc.input.schema_version === 'solarlab.experiment-preparation.v1', 'Expected an experiment preparation preview');
  requireValue(object(doc.identity) && doc.identity.configuration_schema_sha256 === configurationSchemaSha256,
    'Saved experiment schema identity does not match this client');
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
  const input = editable(doc.input, []);
  encode(input); // Check representability without using this text as an export.
  return input as SpatialExperimentInput;
}

/** Prepared metadata identity only; backend semantic validators remain authoritative. */
import { configurationSchemaSha256 } from './generated/configuration-schema';
import type { ConfigurationSchemaMetadata } from './generated/configuration-schema';

type SchemaErrorCode = 'http' | 'malformed' | 'verification_unavailable' | 'identity_mismatch';

export class ConfigurationSchemaError extends Error {
  readonly code: SchemaErrorCode;

  constructor(code: SchemaErrorCode, message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = 'ConfigurationSchemaError';
    this.code = code;
  }
}

/** The caller selects the endpoint; this helper does not activate an existing UI/API. */
export async function fetchConfigurationSchema(
  endpoint: string | URL,
): Promise<ConfigurationSchemaMetadata> {
  const response = await fetch(endpoint);
  if (!response.ok) {
    throw new ConfigurationSchemaError('http', `Configuration schema: HTTP ${response.status}`);
  }
  const subtle = globalThis.crypto?.subtle;
  if (!subtle || typeof subtle.digest !== 'function') {
    throw new ConfigurationSchemaError('verification_unavailable', 'Schema identity verification is unavailable');
  }
  const bytes = await response.arrayBuffer();
  let metadata: unknown;
  try {
    metadata = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes));
  } catch (cause) {
    throw new ConfigurationSchemaError('malformed', 'Configuration schema is not valid UTF-8 JSON', { cause });
  }
  let digest: ArrayBuffer;
  try {
    digest = await subtle.digest('SHA-256', bytes);
  } catch (cause) {
    throw new ConfigurationSchemaError('verification_unavailable', 'Schema identity verification failed', { cause });
  }
  const received = Array.from(new Uint8Array(digest), value => value.toString(16).padStart(2, '0')).join('');
  if (received !== configurationSchemaSha256) {
    throw new ConfigurationSchemaError('identity_mismatch', 'Configuration schema identity does not match this client');
  }
  // Acceptance is bound to the received bytes, never a response header or reserialized object.
  return metadata as ConfigurationSchemaMetadata;
}

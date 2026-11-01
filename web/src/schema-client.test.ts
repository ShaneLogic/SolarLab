// @vitest-environment node
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { configurationSchema, configurationSchemaSha256 } from './generated/configuration-schema';
import { fetchConfigurationSchema } from './schema-client';
import { configurationSchemaBytes, webcrypto } from './test-helpers/schema-fixture.mjs';

const endpoint = 'https://schema.invalid/configuration-schema';
let authoritative: ArrayBuffer;

beforeAll(() => {
  authoritative = configurationSchemaBytes();
});

beforeEach(() => { vi.stubGlobal('crypto', webcrypto); });
afterEach(() => { vi.unstubAllGlobals(); });

function respond(body: BodyInit, status = 200, etag: string = configurationSchemaSha256): void {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(body, {
    status, headers: { 'Content-Type': 'application/json', ETag: `"${etag}"` },
  })));
}

describe('explicit configuration schema transport', () => {
  it('accepts authoritative bytes using WebCrypto, independent of an advertised digest', async () => {
    respond(authoritative, 200, 'untrusted-header');
    const metadata = await fetchConfigurationSchema(endpoint);
    expect(metadata).toEqual(configurationSchema);
    expect(metadata).not.toBe(configurationSchema);
    expect(fetch).toHaveBeenCalledWith(endpoint);
  });

  it('rejects reformatting even when the JSON data and advertised identity match', async () => {
    const reformatted = JSON.stringify(JSON.parse(new TextDecoder().decode(authoritative)), null, 2);
    expect(JSON.parse(reformatted)).toEqual(configurationSchema);
    respond(reformatted);
    await expect(fetchConfigurationSchema(endpoint)).rejects.toMatchObject({ code: 'identity_mismatch' });
  });

  it('rejects a stale payload despite the expected digest header', async () => {
    respond(JSON.stringify({ ...configurationSchema, schema_version: 'stale-fixture' }));
    await expect(fetchConfigurationSchema(endpoint)).rejects.toMatchObject({ code: 'identity_mismatch' });
  });

  it.each([
    ['truncated JSON', '{'],
    ['invalid UTF-8', new Uint8Array([0xff]).buffer],
  ] as const)('reports malformed %s', async (_label, body) => {
    respond(body);
    await expect(fetchConfigurationSchema(endpoint)).rejects.toMatchObject({ code: 'malformed' });
  });

  it('reports HTTP errors before interpreting their body as metadata', async () => {
    respond('temporarily unavailable', 503);
    await expect(fetchConfigurationSchema(endpoint)).rejects.toMatchObject({
      code: 'http', message: 'Configuration schema: HTTP 503',
    });
  });

  it('fails explicitly when verification is unavailable', async () => {
    respond(authoritative);
    vi.stubGlobal('crypto', undefined);
    await expect(fetchConfigurationSchema(endpoint)).rejects.toMatchObject({
      name: 'ConfigurationSchemaError', code: 'verification_unavailable',
    });
  });

  it('preserves a WebCrypto failure without accepting metadata', async () => {
    respond(authoritative);
    const cause = new Error('verification disabled');
    vi.stubGlobal('crypto', { subtle: { digest: vi.fn().mockRejectedValue(cause) } });
    await expect(fetchConfigurationSchema(endpoint)).rejects.toMatchObject({
      code: 'verification_unavailable', cause,
    });
  });

  it('propagates a fetch failure without returning metadata', async () => {
    const failure = new TypeError('connection failed');
    vi.stubGlobal('fetch', vi.fn().mockRejectedValue(failure));
    await expect(fetchConfigurationSchema(endpoint)).rejects.toBe(failure);
  });
});

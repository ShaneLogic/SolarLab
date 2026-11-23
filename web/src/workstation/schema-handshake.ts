/**
 * Workstation startup schema & execution-catalog digest handshake (P06-07).
 *
 * Verifies that generated frontend DTO metadata (`configurationSchemaSha256`)
 * and backend `/configuration-schema` + `/experiments` (`X-Solarlab-Registry`,
 * `X-Solarlab-Job-Schema`, `ETag`) agree on exact SHA-256 content digests.
 * Any digest or contract mismatch surfaces an explicit `[role="alert"]`
 * diagnostic banner and refuses silent field dropping or coercion.
 */
import {
  configurationSchema,
  configurationSchemaSha256,
  type ConfigurationSchemaMetadata,
} from '../generated/configuration-schema'
import { ConfigurationSchemaError, fetchConfigurationSchema } from '../schema-client'

const SHA256_HEX = /^[a-f0-9]{64}$/

export interface VerifiedCatalogIdentity {
  readonly registrySha256: string
  readonly jobSchemaSha256: string
  readonly modelRegistrySha256: string
  readonly bindingsCount: number
  readonly bindingIds: readonly string[]
}

export type WorkstationHandshakeResult =
  | {
      readonly status: 'verified'
      readonly configurationSchemaSha256: string
      readonly metadata: ConfigurationSchemaMetadata
      readonly catalog: VerifiedCatalogIdentity | null
    }
  | {
      readonly status: 'mismatch'
      readonly code: 'identity_mismatch' | 'malformed' | 'catalog_mismatch'
      readonly message: string
      readonly expectedConfigurationSchemaSha256: string
      readonly receivedRegistrySha256?: string
      readonly receivedJobSchemaSha256?: string
    }
  | {
      readonly status: 'error'
      readonly code: 'verification_unavailable' | 'http'
      readonly message: string
      readonly expectedConfigurationSchemaSha256: string
    }
  | {
      readonly status: 'unavailable'
      readonly reason: string
      readonly expectedConfigurationSchemaSha256: string
    }

export interface WorkstationHandshakeOptions {
  readonly schemaEndpoint?: string | URL
  readonly experimentsEndpoint?: string | URL
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

async function sha256Hex(bytes: ArrayBuffer): Promise<string | null> {
  const subtle = globalThis.crypto?.subtle
  if (!subtle || typeof subtle.digest !== 'function') return null
  const digest = await subtle.digest('SHA-256', bytes)
  return Array.from(new Uint8Array(digest), b => b.toString(16).padStart(2, '0')).join('')
}

async function probeCatalogIdentity(
  endpoint: string | URL,
): Promise<
  | { kind: 'ok'; catalog: VerifiedCatalogIdentity }
  | { kind: 'unavailable' }
  | {
      kind: 'mismatch'
      message: string
      receivedRegistrySha256?: string
      receivedJobSchemaSha256?: string
    }
  | { kind: 'error'; message: string }
> {
  let response: Response
  try {
    response = await fetch(endpoint)
  } catch {
    return { kind: 'unavailable' }
  }

  if (response.status === 404 || response.status === 503) {
    return { kind: 'unavailable' }
  }
  if (response.status === 409) {
    return {
      kind: 'mismatch',
      message: 'Backend reported HTTP 409 schema_registry_mismatch on /experiments.',
    }
  }
  if (!response.ok) {
    return {
      kind: 'error',
      message: `Execution catalog probe failed with HTTP ${response.status}.`,
    }
  }

  const registryHeader = response.headers.get('X-Solarlab-Registry') ?? ''
  const jobSchemaHeader = response.headers.get('X-Solarlab-Job-Schema') ?? ''
  const etagHeader = response.headers.get('ETag') ?? response.headers.get('etag') ?? ''

  if (!SHA256_HEX.test(registryHeader) || !SHA256_HEX.test(jobSchemaHeader)) {
    return {
      kind: 'mismatch',
      message:
        'Execution catalog response is missing valid SHA-256 X-Solarlab-Registry or X-Solarlab-Job-Schema headers.',
      receivedRegistrySha256: registryHeader || undefined,
      receivedJobSchemaSha256: jobSchemaHeader || undefined,
    }
  }

  const bytes = await response.arrayBuffer()
  const bodyDigest = await sha256Hex(bytes)
  if (bodyDigest === null) {
    return {
      kind: 'error',
      message: 'WebCrypto SHA-256 digest is unavailable for execution catalog verification.',
    }
  }
  if (bodyDigest !== registryHeader) {
    return {
      kind: 'mismatch',
      message: `Execution catalog payload digest (${bodyDigest.slice(0, 12)}…) does not match X-Solarlab-Registry (${registryHeader.slice(0, 12)}…).`,
      receivedRegistrySha256: registryHeader,
      receivedJobSchemaSha256: jobSchemaHeader,
    }
  }
  if (etagHeader && etagHeader !== `"${registryHeader}"`) {
    return {
      kind: 'mismatch',
      message: `Execution catalog ETag (${etagHeader}) disagrees with X-Solarlab-Registry ("${registryHeader.slice(0, 12)}…").`,
      receivedRegistrySha256: registryHeader,
      receivedJobSchemaSha256: jobSchemaHeader,
    }
  }

  let parsed: unknown
  try {
    parsed = JSON.parse(new TextDecoder('utf-8', { fatal: true }).decode(bytes))
  } catch {
    return {
      kind: 'mismatch',
      message: 'Execution catalog payload is not valid UTF-8 JSON.',
      receivedRegistrySha256: registryHeader,
      receivedJobSchemaSha256: jobSchemaHeader,
    }
  }

  if (
    !isRecord(parsed) ||
    parsed.schema !== 'solarlab.execution-catalog.v1' ||
    parsed.job_schema_sha256 !== jobSchemaHeader ||
    !SHA256_HEX.test(String(parsed.model_registry_sha256 ?? '')) ||
    !Array.isArray(parsed.bindings)
  ) {
    return {
      kind: 'mismatch',
      message:
        'Execution catalog body disagrees with X-Solarlab-Job-Schema or lacks required solarlab.execution-catalog.v1 fields.',
      receivedRegistrySha256: registryHeader,
      receivedJobSchemaSha256: jobSchemaHeader,
    }
  }

  const bindingIds = parsed.bindings
    .filter(isRecord)
    .map(b => String(b.experiment_id ?? ''))
    .filter(Boolean)

  return {
    kind: 'ok',
    catalog: Object.freeze({
      registrySha256: registryHeader,
      jobSchemaSha256: jobSchemaHeader,
      modelRegistrySha256: String(parsed.model_registry_sha256),
      bindingsCount: parsed.bindings.length,
      bindingIds: Object.freeze(bindingIds),
    }),
  }
}

export async function verifyWorkstationHandshake(
  options: WorkstationHandshakeOptions = {},
): Promise<WorkstationHandshakeResult> {
  if (typeof globalThis.fetch !== 'function') {
    return {
      status: 'unavailable',
      reason: 'fetch is not available in this environment',
      expectedConfigurationSchemaSha256: configurationSchemaSha256,
    }
  }

  const schemaEndpoint = options.schemaEndpoint ?? '/configuration-schema'
  const experimentsEndpoint = options.experimentsEndpoint ?? '/experiments'

  let metadata: ConfigurationSchemaMetadata | null = null
  let schemaUnavailableReason: string | null = null

  try {
    metadata = await fetchConfigurationSchema(schemaEndpoint)
    if (metadata.schema_version !== configurationSchema.schema_version) {
      return {
        status: 'mismatch',
        code: 'identity_mismatch',
        message: `Configuration schema version "${String(metadata.schema_version)}" does not match client "${configurationSchema.schema_version}".`,
        expectedConfigurationSchemaSha256: configurationSchemaSha256,
      }
    }
  } catch (err) {
    if (err instanceof ConfigurationSchemaError) {
      if (err.code === 'identity_mismatch' || err.code === 'malformed') {
        return {
          status: 'mismatch',
          code: err.code,
          message: err.message,
          expectedConfigurationSchemaSha256: configurationSchemaSha256,
        }
      }
      if (err.code === 'verification_unavailable') {
        return {
          status: 'error',
          code: 'verification_unavailable',
          message: err.message,
          expectedConfigurationSchemaSha256: configurationSchemaSha256,
        }
      }
      if (err.code === 'http') {
        if (/\bHTTP (404|503)\b/.test(err.message)) {
          schemaUnavailableReason = err.message
        } else if (/\bHTTP 409\b/.test(err.message)) {
          return {
            status: 'mismatch',
            code: 'identity_mismatch',
            message: err.message,
            expectedConfigurationSchemaSha256: configurationSchemaSha256,
          }
        } else {
          return {
            status: 'error',
            code: 'http',
            message: err.message,
            expectedConfigurationSchemaSha256: configurationSchemaSha256,
          }
        }
      }
    } else {
      schemaUnavailableReason = err instanceof Error ? err.message : String(err)
    }
  }

  const catalogResult = await probeCatalogIdentity(experimentsEndpoint)
  if (catalogResult.kind === 'mismatch') {
    return {
      status: 'mismatch',
      code: 'catalog_mismatch',
      message: catalogResult.message,
      expectedConfigurationSchemaSha256: configurationSchemaSha256,
      receivedRegistrySha256: catalogResult.receivedRegistrySha256,
      receivedJobSchemaSha256: catalogResult.receivedJobSchemaSha256,
    }
  }
  if (catalogResult.kind === 'error') {
    return {
      status: 'error',
      code: 'http',
      message: catalogResult.message,
      expectedConfigurationSchemaSha256: configurationSchemaSha256,
    }
  }

  if (metadata === null) {
    return {
      status: 'unavailable',
      reason: schemaUnavailableReason ?? 'Configuration schema endpoint is not mounted',
      expectedConfigurationSchemaSha256: configurationSchemaSha256,
    }
  }

  return {
    status: 'verified',
    configurationSchemaSha256,
    metadata,
    catalog: catalogResult.kind === 'ok' ? catalogResult.catalog : null,
  }
}

export function renderHandshakeBanner(
  host: HTMLElement,
  result: WorkstationHandshakeResult,
): void {
  while (host.firstChild) host.removeChild(host.firstChild)
  host.dataset.handshakeStatus = result.status

  if (result.status === 'verified') {
    host.hidden = true
    host.dataset.schemaSha256 = result.configurationSchemaSha256
    host.dataset.registrySha256 = result.catalog?.registrySha256 ?? ''
    host.dataset.jobSchemaSha256 = result.catalog?.jobSchemaSha256 ?? ''
    return
  }

  if (result.status === 'unavailable') {
    host.hidden = true
    host.dataset.schemaSha256 = result.expectedConfigurationSchemaSha256
    return
  }

  host.hidden = false
  const banner = document.createElement('div')
  banner.className = 'jv2d-warning schema-handshake-alert'
  banner.setAttribute('role', 'alert')
  banner.setAttribute('data-test', 'schema-handshake-alert')
  banner.dataset.handshakeCode = result.code
  const prefix =
    result.status === 'mismatch'
      ? 'Schema / registry version mismatch'
      : 'Schema identity verification error'
  banner.textContent =
    `${prefix} (client DTO sha256:${result.expectedConfigurationSchemaSha256.slice(0, 12)}…): ` +
    `${result.message} Silent field dropping or coercion is refused; reload or rebuild client DTOs to match the backend.`
  host.appendChild(banner)
}

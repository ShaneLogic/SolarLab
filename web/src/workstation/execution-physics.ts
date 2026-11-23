/**
 * Authoritative post-run physics & qualification extractor (P06-08).
 *
 * Post-run badges and result headers read ONLY what the backend actually
 * executed (`execution.worker_result`, `execution_manifest`, `effective_physics`,
 * `active_physics`, and `scientific_qualification`) — never frontend `TIER_FLAGS`
 * or `deviceSnapshot.device.mode` guesses.
 */

export interface ExecutionPhysicsReceipt {
  readonly source: 'execution_manifest' | 'worker_result' | 'backend_receipt' | 'unavailable'
  readonly engine?: string
  readonly driver?: string
  readonly manifestSha256?: string
  readonly executionSha256?: string
  readonly scientificQualification: boolean | null
  readonly effectiveModels?: Readonly<Record<string, string>>
  readonly badgeLabel: string
  readonly summary: string
}

export interface QualificationBadgeInfo {
  readonly qualified: boolean
  readonly explicitUnqualified: boolean
  readonly label: string
  readonly detail: string
}

const ENVELOPE_KEYS = new Set([
  'active_physics',
  'execution',
  'execution_manifest',
  'effective_physics',
  'scientific_qualification',
  'artifacts',
  'run_id',
  'attempt_id',
])

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}

function asNonEmptyString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : undefined
}

function parseEffectiveModels(raw: unknown): Readonly<Record<string, string>> | undefined {
  if (Array.isArray(raw)) {
    const entries = raw
      .map((item, idx) => {
        if (typeof item === 'string' && item.trim()) return [`model_${idx + 1}`, item.trim()] as const
        if (isRecord(item) && typeof item.family === 'string' && typeof item.model_id === 'string') {
          return [item.family, item.model_id] as const
        }
        return null
      })
      .filter((entry): entry is readonly [string, string] => entry !== null)
    return entries.length > 0 ? Object.freeze(Object.fromEntries(entries)) : undefined
  }
  if (isRecord(raw)) {
    const entries: Array<[string, string]> = []
    for (const [key, val] of Object.entries(raw)) {
      if (val === null || val === undefined) continue
      if (typeof val === 'string' || typeof val === 'number' || typeof val === 'boolean') {
        entries.push([key, String(val)])
      } else if (Array.isArray(val)) {
        entries.push([key, val.map(String).join(', ')])
      } else if (isRecord(val) && typeof val.model_id === 'string') {
        entries.push([key, val.model_id])
      }
    }
    return entries.length > 0 ? Object.freeze(Object.fromEntries(entries)) : undefined
  }
  return undefined
}

export function extractExecutionPhysics<T extends object>(
  rawResult: T,
): { receipt: ExecutionPhysicsReceipt; pure: T } {
  const record: Record<string, unknown> = isRecord(rawResult) ? rawResult : {}
  const pureCopy: Record<string, unknown> = {}
  for (const [key, value] of Object.entries(record)) {
    if (!ENVELOPE_KEYS.has(key)) {
      pureCopy[key] = value
    }
  }

  const executionObj = isRecord(record.execution) ? record.execution : undefined
  const workerResult = isRecord(executionObj?.worker_result)
    ? executionObj.worker_result
    : executionObj
  const manifestObj = isRecord(record.execution_manifest)
    ? record.execution_manifest
    : isRecord(workerResult?.execution_manifest)
      ? workerResult.execution_manifest
      : undefined

  const effectiveRaw =
    record.effective_physics ??
    manifestObj?.effective_physics ??
    manifestObj?.selected_models ??
    workerResult?.effective_physics
  const effectiveModels = parseEffectiveModels(effectiveRaw)
  const effectiveText = typeof effectiveRaw === 'string' ? asNonEmptyString(effectiveRaw) : undefined

  const engine =
    asNonEmptyString(manifestObj?.engine) ??
    asNonEmptyString(workerResult?.engine)
  const driver =
    asNonEmptyString(manifestObj?.driver) ??
    asNonEmptyString(workerResult?.driver)
  const manifestSha256 =
    asNonEmptyString(manifestObj?.manifest_sha256) ??
    asNonEmptyString(workerResult?.manifest_sha256)
  const executionSha256 =
    asNonEmptyString(manifestObj?.execution_sha256) ??
    asNonEmptyString(manifestObj?.H_execution) ??
    asNonEmptyString(workerResult?.H_execution) ??
    asNonEmptyString(workerResult?.execution_sha256)

  const rawQual =
    typeof record.scientific_qualification === 'boolean'
      ? record.scientific_qualification
      : typeof manifestObj?.scientific_qualification === 'boolean'
        ? manifestObj.scientific_qualification
        : typeof workerResult?.scientific_qualification === 'boolean'
          ? workerResult.scientific_qualification
          : engine === 'legacy'
            ? false
            : null

  const activePhysicsText =
    asNonEmptyString(record.active_physics) ??
    asNonEmptyString(workerResult?.active_physics) ??
    effectiveText

  const engineDriverLabel = [engine?.toUpperCase(), driver].filter(Boolean).join(' · ')
  const digestSuffix = manifestSha256
    ? ` · manifest:${manifestSha256.slice(0, 12)}…`
    : executionSha256
      ? ` · exec:${executionSha256.slice(0, 12)}…`
      : ''

  if (manifestObj || effectiveModels) {
    const modelsText = effectiveModels
      ? Object.entries(effectiveModels)
          .map(([k, v]) => `${k}=${v}`)
          .join(', ')
      : activePhysicsText ?? 'declared execution manifest'
    const prefix = engineDriverLabel ? `Execution manifest (${engineDriverLabel})` : 'Execution manifest'
    return {
      pure: pureCopy as T,
      receipt: Object.freeze({
        source: 'execution_manifest',
        engine,
        driver,
        manifestSha256,
        executionSha256,
        scientificQualification: rawQual,
        effectiveModels,
        badgeLabel: engineDriverLabel || 'MANIFEST',
        summary: `${prefix}: ${modelsText}${digestSuffix}`,
      }),
    }
  }

  if (workerResult && (engine || driver || manifestSha256 || executionSha256 || rawQual !== null)) {
    const baseText = activePhysicsText ?? `Executed via ${engineDriverLabel || 'worker'}`
    return {
      pure: pureCopy as T,
      receipt: Object.freeze({
        source: 'worker_result',
        engine,
        driver,
        manifestSha256,
        executionSha256,
        scientificQualification: rawQual,
        badgeLabel: engineDriverLabel || 'WORKER',
        summary: `${baseText}${digestSuffix}`,
      }),
    }
  }

  if (activePhysicsText) {
    return {
      pure: pureCopy as T,
      receipt: Object.freeze({
        source: 'backend_receipt',
        scientificQualification: rawQual,
        badgeLabel: 'BACKEND RECEIPT',
        summary: activePhysicsText,
      }),
    }
  }

  return {
    pure: pureCopy as T,
    receipt: Object.freeze({
      source: 'unavailable',
      scientificQualification: rawQual,
      badgeLabel: 'UNVERIFIED',
      summary: 'Execution manifest not recorded on run payload',
    }),
  }
}

export function resolveRunExecutionPhysics(run: {
  readonly activePhysics: string
  readonly executionPhysics?: ExecutionPhysicsReceipt
  readonly scientificQualification?: boolean | null
}): ExecutionPhysicsReceipt {
  if (run.executionPhysics) return run.executionPhysics
  const qual =
    typeof run.scientificQualification === 'boolean' ? run.scientificQualification : null
  const text = asNonEmptyString(run.activePhysics)
  if (text && text !== 'unknown') {
    return Object.freeze({
      source: 'backend_receipt',
      scientificQualification: qual,
      badgeLabel: 'BACKEND RECEIPT',
      summary: text,
    })
  }
  return Object.freeze({
    source: 'unavailable',
    scientificQualification: qual,
    badgeLabel: 'UNVERIFIED',
    summary: 'Execution manifest not recorded on run payload',
  })
}

export function formatQualificationBadge(
  receipt: ExecutionPhysicsReceipt,
): QualificationBadgeInfo {
  if (receipt.scientificQualification === false) {
    return {
      qualified: false,
      explicitUnqualified: true,
      label: 'Unqualified · Engineering Preview',
      detail:
        'scientific_qualification=false — compatibility or engineering preview execution only; not qualified for scientific signoff.',
    }
  }
  if (receipt.scientificQualification === true) {
    return {
      qualified: true,
      explicitUnqualified: false,
      label: 'Scientifically Qualified',
      detail: 'Execution manifest carries scientific_qualification=true.',
    }
  }
  return {
    qualified: false,
    explicitUnqualified: false,
    label: 'Unverified Qualification',
    detail:
      'Run payload did not carry an explicit scientific_qualification boolean; do not treat as scientific signoff.',
  }
}

import { describe, expect, it } from 'vitest';
import { configurationSchema } from './configuration-schema';
import type { ConfigurationSchemaMetadata } from './configuration-schema';

function checkReferences(document: Record<string, unknown>): void {
  function resolve(reference: string): void {
    expect(reference.startsWith('#/')).toBe(true);
    let target: unknown = document;
    for (const part of reference.slice(2).split('/')) {
      target = (target as Record<string, unknown>)[part.replace(/~1/g, '/').replace(/~0/g, '~')];
    }
    expect(target).toBeTypeOf('object');
    expect(target).not.toBeNull();
  }
  function walk(value: unknown): void {
    if (Array.isArray(value)) {
      value.forEach(walk);
    } else if (value !== null && typeof value === 'object') {
      const node = value as Record<string, unknown>;
      if (typeof node.$ref === 'string') resolve(node.$ref);
      const discriminator = node.discriminator as { mapping?: Record<string, string> } | undefined;
      Object.values(discriminator?.mapping ?? {}).forEach(resolve);
      Object.values(node).forEach(walk);
    }
  }
  walk(document);
}

describe('prepared configuration schema data', () => {
  it('keeps each complete DTO and scalar schema in its own reference scope', () => {
    Object.values(configurationSchema.dto_schemas).forEach(entry => checkReferences(entry.schema));
    Object.values(configurationSchema.parameter_schemas).forEach(checkReferences);
    const tandem = configurationSchema.dto_schemas.TandemInput.schema;
    expect(tandem.properties.top_cell.$ref).toBe('#/$defs/DeviceInput');
    expect(tandem.$defs.DeviceInput.properties.layers.items.$ref).toBe('#/$defs/FullLayerInput');
  });

  it('preserves editable presence, units, unions and source constraints', () => {
    const fields = configurationSchema.dto_schemas.FullParameterInput.schema.properties;
    expect(fields.mu_n).not.toHaveProperty('default');
    expect(fields.mu_n.unit).toBe('m^2/(V s)');
    expect(fields.mu_n.anyOf).not.toContainEqual({ type: 'null' });
    expect(fields.Nc300.anyOf).toContainEqual({ type: 'null' });
    expect(configurationSchema.dto_schemas.DeviceInput.schema.$defs.FullLayerInput.properties.bulk_defects.items.anyOf).toHaveLength(2);
    expect(configurationSchema.dto_schemas.CigsOpticsInput.schema.properties.slices.anyOf)
      .toContainEqual({ type: 'integer', minimum: 1, maximum: 512 });
  });

  it('exposes a metadata type without claiming a production or wire contract', () => {
    const metadata: ConfigurationSchemaMetadata = configurationSchema;
    expect(Object.keys(metadata.dto_schemas)).toHaveLength(20);
    expect(metadata.dto_schemas.SweepInput.representation).toBe('editable_input');
    expect(metadata.dto_schemas.JVExperimentInput.representation).toBe('editable_input');
    expect(metadata.dto_schemas.JVExperimentInput.schema.properties).not.toHaveProperty('can_execute');
    expect(metadata.dto_schemas.SpatialExperimentInput.representation).toBe('editable_input');
    expect(metadata.dto_schemas.SpatialExperimentInput.schema.properties.schema_version.const)
      .toBe('solarlab.experiment-preparation.v1');
    expect(metadata.dto_schemas.SpatialExperimentInput.schema.properties).not.toHaveProperty('can_execute');
    expect(metadata.status).toBe('prepared_pending_dependencies');
    expect(metadata.backend_semantic_validation_required).toBe(true);
    expect(metadata.structured_parameter_schemas).toEqual({});
    expect(metadata.pending).toContain('input_resolved_result_wire_types');
  });
});

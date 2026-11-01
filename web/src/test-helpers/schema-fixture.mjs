import { execFileSync } from 'node:child_process';
import { webcrypto } from 'node:crypto';
import { fileURLToPath } from 'node:url';

export { webcrypto };

/** Exact bytes from the real Python DTO exporter; Node-only test fixture. */
export function configurationSchemaBytes() {
  // Real Python DTO metadata supplies the bytes; JSON.stringify in JavaScript
  // cannot reproduce Python's numeric JSON spelling in general.
  const root = fileURLToPath(new URL('../../../', import.meta.url));
  const code = [
    'from pathlib import Path',
    'import sys',
    'import solarlab.config.schema as schema',
    'assert Path(schema.__file__).resolve() == Path("src/solarlab/config/schema.py").resolve()',
    'sys.stdout.buffer.write(schema.configuration_schema_representation()[0])',
  ].join('\n');
  const data = execFileSync(process.env.SOLARLAB_PYTHON || 'python3', ['-B', '-c', code], {
    cwd: root, env: { ...process.env, PYTHONPATH: `${root}/src` },
    timeout: 10000, maxBuffer: 4 * 1024 ** 2,
  });
  return Uint8Array.from(data).buffer;
}

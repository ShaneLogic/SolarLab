import { afterEach, describe, expect, it, vi } from 'vitest'
import { configurationSchema } from '../generated/configuration-schema'
import { mountQuantityInput } from './quantity-input'
import type { QuantityInput, QuantityInputOptions, QuantityValue } from './quantity-input'

const controls: QuantityInput[] = []
const fields = configurationSchema.dto_schemas.FullParameterInput.schema.properties

function mount(initial: QuantityValue, extra: Partial<QuantityInputOptions> = {}) {
  const root = document.createElement('div')
  document.body.append(root)
  const onChange = vi.fn()
  const control = mountQuantityInput(root, {
    name: 'mu_n', metadata: fields.mu_n, required: false, initial, onChange, ...extra,
  })
  controls.push(control)
  const input = root.querySelector<HTMLInputElement>('input')!
  const source = root.querySelector<HTMLSelectElement>('select')!
  function type(text: string) {
    input.value = text; input.dispatchEvent(new Event('input'))
  }
  function mode(value: string) {
    source.value = value; source.dispatchEvent(new Event('change'))
  }
  return { root, control, input, source, type, mode, onChange }
}

afterEach(() => {
  for (const control of controls.splice(0)) control.dispose()
  document.body.replaceChildren()
})

describe('quantity overrides from generated field metadata', () => {
  it.each([0, -0, 1.2345678901234567e-17, Number.MIN_VALUE, '1.234567890123456789e-17', '2 cm^2/(V s)'])(
    'retains the exact untouched value and type: %s', value => {
      const { control, onChange, mode } = mount({ kind: 'value', value })
      expect(control.read()).toStrictEqual({ kind: 'value', value })
      expect(onChange).not.toHaveBeenCalled()
      mode('omit'); expect(control.read()).toStrictEqual({ kind: 'omit' })
      mode('value'); expect(control.read()).toStrictEqual({ kind: 'value', value })
    },
  )

  it('takes an explicit text edit as input without converting units or rounding decimal digits', () => {
    const { control, type, onChange } = mount({ kind: 'value', value: 1e-4 })
    type('2.0000000000000000001 cm^2/(V s)')
    expect(control.read()).toStrictEqual({ kind: 'value', value: '2.0000000000000000001 cm^2/(V s)' })
    expect(onChange).toHaveBeenLastCalledWith(control.read())
    type('0'); expect(control.read()).toStrictEqual({ kind: 'value', value: '0' })
    type('-0'); expect(control.read()).toStrictEqual({ kind: 'value', value: '-0' })
  })

  it('does not infer zero, null or omission from empty text', () => {
    const { control, type, mode, input } = mount({ kind: 'value', value: 0 })
    type(' ')
    expect(control.read()).toMatchObject({ kind: 'incomplete', text: ' ' })
    expect(input.getAttribute('aria-invalid')).toBe('true')
    mode('omit')
    expect(control.read()).toStrictEqual({ kind: 'omit' })
    expect(input.disabled).toBe(true)
    mode('value')
    expect(control.read().kind).toBe('incomplete')
  })

  it('keeps omission, explicit null and a populated nullable field distinct', () => {
    const { control, mode, type, source } = mount({ kind: 'omit' }, { name: 'Nc300', metadata: fields.Nc300 })
    expect(control.read()).toStrictEqual({ kind: 'omit' })
    expect([...source.options].map(x => x.value)).toEqual(['value', 'omit', 'null'])
    mode('null'); expect(control.read()).toStrictEqual({ kind: 'value', value: null })
    mode('value'); expect(control.read().kind).toBe('incomplete')
    type('2e18 cm^-3'); expect(control.read()).toStrictEqual({ kind: 'value', value: '2e18 cm^-3' })
    mode('null'); mode('value')
    expect(control.read()).toStrictEqual({ kind: 'value', value: '2e18 cm^-3' })
    control.set({ kind: 'value', value: null })
    expect(control.read()).toStrictEqual({ kind: 'value', value: null })
    mode('value')
    expect(control.read()).toMatchObject({ kind: 'incomplete', message: 'Enter a value or select inheritance explicitly.' })
  })

  it('uses the required/nullability constraints without copying physical defaults', () => {
    const schema = configurationSchema.dto_schemas.FullLayerInput.schema
    const { control, source, root, type } = mount({ kind: 'omit' }, {
      name: 'thickness', metadata: schema.properties.thickness, required: schema.required.includes('thickness'),
    })
    expect([...source.options].map(x => x.value)).toEqual(['value'])
    expect(control.read()).toMatchObject({ kind: 'incomplete', message: 'Enter the required value.' })
    expect(root.textContent).toContain('m')
    type('400 nm')
    expect(control.read()).toStrictEqual({ kind: 'value', value: '400 nm' })
  })

  it('makes a nonnullable null editable without silently sending it as zero', () => {
    const { control, source, type } = mount({ kind: 'value', value: null })
    expect([...source.options].some(x => x.value === 'null')).toBe(false)
    expect(control.read().kind).toBe('incomplete')
    type('1e-4'); expect(control.read()).toStrictEqual({ kind: 'value', value: '1e-4' })
  })

  it('resets to new caller input without emitting an edit or resurrecting an earlier value', () => {
    const { control, onChange, type, mode } = mount({ kind: 'value', value: -0 })
    type('5'); onChange.mockClear()
    control.set({ kind: 'value', value: '6e-8 m^2/(V s)' })
    expect(control.read()).toStrictEqual({ kind: 'value', value: '6e-8 m^2/(V s)' })
    control.set({ kind: 'omit' })
    expect(control.read()).toStrictEqual({ kind: 'omit' })
    expect(onChange).not.toHaveBeenCalled()
    mode('value'); expect(control.read().kind).toBe('incomplete')
  })

  it('renders supplied text safely and leaves semantic unit validation to the backend', () => {
    const { root, control, type } = mount({ kind: 'value', value: 1 }, {
      name: '<script>field</script>', metadata: { ...fields.mu_n, title: '<img src=x>', unit: '<b>unit</b>' },
    })
    expect(root.querySelector('script,img,b')).toBeNull()
    expect(root.textContent).toContain('<img src=x>')
    type('invalid physical unit')
    expect(control.read()).toStrictEqual({ kind: 'value', value: 'invalid physical unit' })
  })

  it('detaches only its own control and listeners on disposal', () => {
    const one = mount({ kind: 'value', value: 1 }), two = mount({ kind: 'value', value: 2 })
    const replacement = document.createElement('p')
    one.root.replaceChildren(replacement)
    one.control.dispose()
    one.type('3'); one.mode('omit'); one.control.set({ kind: 'value', value: 4 })
    expect(one.onChange).not.toHaveBeenCalled()
    expect(one.root.firstChild).toBe(replacement)
    expect(two.control.read()).toStrictEqual({ kind: 'value', value: 2 })
    expect(two.input.isConnected).toBe(true)
  })
})

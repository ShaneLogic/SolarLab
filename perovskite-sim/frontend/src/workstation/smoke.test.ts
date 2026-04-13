import { describe, it, expect } from 'vitest'

describe('vitest infrastructure', () => {
  it('runs in jsdom', () => {
    const div = document.createElement('div')
    div.textContent = 'hello'
    expect(div.textContent).toBe('hello')
  })
})

import test from 'node:test'
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { parse, compileStyle } from '@vue/compiler-sfc'

// User-approved visual baseline: 4d0bcd9aae47ae2b51eb7277f3639d0d97875e14.
// That version includes the cyber dashboard AND the right-hand drawer; the
// literal first version (22e12e1) did not yet have a drawer. Keep this small
// design contract independent of Git, machine paths, copy, and API responses.
// This is a source-level regression gate, not a pixel or accessibility test.
const filename = new URL('../src/App.vue', import.meta.url)
const source = readFileSync(filename, 'utf8')
const { descriptor, errors } = parse(source, { filename: 'App.vue' })
assert.deepEqual(errors, [], 'App.vue must remain a valid single-file component')
const template = descriptor.template.ast
const compiled = compileStyle({
  source: descriptor.styles.map(style => style.content).join('\n'),
  filename: 'App.vue', id: 'visual-contract', scoped: false,
})
assert.deepEqual(compiled.errors, [], 'Visual contract needs valid CSS')
const css = compiled.rawResult.root

const elements = node => (node.children || []).filter(child => child.type === 1)
const walk = node => [node, ...elements(node).flatMap(walk)]
const attr = (node, name) => node.props?.find(prop => prop.type === 6 && prop.name === name)?.value?.content
const directive = (node, name, arg) => node.props?.find(prop =>
  prop.type === 7 && prop.name === name && (arg === undefined || prop.arg?.content === arg))
const hasClass = (node, value) => (attr(node, 'class') || '').split(/\s+/).includes(value)
const findClass = (node, value) => walk(node).filter(child => hasClass(child, value))
const oneClass = (node, value) => {
  const matches = findClass(node, value)
  assert.equal(matches.length, 1, `Expected exactly one .${value}`)
  return matches[0]
}
const root = elements(template)[0]
const normalize = value => value.trim().replace(/\s+/g, ' ').replace(/\s*,\s*/g, ',').toLowerCase()

// Apply later declarations for the same top-level selector, so accidentally
// appending a replacement rule cannot silently preserve a passing old rule.
// Media-query adaptations and setting-panel scrolling are intentionally free.
function declarations(selector) {
  const result = new Map()
  for (const node of css.nodes) {
    if (node.type !== 'rule' || !node.selectors.includes(selector)) continue
    for (const declaration of node.nodes) {
      if (declaration.type !== 'decl') continue
      const previous = result.get(declaration.prop)
      if (!previous?.important || declaration.important) result.set(declaration.prop, declaration)
    }
  }
  return result
}
function expectCss(selector, expected) {
  const actual = declarations(selector)
  for (const [property, value] of Object.entries(expected)) {
    assert.equal(normalize(actual.get(property)?.value || ''), normalize(value), `${selector}: ${property}`)
  }
}

test('original dashboard hierarchy keeps the header, summary and four chart cards', () => {
  assert.ok(hasClass(root, 'dashboard-container'))
  assert.ok(directive(root, 'bind', 'style'), 'Theme variables must reach the dashboard')
  assert.ok(directive(root, 'bind', 'class'), 'Light mode must remain available')
  const children = elements(root)
  const header = children.find(node => hasClass(node, 'tech-header'))
  const summary = children.find(node => hasClass(node, 'summary-card'))
  const grid = children.find(node => hasClass(node, 'dashboard-grid'))
  assert.ok(header && summary && grid, 'Keep the original top-level visual hierarchy')
  assert.ok(children.indexOf(header) < children.indexOf(summary) && children.indexOf(summary) < children.indexOf(grid))
  assert.equal(grid.tag, 'main')
  const cards = elements(grid)
  assert.equal(cards.length, 4, 'The main dashboard remains a two-by-two chart layout')
  for (const card of cards) {
    assert.ok(hasClass(card, 'tech-card'))
    oneClass(card, 'card-header')
    const chart = oneClass(card, 'chart-container')
    assert.ok(attr(chart, 'ref'), 'Each original chart position remains a chart mount')
  }
  assert.ok(hasClass(cards[0], 'interactive-card'), 'Keep the interactive intake card')
  expectCss('.dashboard-grid', { display: 'grid', 'grid-template-columns': 'repeat(2, 1fr)', gap: '2rem' })
  expectCss('.chart-container', { width: '100%', height: '320px' })
})

test('original language, light mode, highlight color and font controls stay in settings', () => {
  const header = oneClass(root, 'tech-header')
  const settings = oneClass(header, 'settings-panel')
  assert.ok(hasClass(settings, 'tech-card'))
  const controls = walk(settings)
  const selects = controls.filter(node => node.tag === 'select' && directive(node, 'model'))
  const optionValues = select => elements(select).filter(node => node.tag === 'option').map(node => attr(node, 'value'))
  assert.ok(selects.some(select => ['zh', 'en'].every(value => optionValues(select).includes(value))), 'Keep both original languages')
  const fonts = ["'Segoe UI', 'Microsoft YaHei', sans-serif", 'Helvetica, Arial, sans-serif', "'Times New Roman', Times, serif", "'Courier New', Courier, monospace"]
  assert.ok(selects.some(select => fonts.every(font => optionValues(select).includes(font))), 'Keep the four original font choices')
  assert.ok(controls.some(node => node.tag === 'input' && attr(node, 'type') === 'color' && directive(node, 'model')), 'Keep the bound highlight color picker')
  const themeSwitch = oneClass(settings, 'theme-switch')
  assert.ok(directive(themeSwitch, 'on', 'click'), 'Keep the light/dark switch interactive')
  oneClass(themeSwitch, 'switch-bg')
  assert.equal(findClass(themeSwitch, 'switch-label').length, 2)
  expectCss('.settings-panel', { position: 'absolute', right: '0', width: '320px' })
})

test('the cyber palette, glass cards and gradient title retain the approved design', () => {
  // Values live in the reactive theme as well as CSS; do not lock script names
  // or implementation syntax, only the palette and font-variable capability.
  for (const value of ['#00f3ff', '#ff00ea', '#111827', '#030712', '#f0f2f5', '#e2e8f0', '--primary-color', '--font-family']) {
    assert.ok(source.includes(value), `Preserve original theme value ${value}`)
  }
  oneClass(root, 'glow-text')
  oneClass(root, 'version-tag')
  expectCss('.dashboard-container', {
    background: 'var(--bg-gradient)', color: 'var(--text-main)', 'font-family': 'var(--font-family)', padding: '2rem',
  })
  expectCss('.tech-card', {
    background: 'var(--card-bg)', 'backdrop-filter': 'blur(20px)', 'border-radius': '16px',
  })
  expectCss('.glow-text', {
    background: 'linear-gradient(to right, var(--primary-color), #ff00ea)', '-webkit-background-clip': 'text', color: 'transparent',
  })
  expectCss('.version-tag', { background: 'linear-gradient(45deg, #ff00ea, var(--primary-color))' })
})

test('the record drawer stays on the right with the original glass overlay and transitions', () => {
  const drawer = oneClass(root, 'drawer-panel')
  assert.ok(hasClass(drawer, 'tech-card'))
  const overlay = oneClass(root, 'drawer-overlay')
  const transitions = walk(root).filter(node => node.tag.toLowerCase() === 'transition')
  assert.ok(transitions.some(node => attr(node, 'name') === 'slide' && walk(node).includes(drawer)))
  assert.ok(transitions.some(node => attr(node, 'name') === 'fade' && walk(node).includes(overlay)))
  oneClass(drawer, 'drawer-header')
  oneClass(drawer, 'close-btn')
  expectCss('.drawer-panel', {
    position: 'fixed', top: '0', right: '0', width: '450px', 'max-width': '90vw', height: '100vh',
    'border-radius': '20px 0 0 20px', 'border-left-width': '4px', 'border-left-style': 'solid',
    background: 'var(--card-bg)', 'backdrop-filter': 'blur(20px)', 'overflow-y': 'auto',
  })
  expectCss('.drawer-overlay', { position: 'fixed', inset: '0', background: 'rgba(0, 0, 0, 0.4)', 'backdrop-filter': 'blur(5px)' })
  expectCss('.slide-enter-active', { transition: 'transform 0.4s cubic-bezier(0.4, 0, 0.2, 1)' })
  expectCss('.slide-leave-active', { transition: 'transform 0.4s cubic-bezier(0.4, 0, 0.2, 1)' })
  expectCss('.slide-enter-from', { transform: 'translateX(100%)' })
  expectCss('.slide-leave-to', { transform: 'translateX(100%)' })
  expectCss('.fade-enter-active', { transition: 'opacity 0.4s ease' })
  expectCss('.fade-leave-active', { transition: 'opacity 0.4s ease' })
  expectCss('.fade-enter-from', { opacity: '0' })
  expectCss('.fade-leave-to', { opacity: '0' })
})

test('interface and data-management controls stay inside the original settings panel', () => {
  const settings = oneClass(root, 'settings-panel')
  for (const id of ['admin-key', 'connect', 'disconnect', 'backup', 'restore-file', 'restore', 'delete-all']) {
    const matches = walk(root).filter(node => attr(node, 'data-testid') === id)
    assert.equal(matches.length, 1, `Expected the ${id} interface control`)
    assert.ok(walk(settings).includes(matches[0]), `${id} must not add a new card to the original homepage`)
  }
})

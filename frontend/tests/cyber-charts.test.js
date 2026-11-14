import test from 'node:test'
import assert from 'node:assert/strict'
import { buildCyberChartOptions, statisticalStructure } from '../src/cyber-charts.js'
import { CATEGORY_KEYS, emptyAnalysis } from '../src/dashboard-data.js'

const settings = { isLightMode: false, themeColor: '#00f3ff', fontFamily: 'Test Font', lang: 'zh' }
const labels = { categoryName: key => `类别 ${key}`, noData: '无数据', missingMetric: '缺少指标', returnGlobal: '返回全局', sent: ['积极', '中性', '消极'], graphRoot: '本次样本' }
const series = () => ({
  dates: ['2026-09-22', '2026-09-23', '2026-09-24'], repeat: [0, null, 25],
  positive: [100, null, 0], neutral: [0, null, 100], negative: [0, null, 0],
})
const ready = () => ({ status: 'ready', counts: { learning: 4 }, series: { global: series(), learning: series() } })
const options = (analysis, category = 'global') => buildCyberChartOptions({ analysis, category, settings, labels })

test('non-ready snapshots clear every chart even when stale values remain in the input', () => {
  for (const status of ['not_requested', 'loading', 'failed', 'unavailable', 'empty', 'insufficient_data', 'request_error']) {
    const charts = options({ ...ready(), status })
    for (const chart of Object.values(charts)) {
      assert.deepEqual(chart.series, [], status)
      assert.equal(chart.graphic[0].style.text, labels.noData)
    }
  }
  for (const chart of Object.values(options(emptyAnalysis()))) assert.deepEqual(chart.series, [])
})

test('seven real categories remain separate in the ring and statistical structure', () => {
  const analysis = ready()
  analysis.counts = Object.fromEntries(CATEGORY_KEYS.map((key, index) => [key, index + 1]))
  const charts = options(analysis)
  assert.deepEqual(charts.pie.series[0].data.map(item => item.id), CATEGORY_KEYS)
  assert.deepEqual(charts.pie.series[0].data.map(item => item.value), [1, 2, 3, 4, 5, 6, 7])
  const graph = charts.graph.series[0]
  assert.equal(graph.data[0].name, labels.graphRoot)
  assert.equal(graph.data[0].value, 28)
  assert.deepEqual(graph.data.slice(1).map(node => node.id), CATEGORY_KEYS.map(key => `category:${key}`))
  assert.equal(graph.links.length, 7)
  assert.ok(graph.links.every(link => link.source === 'sample-root'))
})

test('daily charts retain actual dates, zero percentages and null gaps', () => {
  const charts = options(ready(), 'learning')
  assert.deepEqual(charts.repetition.xAxis.data, series().dates)
  assert.deepEqual(charts.repetition.series[0].data, [0, null, 25])
  assert.equal(charts.repetition.series[0].connectNulls, false)
  assert.deepEqual(charts.sentiment.series.map(item => item.data), [[100, null, 0], [0, null, 100], [0, null, 0]])
  assert.ok(charts.sentiment.series.every(item => item.connectNulls === false))
  const structure = statisticalStructure({ analysis: ready(), category: 'learning', settings, labels })
  assert.deepEqual(structure.nodes.slice(1).map(node => [node.id, node.name, node.value]), [
    ['date:2026-09-22', '2026-09-22\n0%', 0], ['date:2026-09-24', '2026-09-24\n25%', 25],
  ])
  assert.equal(structure.links.length, 2)
})

test('missing category metrics never fall back to global values and zero counts create no nodes', () => {
  const analysis = ready()
  analysis.counts.other = 0
  const charts = options(analysis, 'other')
  assert.deepEqual(charts.pie.series[0].data.map(item => item.id), ['learning'])
  for (const key of ['graph', 'repetition', 'sentiment']) {
    assert.deepEqual(charts[key].series, [])
    assert.equal(charts[key].graphic[0].style.text, labels.missingMetric)
  }
  analysis.counts = { other: 0 }
  assert.deepEqual(statisticalStructure({ analysis, category: 'global', settings, labels }), { nodes: [], links: [] })
})

test('isolated 0% and 100% observations remain visible without connecting null gaps', () => {
  const charts = options(ready())
  for (const line of charts.sentiment.series) {
    assert.equal(line.connectNulls, false)
    assert.equal(line.showSymbol, true)
    assert.equal(line.showAllSymbol, true, 'Sparse observations must not depend on axis-label thinning')
    assert.deepEqual(line.data.map((value, dataIndex) => line.symbolSize(value, { dataIndex })), [6, 0, 6])
  }
  assert.deepEqual(charts.sentiment.series[0].data, [100, null, 0])
  const repeat = charts.repetition.series[0]
  assert.deepEqual(repeat.data, [0, null, 25])
  assert.equal(repeat.connectNulls, false)
  assert.equal(repeat.showSymbol, true)
  assert.equal(repeat.showAllSymbol, true)
  assert.ok(repeat.symbolSize > 0)

  const connected = ready()
  connected.series.global.positive = [100, 50, 0]
  const line = options(connected).sentiment.series[0]
  assert.deepEqual(line.data.map((value, dataIndex) => line.symbolSize(value, { dataIndex })), [0, 0, 0], 'Connected sentiment curves keep the original marker-free design')
})

test('the approved chart appearance and theme survive the data adapter', () => {
  const charts = options(ready(), 'learning')
  assert.deepEqual(charts.pie.series[0].radius, ['45%', '75%'])
  assert.deepEqual(charts.pie.series[0].center, ['50%', '45%'])
  assert.deepEqual(charts.graph.series[0].force, { repulsion: 200, edgeLength: 60 })
  assert.equal(charts.graph.series[0].lineStyle.curveness, 0.3)
  assert.equal(charts.repetition.series[0].lineStyle.shadowBlur, 15)
  assert.equal(charts.repetition.series[0].lineStyle.color, settings.themeColor)
  assert.equal(charts.repetition.series[0].areaStyle.color.colorStops[0].color, '#00f3ff90')
  for (const chart of Object.values(charts)) {
    assert.equal(chart.tooltip.renderMode, 'richText')
    assert.equal(chart.textStyle.fontFamily, settings.fontFamily)
  }
  const light = buildCyberChartOptions({ analysis: ready(), settings: { ...settings, isLightMode: true, themeColor: '#123456' }, labels })
  assert.equal(light.pie.series[0].itemStyle.borderColor, '#fff')
  assert.equal(light.repetition.series[0].lineStyle.color, '#123456')
  assert.equal(light.graph.series[0].lineStyle.color, 'rgba(0,0,0,0.1)')
})

test('return-to-global invokes the provided selection callback without inventing analysis', () => {
  const selected = []
  const state = { analysis: ready(), category: 'learning', settings, labels }
  const before = JSON.stringify(state)
  const charts = buildCyberChartOptions(state, key => selected.push(key))
  charts.pie.graphic[0].onclick()
  assert.deepEqual(selected, ['global'])
  assert.equal(JSON.stringify(state), before)
})

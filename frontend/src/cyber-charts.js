import * as echarts from 'echarts'
import { CATEGORY_KEYS } from './dashboard-data.js'

// Original chart styling: 4d0bcd9. Values always come from one validated
// analysis snapshot; these edges describe statistical groupings, not knowledge
// relationships or the user's health, emotions, or browsing behaviour.
const COLORS = {
  entertainment: '#ff00ea', learning: '#00ffaa', news: '#ffd700', social: '#ff4d4f',
  shopping: '#b59aff', tools: '#4da6ff', other: '#94a3b8',
}
const percentage = value => typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 100 ? value : null
const colorFor = (key, settings) => Object.hasOwn(COLORS, key) ? COLORS[key] : settings.themeColor
const emptySeries = () => ({ dates: [], repeat: [], positive: [], neutral: [], negative: [] })

// The original sentiment curves hide markers. With truthful null gaps, a
// single measured day has no line segment, so expose only those isolated
// observations. Keep connected runs visually identical to the original.
const isolatedPointSize = values => (_value, { dataIndex }) => (
  values[dataIndex] !== null
  && (dataIndex === 0 || values[dataIndex - 1] === null)
  && (dataIndex === values.length - 1 || values[dataIndex + 1] === null)
    ? 6 : 0
)

function categoryCounts(analysis) {
  if (analysis?.status !== 'ready') return []
  return Object.entries(analysis.counts || {})
    .filter(([, value]) => Number.isSafeInteger(value) && value > 0)
    .sort(([a], [b]) => {
      const rank = key => CATEGORY_KEYS.includes(key) ? CATEGORY_KEYS.indexOf(key) : CATEGORY_KEYS.length
      return rank(a) - rank(b) || a.localeCompare(b)
    })
}

function selectedSeries(analysis, category) {
  if (analysis?.status !== 'ready') return emptySeries()
  // A missing category must stay empty; borrowing global data would falsely
  // present the global measurements as measurements of that category.
  const supplied = analysis.series?.[category]
  if (!Array.isArray(supplied?.dates)) return emptySeries()
  const result = { dates: [...supplied.dates] }
  for (const metric of ['repeat', 'positive', 'neutral', 'negative']) {
    result[metric] = result.dates.map((_, index) => percentage(supplied[metric]?.[index]))
  }
  return result
}

export function statisticalStructure({ analysis, category = 'global', settings, labels }) {
  const nodes = [], links = []
  if (analysis?.status !== 'ready') return { nodes, links }
  if (category === 'global') {
    const counts = categoryCounts(analysis)
    if (!counts.length) return { nodes, links }
    nodes.push({
      id: 'sample-root', name: labels.graphRoot, value: counts.reduce((sum, [, count]) => sum + count, 0),
      symbolSize: 40, itemStyle: { color: settings.themeColor, shadowBlur: 15, shadowColor: settings.themeColor },
    })
    for (const [key, count] of counts) {
      const color = colorFor(key, settings), size = Math.min(Math.max(count * 2, 15), 50)
      const id = `category:${key}`
      nodes.push({ id, name: labels.categoryName(key), value: count, symbolSize: size, itemStyle: { color, shadowBlur: 10, shadowColor: color } })
      links.push({ source: 'sample-root', target: id, lineStyle: { width: size / 10 } })
    }
  } else {
    const data = selectedSeries(analysis, category)
    if (!data.repeat.some(value => value !== null)) return { nodes, links }
    const color = colorFor(category, settings)
    nodes.push({ id: 'category-root', name: labels.categoryName(category), symbolSize: 50, itemStyle: { color, shadowBlur: 20, shadowColor: color } })
    data.repeat.forEach((value, index) => {
      if (value === null) return
      const id = `date:${data.dates[index]}`
      nodes.push({
        id, name: `${data.dates[index]}\n${value}%`, value, symbolSize: Math.max(10, value / 3),
        itemStyle: { color: settings.isLightMode ? '#94a3b8' : 'rgba(255,255,255,0.6)' },
      })
      links.push({ source: 'category-root', target: id, lineStyle: { width: 1.5 } })
    })
  }
  return { nodes, links }
}

// Pure option builder makes data truthfulness testable without canvas or a
// browser. The four returned options are complete replacements on each render.
export function buildCyberChartOptions({ analysis, category = 'global', settings, labels }, onSelect = () => {}) {
  const ui = {
    text: settings.isLightMode ? '#334155' : '#cbd5e1',
    line: settings.isLightMode ? '#cbd5e1' : '#334155',
    tooltip: settings.isLightMode ? 'rgba(255,255,255,0.95)' : 'rgba(15,23,42,0.95)',
    fontFamily: settings.fontFamily,
  }
  const textStyle = { fontFamily: ui.fontFamily }
  const tooltip = trigger => ({
    trigger, renderMode: 'richText', backgroundColor: ui.tooltip,
    textStyle: { color: ui.text, fontFamily: ui.fontFamily }, borderWidth: 0,
  })
  const emptyGraphic = text => ({
    type: 'text', left: 'center', top: 'center',
    style: { text, fill: ui.text, fontFamily: ui.fontFamily, fontSize: 16, fontWeight: 'bold', opacity: 0.6 },
  })
  const noMetric = analysis?.status === 'ready' ? labels.missingMetric : labels.noData
  const pieData = categoryCounts(analysis).map(([id, value]) => ({ id, name: labels.categoryName(id), value, itemStyle: { color: colorFor(id, settings) } }))
  const pieGraphics = pieData.length ? [] : [emptyGraphic(labels.noData)]
  if (category !== 'global' && pieData.length) {
    pieGraphics.push({
      type: 'group', left: 'center', top: '40%', cursor: 'pointer', onclick: () => onSelect('global'),
      children: [
        { type: 'rect', left: 'center', top: 'center', shape: { r: 16, width: 90, height: 32 }, style: { fill: settings.isLightMode ? 'rgba(0,0,0,0.05)' : 'rgba(255,255,255,0.08)', stroke: settings.themeColor, lineWidth: 1 } },
        { type: 'text', left: 'center', top: 'center', style: { text: labels.returnGlobal, fill: settings.themeColor, fontFamily: ui.fontFamily, fontSize: 13, fontWeight: 'bold' } },
      ],
    })
  }
  const pie = {
    textStyle, tooltip: tooltip('item'),
    legend: { show: Boolean(pieData.length), bottom: '0%', textStyle: { color: ui.text, fontFamily: ui.fontFamily } },
    graphic: pieGraphics,
    series: pieData.length ? [{
      type: 'pie', radius: ['45%', '75%'], center: ['50%', '45%'],
      itemStyle: { borderRadius: 8, borderColor: settings.isLightMode ? '#fff' : '#0f172a', borderWidth: 3 },
      label: { show: false }, emphasis: { scaleSize: 10, itemStyle: { shadowBlur: 20, shadowColor: 'rgba(0,0,0,0.5)' } }, data: pieData,
    }] : [],
  }
  const structure = statisticalStructure({ analysis, category, settings, labels })
  const graph = {
    textStyle, tooltip: tooltip('item'), graphic: structure.nodes.length ? [] : [emptyGraphic(noMetric)],
    series: structure.nodes.length ? [{
      type: 'graph', layout: 'force', data: structure.nodes, links: structure.links, roam: true, draggable: true,
      label: { fontFamily: ui.fontFamily }, force: { repulsion: 200, edgeLength: 60 },
      lineStyle: { color: settings.isLightMode ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.2)', curveness: 0.3 },
    }] : [],
  }
  const data = selectedSeries(analysis, category)
  const axes = valid => ({
    xAxis: { show: valid, type: 'category', data: data.dates, axisLine: { lineStyle: { color: ui.line } }, axisLabel: { color: ui.text, fontFamily: ui.fontFamily } },
    yAxis: { show: valid, type: 'value', min: 0, max: 100, splitLine: { lineStyle: { color: ui.line, type: 'dashed' } }, axisLabel: { color: ui.text, fontFamily: ui.fontFamily, formatter: '{value}%' } },
  })
  const hasRepeat = data.repeat.some(value => value !== null)
  const repetition = {
    textStyle, tooltip: tooltip('axis'), grid: { left: '8%', right: '5%', bottom: '15%', top: '15%' }, ...axes(hasRepeat),
    graphic: hasRepeat ? [] : [emptyGraphic(noMetric)],
    series: hasRepeat ? [{
      data: data.repeat, type: 'line', smooth: true, smoothMonotone: 'x', connectNulls: false,
      showSymbol: true, showAllSymbol: true, symbolSize: 6,
      lineStyle: { color: settings.themeColor, width: 4, shadowColor: settings.themeColor, shadowBlur: 15 },
      itemStyle: { color: settings.themeColor, borderWidth: 2, borderColor: '#fff' },
      areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: `${settings.themeColor}90` }, { offset: 1, color: 'transparent' }]) },
    }] : [],
  }
  const hasSentiment = ['positive', 'neutral', 'negative'].some(metric => data[metric].some(value => value !== null))
  const sentiment = {
    textStyle, tooltip: tooltip('axis'),
    legend: { show: hasSentiment, top: '0%', textStyle: { color: ui.text, fontFamily: ui.fontFamily }, icon: 'circle' },
    grid: { left: '8%', right: '5%', bottom: '20%', top: '20%' }, ...axes(hasSentiment),
    graphic: hasSentiment ? [] : [emptyGraphic(noMetric)],
    series: hasSentiment ? [
      { name: labels.sent[0], type: 'line', smooth: true, smoothMonotone: 'x', connectNulls: false, data: data.positive, itemStyle: { color: '#00ffaa' }, lineStyle: { width: 3 }, showSymbol: true, showAllSymbol: true, symbolSize: isolatedPointSize(data.positive) },
      { name: labels.sent[1], type: 'line', smooth: true, smoothMonotone: 'x', connectNulls: false, data: data.neutral, itemStyle: { color: ui.text }, lineStyle: { width: 2, type: 'dotted' }, showSymbol: true, showAllSymbol: true, symbolSize: isolatedPointSize(data.neutral) },
      { name: labels.sent[2], type: 'line', smooth: true, smoothMonotone: 'x', connectNulls: false, data: data.negative, itemStyle: { color: '#ff4d4f' }, lineStyle: { width: 3 }, showSymbol: true, showAllSymbol: true, symbolSize: isolatedPointSize(data.negative) },
    ] : [],
  }
  return { pie, graph, repetition, sentiment }
}

export function createCyberCharts(elements, onSelect = () => {}) {
  const names = ['pie', 'graph', 'repetition', 'sentiment']
  for (const name of names) {
    if (!elements[name]) throw new Error(`Missing ${name} chart container`)
  }
  const charts = {}
  try {
    for (const name of names) charts[name] = echarts.init(elements[name])
  } catch (error) {
    for (const chart of Object.values(charts)) chart.dispose()
    throw error
  }
  let disposed = false, selectableCategories = new Set()
  const select = category => {
    if (!disposed && (category === 'global' || selectableCategories.has(category))) onSelect(category)
  }
  charts.pie.on('click', event => {
    if (event.componentType === 'series' && event.data?.id) select(event.data.id)
  })
  return {
    render(state) {
      if (disposed) return
      selectableCategories = new Set(categoryCounts(state.analysis).map(([key]) => key))
      const options = buildCyberChartOptions(state, select)
      for (const name of names) charts[name].setOption(options[name], { notMerge: true })
    },
    resize() { if (!disposed) for (const chart of Object.values(charts)) chart.resize() },
    dispose() {
      if (disposed) return
      disposed = true
      for (const chart of Object.values(charts)) chart.dispose()
      selectableCategories.clear()
    },
  }
}

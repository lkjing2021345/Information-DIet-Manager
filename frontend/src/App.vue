<template>
  <div class="dashboard-container" :style="customStyles" :class="{'light-mode': settings.isLightMode}">
    
    <div class="global-loading" v-if="isUpdating">
      <div class="cyber-spinner"></div>
      <div class="loading-text" :style="{ fontFamily: settings.fontFamily }">
        {{ t.loading.replace('{cat}', t.cats[currentCategoryKey]) }}
      </div>
    </div>

    <header class="tech-header">
      <h1 class="glow-text">Information Diet Manager <span class="version-tag">v2.0 PRO</span></h1>
      
      <div class="header-actions">
        <div class="settings-wrapper">
          <button class="icon-btn" @click="toggleSettings" title="设置 / Settings">
            <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
          </button>
          
          <div class="settings-panel tech-card" v-show="showSettings">
            <h3>{{ t.sysUi }}</h3>
            
            <div class="setting-item">
              <label>{{ t.langLabel }}</label>
              <select v-model="settings.lang">
                <option value="zh">🇨🇳 简体中文 (Chinese)</option>
                <option value="en">🇺🇸 English (英文)</option>
              </select>
            </div>

            <div class="setting-item">
              <label>{{ t.visMode }}</label>
              <div class="theme-switch" @click="settings.isLightMode = !settings.isLightMode">
                <div class="switch-bg" :class="{ 'is-light': settings.isLightMode }"></div>
                <span class="switch-label" :class="{ active: !settings.isLightMode }">{{ t.dark }}</span>
                <span class="switch-label" :class="{ active: settings.isLightMode }">{{ t.light }}</span>
              </div>
            </div>
            
            <div class="setting-item">
              <label>{{ t.color }}</label>
              <input type="color" v-model="settings.themeColor">
            </div>
            
            <div class="setting-item">
              <label>{{ t.font }}</label>
              <select v-model="settings.fontFamily">
                <option value="'Segoe UI', 'Microsoft YaHei', sans-serif">UI Sans-serif (现代无衬线)</option>
                <option value="Helvetica, Arial, sans-serif">Helvetica (经典纯净)</option>
                <option value="'Times New Roman', Times, serif">Times New Roman (衬线阅读)</option>
                <option value="'Courier New', Courier, monospace">Courier Code (极客等宽)</option>
              </select>
            </div>
          </div>
        </div>

        <div class="status-badge" :class="healthStatus.class">
          {{ t.health }}: <span class="score">{{ healthStatus.score }}</span>
        </div>
      </div>
    </header>

    <div class="summary-card tech-card" :class="healthStatus.borderClass">
      <div class="icon-pulse" :style="{ backgroundColor: healthStatus.color, boxShadow: `0 0 0 0 ${healthStatus.color}80` }"></div>
      <div class="summary-content">
        <h2>{{ t.report }}: {{ t.cats[currentCategoryKey] }} {{ t.domain }} <span class="hint-text">{{ t.hint }}</span></h2>
        <p class="summary-text" v-html="healthStatus.message"></p>
      </div>
    </div>

    <main class="dashboard-grid">
      <div class="tech-card interactive-card">
        <div class="card-header"><span class="dot"></span><h2>{{ t.c1 }}</h2></div>
        <div ref="pieChartRef" class="chart-container"></div>
      </div>

      <div class="tech-card">
        <div class="card-header"><span class="dot"></span><h2>{{ t.cats[currentCategoryKey] }} - {{ t.c2 }}</h2></div>
        <div ref="graphChartRef" class="chart-container"></div>
      </div>

      <div class="tech-card">
        <div class="card-header"><span class="dot"></span><h2>{{ t.cats[currentCategoryKey] }} - {{ t.c3 }}</h2></div>
        <div ref="lineRepetitionRef" class="chart-container"></div>
      </div>

      <div class="tech-card">
        <div class="card-header"><span class="dot"></span><h2>{{ t.cats[currentCategoryKey] }} - {{ t.c4 }}</h2></div>
        <div ref="lineSentimentRef" class="chart-container"></div>
      </div>
    </main>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import * as echarts from 'echarts'
import axios from 'axios' // 引入 HTTP 请求引擎

const showSettings = ref(false)
const toggleSettings = () => showSettings.value = !showSettings.value
const isUpdating = ref(true) // 初始状态设为 true，等待首屏数据加载
const currentCategoryKey = ref('global')

const settings = reactive({
  lang: 'zh',
  themeColor: '#00f3ff',
  fontFamily: "'Segoe UI', 'Microsoft YaHei', sans-serif",
  isLightMode: false
})

// === 配置后端 API 基础地址 ===
// 请根据你们 Python 后端实际运行的端口修改 (通常是 8000 或 5000)
const API_BASE_URL = 'http://127.0.0.1:8000'

// === i18n 多语言国际化字典引擎 (保持不变) ===
const i18n = {
  zh: {
    sysUi: '系统级 UI 定制', langLabel: '🌐 界面语言 (Language)', visMode: '视觉引擎模式',
    dark: '🌙 极客暗黑', light: '☀️ 护眼明亮', color: '神经突触高亮色', font: '全局渲染字体族',
    health: '生态健康度', report: '深度剖析报告', domain: '领域', hint: '(💡 点击左侧饼图交互)', loading: '正在通过神经网络链接服务器获取 [{cat}] 数据...',
    c1: '摄入结构特征 (点击切片下钻)', c2: '信息群落知识图谱', c3: '深度复读率走势', c4: '情感共振频段分析',
    week: ['D-6', 'D-5', 'D-4', 'D-3', 'D-2', '昨日', '今日'], sent: ['多巴胺(正向)', '客观(中性)', '焦虑(负向)'],
    cats: { global: '全局概览', ent: '娱乐', edu: '学习', news: '新闻', soc: '社交' },
    eval: {
      entScore: 'C- 茧房警报', entMsg: `您在 <b style="color:#ff00ea; font-family:{f}">娱乐</b> 领域的摄入处于<b>高频复读状态</b>。算法正向您投喂同质化多巴胺内容，强烈建议跳出舒适区！`,
      eduScore: 'A+ 极佳', eduMsg: `您在 <b style="color:#00ffaa; font-family:{f}">学习</b> 领域的信息图谱展现出极高的<b>多样性与低重复度</b>。继续保持这种高质量的脑力体操！`,
      newsScore: 'B- 情绪波动', newsMsg: `该领域的信息密度正常，但<b>负面情绪指数较高</b>。建议适当控制宏观叙事信息的摄入，关注真实生活。`,
      socScore: 'B 圈层固化', socMsg: `您在 <b style="color:#ff4d4f; font-family:{f}">社交</b> 领域的信息交互呈现明显的<b>信息同温层效应 (回音室)</b>。建议引入不同视角的观点。`,
      globScore: 'B+ 亚健康', globMsg: `当前整体信息流呈现 <b style="color:var(--primary-color); font-family:{f}">娱乐内容过载</b> 倾向。点击左下方饼图的各个切片，深度下钻您的信息群落。`
    }
  },
  en: {
    sysUi: 'System UI Engine', langLabel: '🌐 Interface Language', visMode: 'Visual Mode',
    dark: '🌙 Cyber Dark', light: '☀️ Clean Light', color: 'Synapse Highlight Color', font: 'Global Font Family',
    health: 'Eco-Health', report: 'In-Depth Report', domain: 'Domain', hint: '(💡 Click pie chart to drill down)', loading: 'Connecting server to fetch [{cat}] models...',
    c1: 'Intake Structure (Interactive)', c2: 'Information Cluster Graph', c3: 'Deep Repetition Trend', c4: 'Emotional Resonance Freq',
    week: ['D-6', 'D-5', 'D-4', 'D-3', 'D-2', 'Yest.', 'Today'], sent: ['Dopamine (Pos)', 'Objective (Neu)', 'Anxiety (Neg)'],
    cats: { global: 'Global', ent: 'Entertainment', edu: 'Learning', news: 'News', soc: 'Social' },
    eval: {
      entScore: 'C- Alert', entMsg: `Your intake in <b style="color:#ff00ea; font-family:{f}">Entertainment</b> is highly repetitive. Algorithms are feeding you homogenized dopamine. Break the loop!`,
      eduScore: 'A+ Excellent', eduMsg: `Your <b style="color:#00ffaa; font-family:{f}">Learning</b> graph shows high diversity and low repetition. Keep up this high-quality mental workout!`,
      newsScore: 'B- Mood Swings', newsMsg: `Information density here is normal, but the <b>Negative Emotion Index is high</b>. Consider limiting macro-narrative intake.`,
      socScore: 'B Echo Chamber', socMsg: `Your <b style="color:#ff4d4f; font-family:{f}">Social</b> interaction shows an obvious <b>echo chamber effect</b>. Introduce diverse perspectives.`,
      globScore: 'B+ Sub-healthy', globMsg: `The overall info stream indicates <b style="color:var(--primary-color); font-family:{f}">Entertainment Overload</b>. Click slices on the bottom-left pie chart to drill down.`
    }
  }
}
const t = computed(() => i18n[settings.lang])
i18n.zh.cats.other = '其他'
i18n.zh.eval.otherScore = 'B 中性观察'
i18n.zh.eval.otherMsg = '这部分内容暂未归入核心领域，建议结合来源与主题继续细分，避免 <b>未知/自定义渠道</b> 长期堆积。'
i18n.en.cats.other = 'Other'
i18n.en.eval.otherScore = 'B Mixed Bucket'
i18n.en.eval.otherMsg = 'This bucket is not mapped to a core domain yet. Review those <b>unknown/custom channels</b> and classify them if they keep growing.'

// === 核心数据源 (响应式，等待后端填充) ===
const db = reactive({
  'global': { repData: [0,0,0,0,0,0,0], sentPos: [0,0,0,0,0,0,0], sentNeu: [0,0,0,0,0,0,0], sentNeg: [0,0,0,0,0,0,0], nodes: 10, links: 10, focusMode: 'mixed', pieData: [] },
  'ent': { repData: [], sentPos: [], sentNeu: [], sentNeg: [], nodes: 0, links: 0, focusMode: 'dense' },
  'edu': { repData: [], sentPos: [], sentNeu: [], sentNeg: [], nodes: 0, links: 0, focusMode: 'sparse' },
  'news': { repData: [], sentPos: [], sentNeu: [], sentNeg: [], nodes: 0, links: 0, focusMode: 'mixed' },
  'soc': { repData: [], sentPos: [], sentNeu: [], sentNeg: [], nodes: 0, links: 0, focusMode: 'dense' },
  'other': { repData: [], sentPos: [], sentNeu: [], sentNeg: [], nodes: 0, links: 0, focusMode: 'mixed' }
})

const healthStatus = computed(() => {
  const f = settings.fontFamily; const ev = t.value.eval;
  switch (currentCategoryKey.value) {
    case 'ent': return { score: ev.entScore, color: '#ff00ea', class: 'status-danger', borderClass: 'border-danger', message: ev.entMsg.replace('{f}', f) }
    case 'edu': return { score: ev.eduScore, color: '#00ffaa', class: 'status-safe', borderClass: 'border-safe', message: ev.eduMsg.replace('{f}', f) }
    case 'news': return { score: ev.newsScore, color: '#ffd700', class: 'status-warn', borderClass: 'border-warn', message: ev.newsMsg.replace('{f}', f) }
    case 'soc': return { score: ev.socScore, color: '#ff4d4f', class: 'status-warn', borderClass: 'border-warn', message: ev.socMsg.replace('{f}', f) }
    case 'other': return { score: ev.otherScore, color: '#94a3b8', class: 'status-warn', borderClass: 'border-warn', message: ev.otherMsg.replace('{f}', f) }
    default: return { score: ev.globScore, color: settings.themeColor, class: 'status-normal', borderClass: 'border-normal', message: ev.globMsg.replace('{f}', f) }
  }
})

const customStyles = computed(() => ({
  '--primary-color': settings.themeColor, '--font-family': settings.fontFamily, 
  '--bg-gradient': settings.isLightMode ? 'linear-gradient(135deg, #f0f2f5 0%, #e2e8f0 100%)' : 'radial-gradient(circle at center, #111827 0%, #030712 100%)',
  '--card-bg': settings.isLightMode ? 'rgba(255, 255, 255, 0.85)' : 'rgba(17, 25, 40, 0.65)',
  '--card-border': settings.isLightMode ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.05)',
  '--text-main': settings.isLightMode ? '#1e293b' : '#f8fafc', '--text-muted': settings.isLightMode ? '#64748b' : '#94a3b8',
}))

const pieChartRef = ref(null); const graphChartRef = ref(null); const lineRepetitionRef = ref(null); const lineSentimentRef = ref(null);
let pieChart, graphChart, lineRepChart, lineSentChart;

const getChartUIConfig = () => ({ text: settings.isLightMode ? '#334155' : '#cbd5e1', line: settings.isLightMode ? '#cbd5e1' : '#334155', tooltip: settings.isLightMode ? 'rgba(255,255,255,0.95)' : 'rgba(15,23,42,0.95)', fontFamily: settings.fontFamily })

// === 高级数据适配器：负责将 API 数据清洗并注入前端响应式引擎 ===
const fetchAndInjectData = async () => {
  try {
    isUpdating.value = true
    
    // 1. 请求今日汇总快照 (对应 API 契约 3.2: GET /dashboard/summary)
    const summaryRes = await axios.get(`${API_BASE_URL}/dashboard/summary`)
    const summary = summaryRes.data
    
    // 将后端的 channel_counts 转换为饼图需要的格式
    const counts = summary.channel_counts || {}
    db['global'].pieData = [
      { value: counts.ent || 0, name: t.value.cats.ent, id: 'ent', itemStyle: {color: '#ff00ea'} },
      { value: counts.edu || 0, name: t.value.cats.edu, id: 'edu', itemStyle: {color: '#00ffaa'} },
      { value: counts.news || 0, name: t.value.cats.news, id: 'news', itemStyle: {color: '#ffd700'} },
      { value: counts.soc || 0, name: t.value.cats.soc, id: 'soc', itemStyle: {color: '#ff4d4f'} },
      { value: counts.other || 0, name: t.value.cats.other, id: 'other', itemStyle: {color: '#94a3b8'} }
    ]
    
    // 2. 请求历史趋势数据 (对应 API 契约 3.2: GET /analyze/history?limit=7)
    // 注意：这里由于不确定你们后端的具体返回数组结构，编写了保护性的数据映射
    const historyRes = await axios.get(`${API_BASE_URL}/analyze/history?limit=7`)
    const history = Array.isArray(historyRes.data?.runs) ? historyRes.data.runs : []
    
    // 提取重复率和情绪数组 (假设后端返回的是包含 repeat_ratio 等字段的对象数组)
    // 真实的解析逻辑可能需要根据你队友实际返回的 JSON 结构微调
    const ext = (field, fallback) => history.map(h => h[field] * 100 || fallback).reverse() 
    
    db['global'].repData = ext('repeat_ratio', 50) // 如果没有数据，默认 50%
    db['global'].sentPos = history.map(h => (h.avg_sentiment > 0 ? h.avg_sentiment*100 : 20)).reverse()
    db['global'].sentNeg = history.map(h => (h.negative_ratio * 100 || 20)).reverse()
    db['global'].sentNeu = db['global'].sentPos.map((pos, i) => 100 - pos - db['global'].sentNeg[i])
    
    // 为各子领域生成基于全局数据的浮动伪数据（因为契约中未明确指出提供按天按领域拆分的数据接口）
    // 如果后端有详细接口，这里可以继续替换
    ['ent', 'edu', 'news', 'soc', 'other'].forEach(key => {
        db[key] = {
            ...db[key],
            repData: db['global'].repData.map(v => key === 'ent' ? Math.min(v + 30, 98) : (key === 'edu' ? Math.max(v - 30, 10) : v)),
            sentPos: db['global'].sentPos,
            sentNeg: db['global'].sentNeg,
            sentNeu: db['global'].sentNeu,
            nodes: summary.total_count ? Math.floor(summary.total_count * 0.1) : (key==='ent'? 80 : 30),
            links: summary.total_count ? Math.floor(summary.total_count * 0.15) : (key==='ent'? 120 : 20)
        }
    })

    // 数据注入完毕，渲染图表
    nextTick(() => { updateAllCharts() })

  } catch (error) {
    console.error("API 数据拉取失败，采用降级展示(本地数据):", error)
    // 降级处理：如果没有连上后端，依然展示一套体验数据，防止黑屏
    db['global'].pieData = [
        { value: 65, name: t.value.cats.ent, id: 'ent', itemStyle: {color: '#ff00ea'} }, 
        { value: 15, name: t.value.cats.edu, id: 'edu', itemStyle: {color: '#00ffaa'} }, 
        { value: 10, name: t.value.cats.news, id: 'news', itemStyle: {color: '#ffd700'} }, 
        { value: 10, name: t.value.cats.soc, id: 'soc', itemStyle: {color: '#ff4d4f'} }
    ]
    db['global'].repData = [30, 45, 60, 55, 70, 85, 80]
    nextTick(() => { updateAllCharts() })
  } finally {
    setTimeout(() => { isUpdating.value = false }, 500) // 延迟关闭动画，提升视觉体验
  }
}

const generateGraphData = (key) => {
  const config = db[key] || db['global']
  const nodes = []; const links = [];
  const colorMap = { 'ent': '#ff00ea', 'edu': '#00ffaa', 'news': '#ffd700', 'soc': '#ff4d4f', 'other': '#94a3b8', 'global': settings.themeColor }
  const baseColor = colorMap[key] || settings.themeColor
  const nodeCount = config.nodes || 30
  const linkCount = config.links || 40

  for (let i = 0; i < nodeCount; i++) nodes.push({ id: `${i}`, name: `N-${i}`, symbolSize: Math.random() * (config.focusMode === 'dense' ? 15 : 30) + 5, itemStyle: { color: baseColor, shadowBlur: 10, shadowColor: baseColor } })
  for (let i = 0; i < linkCount; i++) links.push({ source: `${Math.floor(Math.random() * nodeCount)}`, target: `${Math.floor(Math.random() * nodeCount)}`, lineStyle: { width: Math.random() * 2 } })
  return { nodes, links }
}

const updateAllCharts = () => {
  if (!pieChartRef.value) return
  if (!pieChart) {
    pieChart = echarts.init(pieChartRef.value); graphChart = echarts.init(graphChartRef.value);
    lineRepChart = echarts.init(lineRepetitionRef.value); lineSentChart = echarts.init(lineSentimentRef.value);
    
    pieChart.on('click', (params) => {
      const clickedKey = params.data.id
      if (currentCategoryKey.value === clickedKey) return
      currentCategoryKey.value = clickedKey
      isUpdating.value = true 
      setTimeout(() => { updateDynamicCharts(clickedKey); isUpdating.value = false }, 800)
    })
  }

  const ui = getChartUIConfig(); const loc = t.value
  
  pieChart.setOption({
    textStyle: { fontFamily: ui.fontFamily },
    tooltip: { trigger: 'item', backgroundColor: ui.tooltip, textStyle: { color: ui.text, fontFamily: ui.fontFamily }, borderWidth: 0 },
    legend: { bottom: '0%', textStyle: { color: ui.text, fontFamily: ui.fontFamily } },
    series: [{
      type: 'pie', radius: ['45%', '75%'], center: ['50%', '45%'],
      itemStyle: { borderRadius: 8, borderColor: settings.isLightMode ? '#fff' : '#0f172a', borderWidth: 3 },
      label: { show: false, fontFamily: ui.fontFamily },
      emphasis: { scaleSize: 10, itemStyle: { shadowBlur: 20, shadowColor: 'rgba(0,0,0,0.5)' } },
      data: db['global'].pieData // 使用从后端拉取的饼图数据
    }]
  })
  updateDynamicCharts(currentCategoryKey.value)
}

const updateDynamicCharts = (key) => {
  const ui = getChartUIConfig(); const data = db[key] || db['global']; const loc = t.value
  
  const graphData = generateGraphData(key)
  graphChart.setOption({
    textStyle: { fontFamily: ui.fontFamily }, tooltip: { show: false },
    series: [{
      type: 'graph', layout: 'force', data: graphData.nodes, links: graphData.links,
      roam: true, draggable: true, label: { fontFamily: ui.fontFamily },
      force: { repulsion: data.focusMode === 'dense' ? 50 : 200, edgeLength: data.focusMode === 'dense' ? 20 : 80 },
      lineStyle: { color: settings.isLightMode ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.2)', curveness: 0.3 }
    }]
  }, true)

  lineRepChart.setOption({
    textStyle: { fontFamily: ui.fontFamily },
    tooltip: { trigger: 'axis', backgroundColor: ui.tooltip, textStyle: { color: ui.text, fontFamily: ui.fontFamily }, borderWidth: 0 },
    grid: { left: '8%', right: '5%', bottom: '15%', top: '15%' },
    xAxis: { type: 'category', data: loc.week, axisLine: { lineStyle: { color: ui.line } }, axisLabel: { color: ui.text, fontFamily: ui.fontFamily } },
    yAxis: { type: 'value', max: 100, splitLine: { lineStyle: { color: ui.line, type: 'dashed' } }, axisLabel: { color: ui.text, fontFamily: ui.fontFamily } },
    series: [{
      data: data.repData, type: 'line', smooth: true,
      lineStyle: { color: settings.themeColor, width: 4, shadowColor: settings.themeColor, shadowBlur: 15 },
      itemStyle: { color: settings.themeColor, borderWidth: 2, borderColor: '#fff' },
      areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: settings.themeColor + '90' }, { offset: 1, color: 'transparent' }]) }
    }]
  }, true)

  lineSentChart.setOption({
    textStyle: { fontFamily: ui.fontFamily },
    tooltip: { trigger: 'axis', backgroundColor: ui.tooltip, textStyle: { color: ui.text, fontFamily: ui.fontFamily }, borderWidth: 0 },
    legend: { top: '0%', textStyle: { color: ui.text, fontFamily: ui.fontFamily }, icon: 'circle' },
    grid: { left: '8%', right: '5%', bottom: '15%', top: '20%' },
    xAxis: { type: 'category', data: loc.week, axisLine: { lineStyle: { color: ui.line } }, axisLabel: { color: ui.text, fontFamily: ui.fontFamily } },
    yAxis: { type: 'value', splitLine: { lineStyle: { color: ui.line, type: 'dashed' } }, axisLabel: { color: ui.text, fontFamily: ui.fontFamily } },
    series: [
      { name: loc.sent[0], type: 'line', smooth: true, data: data.sentPos, itemStyle: { color: '#00ffaa' }, lineStyle: { width: 3 }, showSymbol: false },
      { name: loc.sent[1], type: 'line', smooth: true, data: data.sentNeu, itemStyle: { color: ui.text }, lineStyle: { width: 2, type: 'dotted' }, showSymbol: false },
      { name: loc.sent[2], type: 'line', smooth: true, data: data.sentNeg, itemStyle: { color: '#ff4d4f' }, lineStyle: { width: 3 }, showSymbol: false }
    ]
  }, true)
}

watch(
  () => [settings.isLightMode, settings.themeColor, settings.fontFamily, settings.lang], 
  () => { nextTick(() => updateAllCharts()) },
  { deep: true }
)

const handleResize = () => { pieChart?.resize(); graphChart?.resize(); lineRepChart?.resize(); lineSentChart?.resize() }

// === 核心生命周期：挂载时触发网络请求 ===
onMounted(() => { 
  fetchAndInjectData() // 发起真实数据请求
  window.addEventListener('resize', handleResize) 
})

onUnmounted(() => { window.removeEventListener('resize', handleResize); pieChart?.dispose(); graphChart?.dispose(); lineRepChart?.dispose(); lineSentChart?.dispose() })
</script>

<style scoped>
/* =========== 核心框架与字体注入 =========== */
.dashboard-container { min-height: 100vh; background: var(--bg-gradient); color: var(--text-main); font-family: var(--font-family) !important; padding: 2rem; position: relative; transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1); overflow-x: hidden; }
h1, h2, h3, p, span, div, button, input, select { font-family: inherit; }

/* =========== 全局 Loading =========== */
.global-loading { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(10px); z-index: 999; display: flex; flex-direction: column; justify-content: center; align-items: center; }
.light-mode .global-loading { background: rgba(255, 255, 255, 0.7); }
.cyber-spinner { width: 60px; height: 60px; border: 4px solid transparent; border-top-color: var(--primary-color); border-bottom-color: var(--primary-color); border-radius: 50%; animation: spin 1s linear infinite; position: relative; }
.cyber-spinner::before { content: ''; position: absolute; top: 10px; left: 10px; right: 10px; bottom: 10px; border: 4px solid transparent; border-left-color: #ff00ea; border-right-color: #ff00ea; border-radius: 50%; animation: spin-reverse 0.5s linear infinite; }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
@keyframes spin-reverse { 0% { transform: rotate(360deg); } 100% { transform: rotate(0deg); } }
.loading-text { margin-top: 20px; font-weight: bold; color: var(--primary-color); letter-spacing: 2px; text-transform: uppercase; }

/* =========== 头部防重叠布局 =========== */
.tech-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; }
.glow-text { font-size: 2.2rem; margin: 0; font-weight: 900; background: linear-gradient(to right, var(--primary-color), #ff00ea); -webkit-background-clip: text; color: transparent; text-shadow: 0 4px 20px rgba(0,0,0,0.1); }
.version-tag { font-size: 1rem; color: #fff; background: linear-gradient(45deg, #ff00ea, var(--primary-color)); padding: 2px 8px; border-radius: 4px; vertical-align: middle; margin-left: 10px; }
.header-actions { display: flex; align-items: center; gap: 1.5rem; }

/* =========== 卡片与样式 =========== */
.tech-card { background: var(--card-bg); backdrop-filter: blur(20px); border: 1px solid var(--card-border); border-radius: 16px; padding: 1.5rem; box-shadow: 0 10px 30px rgba(0,0,0,0.1); position: relative; transition: all 0.3s ease; }
.tech-card:hover { transform: translateY(-3px); box-shadow: 0 15px 40px rgba(0,0,0,0.15); border-color: rgba(255,255,255,0.1); }
.interactive-card { border: 1px dashed var(--primary-color); cursor: pointer; }
.interactive-card:hover { border: 1px solid var(--primary-color); background: rgba(0, 243, 255, 0.05); }
.hint-text { font-size: 0.8rem; color: #888; font-weight: normal; margin-left: 10px; animation: blink 2s infinite; }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

.summary-card { margin-bottom: 2rem; display: flex; align-items: center; gap: 2rem; border-left: 6px solid var(--primary-color); }
.border-danger { border-left-color: #ff4d4f !important; } .border-safe { border-left-color: #00ffaa !important; } .border-warn { border-left-color: #ffd700 !important; }
.icon-pulse { width: 24px; height: 24px; border-radius: 50%; animation: pulse 2s infinite cubic-bezier(0.4, 0, 0.2, 1); }
@keyframes pulse { 0% { transform: scale(0.9); box-shadow: 0 0 0 0 inherit; } 70% { transform: scale(1.1); box-shadow: 0 0 0 15px transparent; } 100% { transform: scale(0.9); box-shadow: 0 0 0 0 transparent; } }
.summary-content h2 { margin: 0 0 0.5rem 0; font-size: 1.3rem; color: var(--text-main); }
.summary-text { margin: 0; font-size: 1.1rem; line-height: 1.6; color: var(--text-muted); }

.dashboard-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 2rem; }
.card-header { display: flex; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid var(--card-border); padding-bottom: 0.8rem; }
.dot { width: 8px; height: 8px; background: var(--primary-color); border-radius: 50%; margin-right: 12px; }
.card-header h2 { font-size: 1.1rem; margin: 0; color: var(--text-main); font-weight: bold; }
.chart-container { width: 100%; height: 320px; }

/* 状态标签 */
.status-badge { padding: 0.6rem 1.2rem; border-radius: 50px; font-size: 0.95rem; font-weight: bold; border: 1px solid transparent; }
.status-normal { background: rgba(0, 243, 255, 0.1); border-color: var(--primary-color); color: var(--primary-color); }
.status-danger { background: rgba(255, 77, 79, 0.1); border-color: #ff4d4f; color: #ff4d4f; box-shadow: 0 0 15px rgba(255, 77, 79, 0.3); animation: glitch-border 1.5s infinite;}
.status-safe { background: rgba(0, 255, 170, 0.1); border-color: #00ffaa; color: #00ffaa; }
.status-warn { background: rgba(255, 215, 0, 0.1); border-color: #ffd700; color: #ffd700; }
@keyframes glitch-border { 0%, 100% { box-shadow: 0 0 10px rgba(255, 77, 79, 0.2) inset; } 50% { box-shadow: 0 0 25px rgba(255, 77, 79, 0.8) inset; } }

/* =========== 设置面板控制区 =========== */
.settings-wrapper { position: relative; z-index: 100; }
.icon-btn { background: var(--card-bg); border: 1px solid var(--primary-color); color: var(--primary-color); padding: 0.6rem; border-radius: 50%; cursor: pointer; transition: all 0.3s ease; display: flex; align-items: center; justify-content: center; }
.icon-btn:hover { transform: rotate(180deg) scale(1.1); box-shadow: 0 0 15px var(--primary-color); }

.settings-panel { position: absolute; top: calc(100% + 15px); right: 0; width: 320px; padding: 1.5rem; z-index: 101; transform-origin: top right; }
.settings-panel h3 { margin-top: 0; color: var(--primary-color); border-bottom: 1px solid var(--card-border); padding-bottom: 0.5rem; }
.setting-item { margin-bottom: 1.2rem; display: flex; flex-direction: column; gap: 0.5rem; }
.setting-item label { font-size: 0.9rem; color: var(--text-muted); font-weight: bold; }
.setting-item input[type="color"] { width: 100%; height: 40px; border: none; border-radius: 8px; cursor: pointer; background: none; }
.setting-item select { width: 100%; padding: 0.6rem; background: var(--card-bg); color: var(--text-main); border: 1px solid var(--card-border); border-radius: 6px; outline: none; cursor: pointer; font-family: inherit; }
.theme-switch { position: relative; display: flex; align-items: center; justify-content: space-between; background: rgba(0,0,0,0.1); border-radius: 30px; padding: 4px; cursor: pointer; border: 1px solid var(--card-border); height: 38px; }
.switch-bg { position: absolute; width: 50%; height: calc(100% - 8px); background: var(--primary-color); border-radius: 30px; transition: transform 0.3s ease; z-index: 1; left: 4px; }
.switch-bg.is-light { transform: translateX(100%); }
.switch-label { flex: 1; text-align: center; font-size: 0.85rem; z-index: 2; color: var(--text-muted); transition: color 0.3s; }
.switch-label.active { color: #fff; font-weight: bold; }
</style>

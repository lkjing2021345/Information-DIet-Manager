<template>
  <div class="dashboard-container" :style="customStyles" :class="{'light-mode': settings.isLightMode}">
    
    <div class="global-loading" v-if="isUpdating">
      <div class="cyber-spinner"></div>
      <div class="loading-text" :style="{ fontFamily: settings.fontFamily }">
        {{ t.loading.replace('{cat}', getCategoryLabel(currentCategoryKey)) }}
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
      <div class="summary-content" style="flex: 1;">
        
        <div class="card-title-row">
          <h2>{{ t.report }}: {{ getCategoryLabel(currentCategoryKey) }} {{ t.domain }} <span class="hint-text">{{ t.hint }}</span></h2>

          <div class="card-actions">
            <button class="force-refresh-btn" 
                    :class="{ 'is-spinning': isForceRefreshing }" 
                    :style="{ color: healthStatus.color, borderColor: healthStatus.color }" 
                    @click="triggerGlobalForceRefresh" 
                    :disabled="isForceRefreshing"
                    title="跳过后端缓存，强制重新跑一遍模型推演">
              <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none" :class="{ 'spin-anim': isForceRefreshing }"><polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>
              {{ isForceRefreshing ? t.refreshing : t.forceRefreshBtn }}
            </button>
            
            <button class="trace-btn" :style="{ color: healthStatus.color, borderColor: healthStatus.color, backgroundColor: `${healthStatus.color}15` }" @click="openDrawer">
              <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
              {{ t.traceBtn }}
            </button>
          </div>
        </div>

        <p class="summary-text" v-html="healthStatus.message"></p>
      </div>
    </div>

    <main class="dashboard-grid">
      <div class="tech-card interactive-card">
        <div class="card-header"><span class="dot"></span><h2>{{ t.c1 }}</h2></div>
        <div ref="pieChartRef" class="chart-container"></div>
      </div>
      <div class="tech-card">
        <div class="card-header">
          <span class="dot"></span>
          <h2>{{ getCategoryLabel(currentCategoryKey) }} - {{ t.c2 }} <span class="mock-hint">{{ t.mockHint }}</span></h2>
        </div>
        <div ref="graphChartRef" class="chart-container"></div>
      </div>
      <div class="tech-card">
        <div class="card-header"><span class="dot"></span><h2>{{ getCategoryLabel(currentCategoryKey) }} - {{ t.c3 }}</h2></div>
        <div ref="lineRepetitionRef" class="chart-container"></div>
      </div>
      <div class="tech-card">
        <div class="card-header"><span class="dot"></span><h2>{{ getCategoryLabel(currentCategoryKey) }} - {{ t.c4 }}</h2></div>
        <div ref="lineSentimentRef" class="chart-container"></div>
      </div>
    </main>

    <transition name="fade">
      <div class="drawer-overlay" v-if="showDrawer" @click="showDrawer = false"></div>
    </transition>
    
    <transition name="slide">
      <aside class="drawer-panel tech-card" v-if="showDrawer" :style="{ borderLeftColor: healthStatus.color, boxShadow: `-10px 0 40px ${healthStatus.color}30` }">
        
        <div class="drawer-header">
          <h3 :style="{ color: healthStatus.color }">{{ getCategoryLabel(currentCategoryKey) }} {{ t.drawerTitle }}</h3>
          <button class="icon-btn close-btn" @click="showDrawer = false" title="关闭">✕</button>
        </div>

        <div class="drawer-loading" v-if="currentDrawerData.isLoading">
          <div class="cyber-spinner small-spinner"></div>
          <p>正在穿透获取底层真实迹线...</p>
        </div>

        <div class="drawer-content" v-else-if="currentDrawerData.hasData">
          <div class="drawer-section">
            <h4 class="section-title">🧬 {{ t.kwTitle }}</h4>
            <div class="tags-container">
              <span class="cyber-tag" v-for="(tag, idx) in currentDrawerData.keywords" :key="idx" :style="{ color: healthStatus.color, borderColor: healthStatus.color, backgroundColor: `${healthStatus.color}15` }">
                # {{ tag }}
              </span>
            </div>
          </div>

          <div class="drawer-section">
            <h4 class="section-title">💊 {{ t.aiTitle }}</h4>
            <div class="rx-box" :style="{ borderLeftColor: healthStatus.color, backgroundColor: `${healthStatus.color}0A` }">
              {{ currentDrawerData.prescription }}
            </div>
          </div>

          <div class="drawer-section timeline-section">
            <h4 class="section-title">⏱️ {{ t.tlTitle }}</h4>
            <ul class="cyber-timeline">
              <li v-for="(item, idx) in currentDrawerData.timeline" :key="idx">
                <div class="timeline-dot" :class="`dot-${item.sentiment}`"></div>
                <div class="timeline-content">
                  
                  <div class="timeline-meta">
                    <span class="time">{{ item.time }}</span>
                    
                    <span class="repeat-badge" v-if="item.visits > 1" title="算法茧房循环重复标记">
                      <svg viewBox="0 0 24 24" width="12" height="12" stroke="currentColor" stroke-width="2" fill="none"><polyline points="1 4 1 10 7 10"></polyline><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path></svg>
                      {{ settings.lang === 'zh' ? `第 ${item.visits} 次重复` : `${item.visits}x Reps` }}
                    </span>

                    <span class="cached-status" :class="item.isCached ? 'cached-yes' : 'cached-no'">
                      Cached: {{ item.isCached ? '是' : '否' }}
                    </span>

                    <button class="force-btn" v-if="item.isCached" @click.stop="triggerForceRun(item)" :disabled="item.isForceRunning" title="跳过缓存，强制重新抓取分析">
                      <span v-if="!item.isForceRunning">⚡ 强制分析</span>
                      <span v-else>↻ 运行中...</span>
                    </button>
                  </div>
                  
                  <a :href="item.url" target="_blank" class="title">{{ item.title }}</a>
                  
                  <div class="timeline-tags-row">
                    <span class="source-tag">{{ item.source }}</span>
                    <span class="sentiment-pill" :class="`bg-${item.sentiment}`">
                      {{ getSentimentLabel(item.sentiment) }}
                    </span>
                  </div>
                </div>
              </li>
            </ul>
          </div>
        </div>

        <div class="drawer-empty" v-else>
          <svg viewBox="0 0 24 24" width="64" height="64" stroke="var(--text-muted)" stroke-width="1" fill="none"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line><line x1="11" y1="8" x2="11" y2="14"></line><line x1="8" y1="11" x2="14" y2="11"></line></svg>
          <p>当前领域暂无底层浏览迹线</p>
          <span class="empty-hint">请确保插件已上报数据且 Pipeline 已处理完毕。</span>
        </div>

      </aside>
    </transition>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import * as echarts from 'echarts'
import axios from 'axios'

const showSettings = ref(false)
const toggleSettings = () => showSettings.value = !showSettings.value
const isUpdating = ref(true)
const currentCategoryKey = ref('global')
const showDrawer = ref(false)
const isForceRefreshing = ref(false)
let pollingTimer = null; 

const settings = reactive({
  lang: 'zh',
  themeColor: '#00f3ff',
  fontFamily: "'Segoe UI', 'Microsoft YaHei', sans-serif",
  isLightMode: false
})

const API_BASE_URL = 'http://127.0.0.1:8000'

const CATEGORY_COLOR_MAP = {
  ent: '#ff00ea',
  edu: '#00ffaa',
  news: '#ffd700',
  soc: '#ff4d4f',
  other: '#94a3b8'
}

const CATEGORY_ALIAS_FALLBACK_MAP = {
  ent: 'ent',
  entertainment: 'ent',
  video: 'ent',
  edu: 'edu',
  learning: 'edu',
  tools: 'edu',
  news: 'news',
  soc: 'soc',
  social: 'soc',
  other: 'other',
  shopping: 'other',
  web: 'other',
  unknown: 'other',
  '娱乐': 'ent',
  '学习': 'edu',
  '新闻': 'news',
  '社交': 'soc',
  '其他': 'other'
}

const i18n = {
  zh: {
    sysUi: '系统级 UI 定制', langLabel: '🌐 界面语言 (Language)', visMode: '视觉引擎模式',
    dark: '🌙 极客暗黑', light: '☀️ 护眼明亮', color: '神经突触高亮色', font: '全局渲染字体族',
    health: '生态健康度', report: '深度剖析报告', domain: '领域', hint: '(💡 点击左侧饼图交互)', loading: '正在通过神经网络链接服务器获取 [{cat}] 数据...',
    c1: '摄入结构特征 (点击切片下钻)', c2: '信息群落知识图谱', c3: '深度复读率走势', c4: '情感共振频段分析', mockHint: '(基于真实特征数据动态拓扑渲染)',
    traceBtn: '深度溯源', forceRefreshBtn: '⚡ 强制大盘推演', refreshing: '推演中...',
    drawerTitle: '领域底层数据透视', kwTitle: '高频特征提取', aiTitle: 'AI 干预处方', tlTitle: '原始浏览行为迹线',
    week: ['D-6', 'D-5', 'D-4', 'D-3', 'D-2', '昨日', '今日'], sent: ['多巴胺(正向)', '客观(中性)', '焦虑(负向)'],
    noData: '暂无真实趋势数据',
    cats: { global: '全局概览', ent: '娱乐', edu: '学习', news: '新闻', soc: '社交' },
    eval: {
      entScore: 'C- 茧房警报', entMsg: `您在 <b style="color:#ff00ea; font-family:{f}">娱乐</b> 领域的摄入处于<b>高频复读状态</b>。算法正向您投喂同质化多巴胺内容，强烈建议跳出舒适区！`,
      eduScore: 'A+ 极佳', eduMsg: `您在 <b style="color:#00ffaa; font-family:{f}">学习</b> 领域的信息图谱展现出极高的<b>多样性与低重复度</b>。继续保持这种高质量的脑力体操！`,
      newsScore: 'B- 情绪波动', newsMsg: `该领域的信息密度正常，但<b>负面情绪指数较高</b>。建议适当控制宏观叙事信息的摄入，关注真实生活。`,
      socScore: 'B 圈层固化', socMsg: `您在 <b style="color:#ff4d4f; font-family:{f}">社交</b> 领域的信息交互呈现明显的<b>信息同温层效应 (回音室)</b>。建议引入不同视角的观点。`,
      globScore: 'B+ 亚健康', 
      globMsg: `当前整体信息流呈现 <b style="color:{color}; font-family:{f}">{maxCat}内容占据主导</b> 倾向。点击左下方饼图的各个切片，深度下钻您的信息群落。`,
      emptyMsg: `当前系统暂无真实的浏览记录。请先通过后端接口或插件录入您的信息饮食数据。`
    }
  },
  en: {
    sysUi: 'System UI Engine', langLabel: '🌐 Interface Language', visMode: 'Visual Mode',
    dark: '🌙 Cyber Dark', light: '☀️ Clean Light', color: 'Synapse Highlight Color', font: 'Global Font Family',
    health: 'Eco-Health', report: 'In-Depth Report', domain: 'Domain', hint: '(💡 Click pie chart to drill down)', loading: 'Connecting server to fetch [{cat}] models...',
    c1: 'Intake Structure (Interactive)', c2: 'Information Cluster Graph', c3: 'Deep Repetition Trend', c4: 'Emotional Resonance Freq', mockHint: '(Data-driven Topology)',
    traceBtn: 'Deep Trace', forceRefreshBtn: '⚡ Force Analyze', refreshing: 'Running...',
    drawerTitle: 'Raw Data Penetration', kwTitle: 'High-Freq Features', aiTitle: 'AI Intervention Rx', tlTitle: 'Raw Browsing Timeline',
    week: ['D-6', 'D-5', 'D-4', 'D-3', 'D-2', 'Yest.', 'Today'], sent: ['Dopamine (Pos)', 'Objective (Neu)', 'Anxiety (Neg)'],
    noData: 'No Real Trend Data',
    cats: { global: 'Global', ent: 'Entertainment', edu: 'Learning', news: 'News', soc: 'Social' },
    eval: {
      entScore: 'C- Alert', entMsg: `Your intake in <b style="color:#ff00ea; font-family:{f}">Entertainment</b> is highly repetitive. Algorithms are feeding you homogenized dopamine. Break the loop!`,
      eduScore: 'A+ Excellent', eduMsg: `Your <b style="color:#00ffaa; font-family:{f}">Learning</b> graph shows high diversity and low repetition. Keep up this high-quality mental workout!`,
      newsScore: 'B- Mood Swings', newsMsg: `Information density here is normal, but the <b>Negative Emotion Index is high</b>. Consider limiting macro-narrative intake.`,
      socScore: 'B Echo Chamber', socMsg: `Your <b style="color:#ff4d4f; font-family:{f}">Social</b> interaction shows an obvious <b>echo chamber effect</b>. Introduce diverse perspectives.`,
      globScore: 'B+ Sub-healthy', 
      globMsg: `The overall info stream is dominated by <b style="color:{color}; font-family:{f}">{maxCat}</b>. Click slices on the pie chart to drill down.`,
      emptyMsg: `No actual browsing data found. Please ingest your data first to trigger the analysis pipeline.`
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

const getCategoryLabel = (key) => t.value.cats[key] || t.value.cats.other || 'Other'

const getSentimentLabel = (sent) => {
  if (settings.lang === 'en') return sent === 'pos' ? '😍 Dopamine' : sent === 'neg' ? '😰 Anxiety' : '😐 Neutral';
  return sent === 'pos' ? '😍 多巴胺' : sent === 'neg' ? '😰 焦虑感' : '😐 客观纪实';
}

const normalizeCategoryAlias = (value) => {
  const key = String(value || '').trim().toLowerCase()
  return CATEGORY_ALIAS_FALLBACK_MAP[key] || 'other'
}

const buildPieDataFromVisualization = (visData) => {
  const totals = { ent: 0, edu: 0, news: 0, soc: 0, other: 0 }
  const categories = visData?.categories || {}

  Object.values(categories).forEach((cat) => {
    const alias = normalizeCategoryAlias(cat?.alias || cat?.label)
    const series = Array.isArray(cat?.time_series) ? cat.time_series : []
    const count = series.reduce((sum, item) => sum + Number(item?.count || 0), 0)
    totals[alias] = (totals[alias] || 0) + count
  })

  return [
    { value: totals.ent, name: t.value.cats.ent, id: 'ent', itemStyle: { color: CATEGORY_COLOR_MAP.ent } },
    { value: totals.edu, name: t.value.cats.edu, id: 'edu', itemStyle: { color: CATEGORY_COLOR_MAP.edu } },
    { value: totals.news, name: t.value.cats.news, id: 'news', itemStyle: { color: CATEGORY_COLOR_MAP.news } },
    { value: totals.soc, name: t.value.cats.soc, id: 'soc', itemStyle: { color: CATEGORY_COLOR_MAP.soc } },
    { value: totals.other, name: t.value.cats.other, id: 'other', itemStyle: { color: CATEGORY_COLOR_MAP.other } }
  ]
}

const currentDrawerData = reactive({
  isLoading: false,
  hasData: false,
  keywords: [],      
  prescription: '',  
  timeline: []       
})

const loadDrawerData = async (silent = false, force = false) => {
  if (!silent) {
    currentDrawerData.isLoading = true;
    currentDrawerData.hasData = false;
    currentDrawerData.timeline = [];
    currentDrawerData.keywords = [];
    currentDrawerData.prescription = '';
  }

  try {
    const query = force ? '?days=7&limit_rows=50&force=1' : '?days=7&limit_rows=50'
    const res = await axios.get(`${API_BASE_URL}/items/analyzed${query}`)
    const payload = res.data && typeof res.data === 'object' ? res.data : {}
    let records = Array.isArray(payload.items) ? payload.items : []

    if (!Array.isArray(records) || records.length === 0) {
      console.warn("⚠️ /items/analyzed 接口返回为空，或者格式无法识别:", res.data);
      records = [];
    }

    const catRecords = currentCategoryKey.value === 'global'
      ? records 
      : records.filter(r => {
          const derivedAlias = normalizeCategoryAlias(r.alias || r.category || r.channel)
          return derivedAlias === currentCategoryKey.value
        });

    if (catRecords && catRecords.length > 0) {
      currentDrawerData.hasData = true
      
      currentDrawerData.timeline = catRecords.map(r => {
        const dateObj = new Date(r.ts || r.created_at || Date.now());
        const timeStr = isNaN(dateObj.getTime()) ? '未知' : dateObj.toLocaleTimeString([], {hour: '2-digit', minute:'2-digit'});

        const rawSentiment = String(r.sentiment || '').toLowerCase()
        let sentimentKey = 'neu';
        if (rawSentiment === 'positive' || rawSentiment === 'pos' || Number(r.polarity || 0) > 0.1) sentimentKey = 'pos';
        if (rawSentiment === 'negative' || rawSentiment === 'neg' || Number(r.polarity || 0) < -0.1) sentimentKey = 'neg';

        const cachedFlag = r.is_cached !== undefined ? r.is_cached : (r.cached || false);

        return {
          id: r.id || null, 
          time: timeStr, 
          title: r.title || '未知网页', 
          url: r.url || '#', 
          source: r.source || 'plugin', 
          sentiment: sentimentKey, 
          visits: r.repeat_count || r.visits || 1, 
          isCached: cachedFlag, 
          isForceRunning: false 
        }
      })

      let extractedTags = [];
      catRecords.forEach(r => {
        if (Array.isArray(r.tags)) extractedTags.push(...r.tags);
        if (Array.isArray(r.keywords)) extractedTags.push(...r.keywords);
        if (typeof r.category === 'string' && r.category.trim()) extractedTags.push(r.category.trim());
      });
      extractedTags = [...new Set(extractedTags)].filter(Boolean).slice(0, 8);
      if (extractedTags.length === 0) extractedTags = [getCategoryLabel(currentCategoryKey.value) || '探索中', '近期浏览'];
      currentDrawerData.keywords = extractedTags;

      const repeatCount = currentDrawerData.timeline.filter(i => i.visits > 1).length;
      const negCount = currentDrawerData.timeline.filter(i => i.sentiment === 'neg').length;
      
      if (repeatCount > catRecords.length * 0.4) {
        currentDrawerData.prescription = settings.lang === 'zh' 
          ? `⚠️ 极高同质化预警：在最近记录中发现 ${repeatCount} 次重复。建议立刻关闭当前标签页，起立活动。` 
          : `⚠️ High Repetition: ${repeatCount} repeated visits. Break the loop immediately.`;
      } else if (negCount > catRecords.length * 0.4) {
        currentDrawerData.prescription = settings.lang === 'zh' 
          ? `📰 情绪波动警报：当前信息流负面情绪偏高。请警惕“末日刷屏”现象，保护心理健康。` 
          : `📰 Mood Alert: High negative sentiment detected. Beware of doomscrolling.`;
      } else {
        currentDrawerData.prescription = settings.lang === 'zh' 
          ? `📊 状态评估：信息摄入结构相对健康，请继续保持良好的数字饮食习惯。` 
          : `📊 Status: Information diet is currently healthy.`;
      }
    } else {
      if (!silent) currentDrawerData.hasData = false
    }
  } catch (error) {
    console.error("无法获取抽屉数据:", error)
    if (!silent) currentDrawerData.hasData = false
  } finally {
    if (!silent) currentDrawerData.isLoading = false
  }
}

const openDrawer = () => {
  showDrawer.value = true;
  loadDrawerData(false, false);
}

const triggerForceRun = async (item) => {
  if (item.isForceRunning) return;
  item.isForceRunning = true; 
  
  try {
    await axios.post(`${API_BASE_URL}/analyze/run?force=true`);
    await loadDrawerData(true, true);
    await fetchAndInjectData(true, true);
  } catch (error) {
    console.error("Force run failed:", error);
    alert(settings.lang === 'zh' ? '❌ 强制运行失败，请检查后端服务。' : '❌ Force run failed.');
  } finally {
    item.isForceRunning = false;
  }
}

const db = reactive({
  'global': { repData: [0,0,0,0,0,0,0], sentPos: [0,0,0,0,0,0,0], sentNeu: [0,0,0,0,0,0,0], sentNeg: [0,0,0,0,0,0,0], nodes: 10, links: 10, focusMode: 'mixed', pieData: [], hasData: false },
  'ent': { repData: [], sentPos: [], sentNeu: [], sentNeg: [], nodes: 0, links: 0, focusMode: 'dense', hasData: false },
  'edu': { repData: [], sentPos: [], sentNeu: [], sentNeg: [], nodes: 0, links: 0, focusMode: 'sparse', hasData: false },
  'news': { repData: [], sentPos: [], sentNeu: [], sentNeg: [], nodes: 0, links: 0, focusMode: 'mixed', hasData: false },
  'soc': { repData: [], sentPos: [], sentNeu: [], sentNeg: [], nodes: 0, links: 0, focusMode: 'dense', hasData: false },
  'other': { repData: [], sentPos: [], sentNeu: [], sentNeg: [], nodes: 0, links: 0, focusMode: 'mixed', hasData: false }
})

const healthStatus = computed(() => {
  const f = settings.fontFamily; const ev = t.value.eval;
  const totalPieValue = db['global'].pieData.reduce((sum, item) => sum + item.value, 0);

  if (totalPieValue === 0) return { score: 'N/A 待诊断', color: settings.isLightMode ? '#94a3b8' : '#64748b', class: 'status-normal', borderClass: 'border-normal', message: ev.emptyMsg }

  switch (currentCategoryKey.value) {
    case 'ent': return { score: ev.entScore, color: '#ff00ea', class: 'status-danger', borderClass: 'border-danger', message: ev.entMsg.replace('{f}', f) }
    case 'edu': return { score: ev.eduScore, color: '#00ffaa', class: 'status-safe', borderClass: 'border-safe', message: ev.eduMsg.replace('{f}', f) }
    case 'news': return { score: ev.newsScore, color: '#ffd700', class: 'status-warn', borderClass: 'border-warn', message: ev.newsMsg.replace('{f}', f) }
    case 'soc': return { score: ev.socScore, color: '#ff4d4f', class: 'status-warn', borderClass: 'border-warn', message: ev.socMsg.replace('{f}', f) }
    case 'other':
      return {
        score: ev.otherScore,
        color: '#94a3b8',
        class: 'status-warn',
        borderClass: 'border-warn',
        message: ev.otherMsg.replace('{f}', f)
      }

    default: {
      const maxItem = db['global'].pieData.reduce(
        (prev, current) => (prev.value > current.value) ? prev : current,
        { value: -1 }
      )
      return {
        score: ev.globScore,
        color: settings.themeColor,
        class: 'status-normal',
        borderClass: 'border-normal',
        message: ev.globMsg
          .replace('{f}', f)
          .replace('{color}', maxItem.itemStyle?.color || settings.themeColor)
          .replace('{maxCat}', maxItem.name)
      }
    }
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

const fetchAndInjectData = async (silent = false, force = false) => {
  try {
    if (!silent && !force) isUpdating.value = true;

    const visUrl = force
      ? `${API_BASE_URL}/dashboard/visualization?days=7&force=1` 
      : `${API_BASE_URL}/dashboard/visualization?days=7`;
      
    const visRes = await axios.get(visUrl)
    const visData = visRes.data
    const hasWarning = !!visData.pipeline_warning

    db['global'].pieData = buildPieDataFromVisualization(visData)

    ;['global', 'ent', 'edu', 'news', 'soc', 'other'].forEach((key) => {
      db[key].repData = []
      db[key].sentPos = []
      db[key].sentNeu = []
      db[key].sentNeg = []
      db[key].hasData = false
    })

    const processTimeSeries = (ts) => {
      if (hasWarning || !ts || ts.length === 0) return { repData: [], sentPos: [], sentNeu: [], sentNeg: [], hasData: false }
      return {
        repData: ts.map(h => (h.repeat_ratio || 0) * 100),
        sentPos: ts.map(h => (h.avg_sentiment > 0 ? h.avg_sentiment*100 : 20)),
        sentNeg: ts.map(h => ((h.negative_ratio || 0) * 100 || 20)),
        sentNeu: ts.map(h => { const pos = h.avg_sentiment > 0 ? h.avg_sentiment*100 : 20; const neg = (h.negative_ratio || 0) * 100 || 20; return Math.max(0, 100 - pos - neg); }),
        hasData: true
      }
    }

    Object.assign(db['global'], processTimeSeries(visData.global?.time_series))

    const categories = visData.categories || {}
    Object.values(categories).forEach(cat => {
      const alias = normalizeCategoryAlias(cat.alias || cat.label || cat.name)
      if (alias && db[alias]) Object.assign(db[alias], processTimeSeries(cat.time_series))
    })

    nextTick(() => { updateAllCharts() })
  } catch (error) {
    console.error("API 数据拉取失败:", error)
    ['global', 'ent', 'edu', 'news', 'soc', 'other'].forEach(key => { db[key].hasData = false })
    db['global'].pieData = []
    nextTick(() => { updateAllCharts() })
  } finally {
    if (!silent && !force) setTimeout(() => { isUpdating.value = false }, 500)
  }
}

const triggerGlobalForceRefresh = async () => {
  if (isForceRefreshing.value) return;
  isForceRefreshing.value = true;
  
  try {
    await fetchAndInjectData(true, true); 
    if (showDrawer.value) {
      await loadDrawerData(true, true);
    }
  } catch (error) {
    console.error("大盘强制推演失败:", error);
  } finally {
    isForceRefreshing.value = false;
  }
}

const generateGraphData = (key) => {
  const data = db[key] || db['global'];
  const nodes = []; const links = [];
  if (!data.hasData) return { nodes, links };

  if (key === 'global') {
    nodes.push({ id: 'root', name: '我 (User)', symbolSize: 40, itemStyle: { color: settings.themeColor, shadowBlur: 15, shadowColor: settings.themeColor } });
    db['global'].pieData.filter(p => p.value > 0).forEach(p => {
      const size = Math.min(Math.max(p.value * 2, 15), 50); 
      nodes.push({ id: p.id, name: p.name, symbolSize: size, itemStyle: { color: p.itemStyle.color, shadowBlur: 10, shadowColor: p.itemStyle.color } });
      links.push({ source: 'root', target: p.id, lineStyle: { width: size / 10 } });
    });
  } else {
    const baseColor = db['global'].pieData.find(p => p.id === key)?.itemStyle.color || settings.themeColor;
    nodes.push({ id: 'cat_root', name: getCategoryLabel(key), symbolSize: 50, itemStyle: { color: baseColor, shadowBlur: 20, shadowColor: baseColor } });
    data.repData.forEach((val, idx) => {
      if (val > 0) { 
        const nodeId = `trend_${idx}`;
        nodes.push({ id: nodeId, name: t.value.week[idx], symbolSize: Math.max(10, val / 3), itemStyle: { color: settings.isLightMode ? '#94a3b8' : 'rgba(255,255,255,0.6)' } });
        links.push({ source: 'cat_root', target: nodeId, lineStyle: { width: 1.5 } });
      }
    });
  }
  return { nodes, links };
}

const handleCategorySwitch = (targetKey) => {
  if (currentCategoryKey.value === targetKey) return 
  currentCategoryKey.value = targetKey
  
  isUpdating.value = true 
  
  setTimeout(() => { 
    updateAllCharts(); 
    isUpdating.value = false;
  }, 400)
}

const updateAllCharts = () => {
  if (!pieChartRef.value) return
  if (!pieChart) {
    pieChart = echarts.init(pieChartRef.value); graphChart = echarts.init(graphChartRef.value);
    lineRepChart = echarts.init(lineRepetitionRef.value); lineSentChart = echarts.init(lineSentimentRef.value);
    pieChart.on('click', (params) => {
      if (params.componentType === 'series' && params?.data?.id) handleCategorySwitch(params.data.id)
    })
  }

  const ui = getChartUIConfig(); const isNotGlobal = currentCategoryKey.value !== 'global' 
  
  const hasGlobalData = db['global'].pieData.reduce((sum, item) => sum + (item.value || 0), 0) > 0;
  const noDataGraphic = { type: 'text', left: 'center', top: 'center', style: { text: t.value.noData, fill: ui.text, fontFamily: ui.fontFamily, fontSize: 16, fontWeight: 'bold', opacity: 0.6 } };
  const backBtnGraphic = isNotGlobal ? { type: 'group', left: 'center', top: '40%', cursor: 'pointer', onclick: () => handleCategorySwitch('global'), children: [ { type: 'rect', left: 'center', top: 'center', shape: { r: 16, width: 90, height: 32 }, style: { fill: settings.isLightMode ? 'rgba(0,0,0,0.05)' : 'rgba(255,255,255,0.08)', stroke: settings.themeColor, lineWidth: 1 } }, { type: 'text', left: 'center', top: 'center', style: { text: settings.lang === 'en' ? '↺ Global' : '↺ 返回全局', fill: settings.themeColor, fontFamily: ui.fontFamily, fontSize: 13, fontWeight: 'bold' } } ] } : null;
  
  const finalGraphics = [];
  if (!hasGlobalData) finalGraphics.push(noDataGraphic);
  if (backBtnGraphic) finalGraphics.push(backBtnGraphic);

  pieChart.setOption({
    textStyle: { fontFamily: ui.fontFamily },
    tooltip: { trigger: 'item', backgroundColor: ui.tooltip, textStyle: { color: ui.text, fontFamily: ui.fontFamily }, borderWidth: 0 },
    legend: { show: hasGlobalData, bottom: '0%', textStyle: { color: ui.text, fontFamily: ui.fontFamily } },
    graphic: finalGraphics, 
    series: hasGlobalData ? [{ type: 'pie', radius: ['45%', '75%'], center: ['50%', '45%'], itemStyle: { borderRadius: 8, borderColor: settings.isLightMode ? '#fff' : '#0f172a', borderWidth: 3 }, label: { show: false }, emphasis: { scaleSize: 10, itemStyle: { shadowBlur: 20, shadowColor: 'rgba(0,0,0,0.5)' } }, data: db['global'].pieData }] : []
  }, true) 

  updateDynamicCharts(currentCategoryKey.value)
}

const updateDynamicCharts = (key) => {
  const ui = getChartUIConfig(); 
  const data = db[key] || db['global']; 
  const loc = t.value
  const graphData = data.hasData ? generateGraphData(key) : { nodes: [], links: [] }
  const noDataGraphic = { type: 'text', left: 'center', top: 'center', style: { text: loc.noData, fill: ui.text, fontFamily: ui.fontFamily, fontSize: 16, fontWeight: 'bold', opacity: 0.6 } }

  graphChart.setOption({ 
    textStyle: { fontFamily: ui.fontFamily }, tooltip: { show: false }, 
    graphic: data.hasData ? [] : [noDataGraphic],
    series: [{ type: 'graph', layout: 'force', data: graphData.nodes, links: graphData.links, roam: true, draggable: true, label: { fontFamily: ui.fontFamily }, force: { repulsion: 200, edgeLength: 60 }, lineStyle: { color: settings.isLightMode ? 'rgba(0,0,0,0.1)' : 'rgba(255,255,255,0.2)', curveness: 0.3 } }] 
  }, true)

  lineRepChart.setOption({ 
    textStyle: { fontFamily: ui.fontFamily }, tooltip: { trigger: 'axis', backgroundColor: ui.tooltip, textStyle: { color: ui.text, fontFamily: ui.fontFamily }, borderWidth: 0 }, grid: { left: '8%', right: '5%', bottom: '15%', top: '15%' }, 
    xAxis: { show: data.hasData, type: 'category', data: loc.week, axisLine: { lineStyle: { color: ui.line } }, axisLabel: { color: ui.text, fontFamily: ui.fontFamily } }, yAxis: { show: data.hasData, type: 'value', max: 100, splitLine: { lineStyle: { color: ui.line, type: 'dashed' } }, axisLabel: { color: ui.text, fontFamily: ui.fontFamily } }, 
    graphic: data.hasData ? [] : [noDataGraphic],
    series: data.hasData ? [{ data: data.repData, type: 'line', smooth: true, lineStyle: { color: settings.themeColor, width: 4, shadowColor: settings.themeColor, shadowBlur: 15 }, itemStyle: { color: settings.themeColor, borderWidth: 2, borderColor: '#fff' }, areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: settings.themeColor + '90' }, { offset: 1, color: 'transparent' }]) } }] : [] 
  }, true)

  lineSentChart.setOption({ 
    textStyle: { fontFamily: ui.fontFamily }, tooltip: { trigger: 'axis', backgroundColor: ui.tooltip, textStyle: { color: ui.text, fontFamily: ui.fontFamily }, borderWidth: 0 }, legend: { show: data.hasData, top: '0%', textStyle: { color: ui.text, fontFamily: ui.fontFamily }, icon: 'circle' }, grid: { left: '8%', right: '5%', bottom: '20%', top: '20%' }, 
    xAxis: { show: data.hasData, type: 'category', data: loc.week, axisLine: { lineStyle: { color: ui.line } }, axisLabel: { color: ui.text, fontFamily: ui.fontFamily } }, yAxis: { show: data.hasData, type: 'value', splitLine: { lineStyle: { color: ui.line, type: 'dashed' } }, axisLabel: { color: ui.text, fontFamily: ui.fontFamily } }, 
    graphic: data.hasData ? [] : [noDataGraphic],
    series: data.hasData ? [ 
      { name: loc.sent[0], type: 'line', smooth: true, data: data.sentPos, itemStyle: { color: '#00ffaa' }, lineStyle: { width: 3 }, showSymbol: false }, 
      { name: loc.sent[1], type: 'line', smooth: true, data: data.sentNeu, itemStyle: { color: ui.text }, lineStyle: { width: 2, type: 'dotted' }, showSymbol: false }, 
      { name: loc.sent[2], type: 'line', smooth: true, data: data.sentNeg, itemStyle: { color: '#ff4d4f' }, lineStyle: { width: 3 }, showSymbol: false } 
    ] : [] 
  }, true)
}

watch(() => [settings.isLightMode, settings.themeColor, settings.fontFamily, settings.lang], () => { nextTick(() => updateAllCharts()) }, { deep: true })
const handleResize = () => { pieChart?.resize(); graphChart?.resize(); lineRepChart?.resize(); lineSentChart?.resize() }

onMounted(() => { 
  fetchAndInjectData(); 
  window.addEventListener('resize', handleResize);
  
  pollingTimer = setInterval(() => {
    fetchAndInjectData(true); 
    if (showDrawer.value) {
      loadDrawerData(true, false);
    }
  }, 30000);
})

onUnmounted(() => { 
  window.removeEventListener('resize', handleResize); 
  if (pollingTimer) clearInterval(pollingTimer); 
  pieChart?.dispose(); graphChart?.dispose(); lineRepChart?.dispose(); lineSentChart?.dispose();
})
</script>

<style scoped>
/* =========== 基础架构 UI =========== */
.dashboard-container { min-height: 100vh; background: var(--bg-gradient); color: var(--text-main); font-family: var(--font-family) !important; padding: 2rem; position: relative; transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1); overflow-x: hidden; }
h1, h2, h3, h4, p, span, div, button, input, select, a { font-family: inherit; }

.global-loading { position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; background: rgba(15, 23, 42, 0.8); backdrop-filter: blur(10px); z-index: 999; display: flex; flex-direction: column; justify-content: center; align-items: center; }
.light-mode .global-loading { background: rgba(255, 255, 255, 0.7); }
.cyber-spinner { width: 60px; height: 60px; border: 4px solid transparent; border-top-color: var(--primary-color); border-bottom-color: var(--primary-color); border-radius: 50%; animation: spin 1s linear infinite; position: relative; }
.cyber-spinner::before { content: ''; position: absolute; top: 10px; left: 10px; right: 10px; bottom: 10px; border: 4px solid transparent; border-left-color: #ff00ea; border-right-color: #ff00ea; border-radius: 50%; animation: spin-reverse 0.5s linear infinite; }
@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
@keyframes spin-reverse { 0% { transform: rotate(360deg); } 100% { transform: rotate(0deg); } }
.loading-text { margin-top: 20px; font-weight: bold; color: var(--primary-color); letter-spacing: 2px; text-transform: uppercase; }

.tech-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2rem; }
.glow-text { font-size: 2.2rem; margin: 0; font-weight: 900; background: linear-gradient(to right, var(--primary-color), #ff00ea); -webkit-background-clip: text; color: transparent; text-shadow: 0 4px 20px rgba(0,0,0,0.1); }
.version-tag { font-size: 1rem; color: #fff; background: linear-gradient(45deg, #ff00ea, var(--primary-color)); padding: 2px 8px; border-radius: 4px; vertical-align: middle; margin-left: 10px; }
.header-actions { display: flex; align-items: center; gap: 1.5rem; }

.tech-card { background: var(--card-bg); backdrop-filter: blur(20px); border: 1px solid var(--card-border); border-radius: 16px; padding: 1.5rem; box-shadow: 0 10px 30px rgba(0,0,0,0.1); position: relative; transition: all 0.3s ease; }
.tech-card:hover { transform: translateY(-3px); box-shadow: 0 15px 40px rgba(0,0,0,0.15); border-color: rgba(255,255,255,0.1); }
.interactive-card { border: 1px dashed var(--primary-color); cursor: pointer; }
.interactive-card:hover { border: 1px solid var(--primary-color); background: rgba(0, 243, 255, 0.05); }

/* =========== 诊断概览卡片 =========== */
.summary-card { margin-bottom: 2rem; display: flex; align-items: center; gap: 2rem; border-left: 6px solid var(--primary-color); }
.border-danger { border-left-color: #ff4d4f !important; } .border-safe { border-left-color: #00ffaa !important; } .border-warn { border-left-color: #ffd700 !important; }
.icon-pulse { width: 24px; height: 24px; border-radius: 50%; animation: pulse 2s infinite cubic-bezier(0.4, 0, 0.2, 1); flex-shrink: 0;}
@keyframes pulse { 0% { transform: scale(0.9); box-shadow: 0 0 0 0 inherit; } 70% { transform: scale(1.1); box-shadow: 0 0 0 15px transparent; } 100% { transform: scale(0.9); box-shadow: 0 0 0 0 transparent; } }

.card-title-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
.card-title-row h2 { margin: 0; font-size: 1.3rem; color: var(--text-main); }

.card-actions { display: flex; gap: 12px; align-items: center; }
.force-refresh-btn, .trace-btn { display: flex; align-items: center; gap: 6px; background: transparent; border: 1px solid; padding: 6px 16px; border-radius: 20px; font-size: 0.9rem; font-weight: bold; cursor: pointer; transition: all 0.2s; }
.force-refresh-btn:hover:not(:disabled), .trace-btn:hover { background: rgba(255,255,255,0.1); transform: scale(1.05); }
.force-refresh-btn:disabled { opacity: 0.6; cursor: not-allowed; }
.spin-anim { animation: spin 1s linear infinite; }

.summary-text { margin: 0; font-size: 1.1rem; line-height: 1.6; color: var(--text-muted); }
.hint-text { font-size: 0.8rem; color: #888; font-weight: normal; margin-left: 10px; animation: blink 2s infinite; }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

/* =========== 图表网格 =========== */
.dashboard-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 2rem; }
.card-header { display: flex; align-items: center; margin-bottom: 1rem; border-bottom: 1px solid var(--card-border); padding-bottom: 0.8rem; }
.dot { width: 8px; height: 8px; background: var(--primary-color); border-radius: 50%; margin-right: 12px; }
.card-header h2 { font-size: 1.1rem; margin: 0; color: var(--text-main); font-weight: bold; }
.chart-container { width: 100%; height: 320px; }

.status-badge { padding: 0.6rem 1.2rem; border-radius: 50px; font-size: 0.95rem; font-weight: bold; border: 1px solid transparent; }
.status-normal { background: rgba(0, 243, 255, 0.1); border-color: var(--primary-color); color: var(--primary-color); }
.status-danger { background: rgba(255, 77, 79, 0.1); border-color: #ff4d4f; color: #ff4d4f; box-shadow: 0 0 15px rgba(255, 77, 79, 0.3); animation: glitch-border 1.5s infinite;}
.status-safe { background: rgba(0, 255, 170, 0.1); border-color: #00ffaa; color: #00ffaa; }
.status-warn { background: rgba(255, 215, 0, 0.1); border-color: #ffd700; color: #ffd700; }
@keyframes glitch-border { 0%, 100% { box-shadow: 0 0 10px rgba(255, 77, 79, 0.2) inset; } 50% { box-shadow: 0 0 25px rgba(255, 77, 79, 0.8) inset; } }

/* =========== 设置控制台 =========== */
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

/* =========== 深度侧滑抽屉 =========== */
.drawer-overlay { position: fixed; inset: 0; background: rgba(0, 0, 0, 0.4); backdrop-filter: blur(5px); z-index: 1000; }
.drawer-panel { position: fixed; top: 0; right: 0; width: 450px; max-width: 90vw; height: 100vh; z-index: 1001; border-radius: 20px 0 0 20px; border-top: none; border-right: none; border-bottom: none; border-left-width: 4px; border-left-style: solid; padding: 2rem; overflow-y: auto; display: flex; flex-direction: column; gap: 2rem; color: var(--text-main); background: var(--card-bg); backdrop-filter: blur(20px); }
.light-mode .drawer-panel { background: rgba(255, 255, 255, 0.95); }

.drawer-header { display: flex; justify-content: space-between; align-items: center; border-bottom: 1px solid var(--card-border); padding-bottom: 1rem; }
.drawer-header h3 { margin: 0; font-size: 1.4rem; }
.close-btn { width: 36px; height: 36px; padding: 0; }

.section-title { font-size: 1.1rem; color: var(--text-main); margin: 0 0 1rem 0; font-weight: bold; }

.tags-container { display: flex; flex-wrap: wrap; gap: 10px; }
.cyber-tag { padding: 6px 14px; border-radius: 20px; border: 1px solid; font-size: 0.9rem; font-weight: bold; box-shadow: 0 4px 10px rgba(0,0,0,0.1); transition: transform 0.2s; }
.cyber-tag:hover { transform: translateY(-2px); }

.rx-box { padding: 1rem; border-left: 4px solid; border-radius: 0 8px 8px 0; font-size: 1rem; line-height: 1.6; color: var(--text-muted); font-weight: 500; }

.tl-hint { font-size: 0.8rem; color: var(--text-muted); font-weight: normal; margin-left: 5px; }
.cyber-timeline { list-style: none; padding: 0; margin: 0; position: relative; }
.cyber-timeline::before { content: ''; position: absolute; left: 5px; top: 10px; bottom: 0; width: 2px; background: var(--card-border); }
.cyber-timeline li { position: relative; padding-left: 25px; margin-bottom: 2rem; }

.timeline-dot { position: absolute; left: 0; top: 5px; width: 12px; height: 12px; border-radius: 50%; background: var(--card-bg); border: 3px solid; transition: all 0.3s; }
.dot-pos { border-color: #00ffaa; box-shadow: 0 0 8px #00ffaa; }
.dot-neg { border-color: #ff4d4f; box-shadow: 0 0 8px #ff4d4f; }
.dot-neu { border-color: var(--text-muted); box-shadow: none; }

.timeline-content { display: flex; flex-direction: column; gap: 6px; }

.timeline-meta { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 2px; }
.timeline-meta .time { font-size: 0.85rem; color: var(--text-muted); font-family: monospace; }
.repeat-badge { display: flex; align-items: center; gap: 4px; background: rgba(255, 77, 79, 0.15); color: #ff4d4f; border: 1px solid #ff4d4f; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: bold; }

.cached-status { font-size: 0.75rem; font-family: monospace; padding: 2px 6px; border-radius: 4px; }
.cached-yes { background: rgba(255, 77, 79, 0.1); color: #ff4d4f; border: 1px dashed #ff4d4f; }
.cached-no { background: rgba(0, 255, 170, 0.1); color: #00ffaa; border: 1px dashed #00ffaa; }

.force-btn { background: #ff4d4f; color: #fff; border: none; padding: 3px 8px; border-radius: 6px; font-size: 0.7rem; font-weight: bold; cursor: pointer; transition: all 0.2s; box-shadow: 0 0 8px rgba(255, 77, 79, 0.4); display: flex; align-items: center; justify-content: center; min-width: 65px;}
.force-btn:hover:not(:disabled) { background: #ff7875; transform: scale(1.05); box-shadow: 0 0 12px rgba(255, 77, 79, 0.8); }
.force-btn:disabled { opacity: 0.6; cursor: not-allowed; transform: none; box-shadow: none; background: #cf1322; }

.timeline-content .title { font-size: 1.05rem; color: var(--text-main); text-decoration: none; line-height: 1.4; transition: color 0.2s; cursor: pointer; font-weight: 500; }
.timeline-content .title:hover { color: var(--primary-color); text-decoration: underline; }

.timeline-tags-row { display: flex; align-items: center; gap: 10px; margin-top: 4px; }
.source-tag { font-size: 0.75rem; background: rgba(128,128,128,0.1); color: var(--text-muted); padding: 2px 6px; border-radius: 4px; }

.sentiment-pill { font-size: 0.75rem; padding: 2px 8px; border-radius: 12px; font-weight: bold; }
.bg-pos { background: rgba(0, 255, 170, 0.1); color: #00ffaa; border: 1px solid #00ffaa; }
.bg-neg { background: rgba(255, 77, 79, 0.1); color: #ff4d4f; border: 1px solid #ff4d4f; }
.bg-neu { background: rgba(128, 128, 128, 0.1); color: var(--text-muted); border: 1px solid var(--text-muted); }

.fade-enter-active, .fade-leave-active { transition: opacity 0.4s ease; }
.fade-enter-from, .fade-leave-to { opacity: 0; }
.slide-enter-active, .slide-leave-active { transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1); }
.slide-enter-from, .slide-leave-to { transform: translateX(100%); }

.drawer-loading { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 60%; color: var(--primary-color); }
.small-spinner { width: 40px; height: 40px; margin-bottom: 20px; }
.drawer-empty { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 70%; text-align: center; color: var(--text-muted); opacity: 0.8; }
.drawer-empty p { font-size: 1.1rem; font-weight: bold; margin: 15px 0 5px 0; color: var(--text-main); }
.drawer-empty .empty-hint { font-size: 0.85rem; }
</style>
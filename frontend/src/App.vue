<template>
  <div class="dashboard-container" :style="customStyles" :class="{'light-mode': settings.isLightMode}">
    <div class="global-loading" v-if="analysis.status === 'loading'" role="status">
      <div class="cyber-spinner"></div><div class="loading-text">{{ t.running }}</div>
    </div>
    <header class="tech-header" :inert="drawerOpen">
      <h1 class="glow-text">Information Diet Manager <span class="version-tag">v2.0 PRO</span></h1>
      <div class="header-actions">
        <div class="settings-wrapper" @keydown.esc.stop="closeSettings">
          <button ref="settingsButton" class="icon-btn" @click="showSettings = !showSettings" title="设置 / Settings" aria-label="设置 / Settings" :aria-expanded="showSettings" aria-controls="ui-settings" data-testid="settings-toggle">
            <svg viewBox="0 0 24 24" width="24" height="24" stroke="currentColor" stroke-width="2" fill="none"><circle cx="12" cy="12" r="3"></circle><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path></svg>
          </button>
          <div id="ui-settings" class="settings-panel tech-card" v-show="showSettings">
            <h3>{{ t.sysUi }}</h3>
            <div class="setting-item"><label for="ui-language">{{ t.langLabel }}</label>
              <select id="ui-language" v-model="settings.lang"><option value="zh">🇨🇳 简体中文 (Chinese)</option><option value="en">🇺🇸 English (英文)</option></select>
            </div>
            <div class="setting-item"><label id="ui-mode-label">{{ t.visMode }}</label>
              <div class="theme-switch" role="switch" tabindex="0" aria-labelledby="ui-mode-label" :aria-checked="settings.isLightMode" data-testid="theme-toggle" @click="toggleTheme" @keydown.space.prevent="toggleTheme" @keydown.enter.prevent="toggleTheme">
                <div class="switch-bg" :class="{ 'is-light': settings.isLightMode }"></div>
                <span class="switch-label" :class="{ active: !settings.isLightMode }">{{ t.dark }}</span><span class="switch-label" :class="{ active: settings.isLightMode }">{{ t.light }}</span>
              </div>
            </div>
            <div class="setting-item"><label for="ui-color">{{ t.color }}</label><input id="ui-color" type="color" v-model="settings.themeColor"></div>
            <div class="setting-item"><label for="ui-font">{{ t.font }}</label>
              <select id="ui-font" v-model="settings.fontFamily">
                <option value="'Segoe UI', 'Microsoft YaHei', sans-serif">UI Sans-serif (现代无衬线)</option>
                <option value="Helvetica, Arial, sans-serif">Helvetica (经典纯净)</option>
                <option value="'Times New Roman', Times, serif">Times New Roman (衬线阅读)</option>
                <option value="'Courier New', Courier, monospace">Courier Code (极客等宽)</option>
              </select>
            </div>
            <section class="api-settings" :aria-label="t.connection">
              <h3>{{ t.connection }}</h3><p class="interface-hint">{{ t.keyHint }}</p>
              <form class="setting-item" @submit.prevent="connect">
                <input v-if="!connected" v-model="keyDraft" type="password" autocomplete="off" :aria-label="t.adminKey" :placeholder="t.adminKey" data-testid="admin-key">
                <button v-if="!connected" class="interface-btn" :disabled="pairing || !keyDraft.trim()" data-testid="connect">{{ pairing ? t.loading : t.connect }}</button>
                <button v-else type="button" class="interface-btn" @click="disconnect" data-testid="disconnect">{{ t.disconnect }}</button>
                <span class="interface-hint">{{ connected ? t.connected : t.disconnected }}</span>
              </form><p v-if="connectionMessage" class="interface-hint" role="status">{{ connectionMessage }}</p>
            </section>
            <section v-if="connected" class="api-settings" :aria-label="t.dataManagement">
              <h3>{{ t.dataManagement }}</h3><p class="interface-hint">{{ t.backupHint }}</p><p class="interface-hint">{{ t.deletionHint }}</p>
              <div class="setting-item">
                <button class="interface-btn" @click="fetchRecords()" :disabled="maintenance || recordsLoading" data-testid="refresh-records">{{ recordsLoading ? t.loading : t.refresh }}</button>
                <button class="interface-btn" @click="downloadBackup" :disabled="maintenance" data-testid="backup">{{ t.backup }}</button>
                <label for="restore-file">{{ t.restoreFile }}</label><input id="restore-file" type="file" accept=".json,application/json" @change="selectBackup" :disabled="maintenance" data-testid="restore-file">
                <button class="interface-btn" @click="restoreBackup" :disabled="maintenance || !restoreFile" data-testid="restore">{{ t.restore }}</button>
                <label for="delete-record-id">{{ t.chooseRecord }}</label>
                <select id="delete-record-id" v-model="deleteRecordId" :disabled="maintenance" data-testid="delete-record-id">
                  <option value="">{{ t.chooseRecord }}</option><option v-for="item in records.items" :key="item.id" :value="String(item.id)">{{ item.id }} · {{ item.title || t.untitled }}</option>
                </select>
                <p class="interface-hint">{{ t.selectionHint }}</p>
                <button class="interface-btn" @click="deleteRecord(deleteRecordId)" :disabled="maintenance || !deleteRecordId" data-testid="delete-record">{{ t.deleteRecord }}</button>
                <button class="interface-btn" @click="deleteAll" :disabled="maintenance" data-testid="delete-all">{{ t.deleteAll }}</button>
              </div><p v-if="maintenanceMessage" class="interface-hint" role="status" data-testid="data-result">{{ maintenanceMessage }}</p>
            </section>
          </div>
        </div>
        <div class="status-badge status-normal">{{ t.experimental }}: <span class="score">{{ connected ? t.connected : t.disconnected }}</span></div>
      </div>
    </header>
    <div class="summary-card tech-card" :inert="drawerOpen" :style="{ borderLeftColor: settings.themeColor }">
      <div class="icon-pulse" :style="{ backgroundColor: settings.themeColor, boxShadow: `0 0 0 0 ${settings.themeColor}80` }"></div>
      <div class="summary-content" style="flex: 1;">
        <div class="card-title-row">
          <h2>{{ t.report }}: {{ categoryLabel(category) }} <span class="hint-text">{{ t.hint }}</span></h2>
          <div class="card-actions">
            <button class="force-refresh-btn" :style="accentStyle" @click="runAnalysis" :disabled="!connected || maintenance || analysis.status === 'loading'" :title="t.analysisLimit" data-testid="run-analysis">
              <svg viewBox="0 0 24 24" width="14" height="14" stroke="currentColor" stroke-width="2" fill="none" :class="{ 'spin-anim': analysis.status === 'loading' }"><polyline points="23 4 23 10 17 10"></polyline><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"></path></svg>
              {{ analysis.status === 'loading' ? t.running : t.run }}
            </button>
            <button ref="recordsButton" class="trace-btn" :style="{ ...accentStyle, backgroundColor: `${settings.themeColor}15` }" @click="openRecords" :disabled="!connected || maintenance" data-testid="records-toggle">
              <svg viewBox="0 0 24 24" width="16" height="16" stroke="currentColor" stroke-width="2" fill="none"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>{{ t.traceBtn }}
            </button>
          </div>
        </div>
        <p class="summary-text"><strong data-testid="saved-total">{{ records.total ?? '—' }}</strong> {{ t.savedPages }} · {{ t.recordLimit }}<br>
          <span v-if="!connected">{{ t.connectHint }} </span><span role="status" aria-live="polite" data-testid="analysis-status">{{ analysisMessage }}</span>
        </p>
        <p v-if="recordsError" class="interface-hint" role="alert">{{ t.recordsError }}</p>
        <p v-if="recordsUpdatedAt" class="interface-hint" data-testid="records-status">{{ t.updated }} {{ formatTime(records.snapshotAt) }} · {{ recordsRefreshed ? t.recordsRefreshed : t.snapshotHint }}<span v-if="recordsError"> · {{ t.stale }}</span></p>
        <p v-if="analysis.generatedAt" class="interface-hint">{{ t.snapshot }} {{ formatTime(analysis.generatedAt) }}</p>
        <p v-if="analysis.window.from_ts != null" class="interface-hint">{{ t.window }} {{ formatUtc(analysis.window.from_ts) }} — {{ formatUtc(analysis.window.to_ts) }} (UTC)</p>
        <p v-if="analysis.window.truncated" role="alert" class="interface-hint">{{ t.truncated.replace('{n}', analysis.window.input_count).replace('{total}', analysis.window.available_count) }}</p>
        <p v-if="analysis.status === 'ready'" class="interface-hint">{{ t.coverage.replace('{n}', analysis.window.processed_count).replace('{total}', analysis.window.available_count) }}</p>
      </div>
    </div>
    <main class="dashboard-grid" :inert="drawerOpen" :aria-label="t.charts">
      <div class="tech-card interactive-card"><div class="card-header"><span class="dot"></span><h2 :title="t.categoriesHint">{{ t.categories }}</h2></div><div ref="pieRef" class="chart-container" role="img" :aria-label="t.categories"></div></div>
      <div class="tech-card"><div class="card-header"><span class="dot"></span><h2 :title="t.graphHint">{{ categoryLabel(category) }} - {{ t.graph }} <span class="mock-hint">{{ t.structureHint }}</span></h2></div><div ref="graphRef" class="chart-container" role="img" :aria-label="t.graphHint"></div></div>
      <div class="tech-card"><div class="card-header"><span class="dot"></span><h2 :title="t.repetitionHint">{{ categoryLabel(category) }} - {{ t.repetition }}</h2></div><div ref="repeatRef" class="chart-container" role="img" :aria-label="t.repetitionHint"></div></div>
      <div class="tech-card"><div class="card-header"><span class="dot"></span><h2 :title="t.sentimentHint">{{ categoryLabel(category) }} - {{ t.sentiment }}</h2></div><div ref="sentimentRef" class="chart-container" role="img" :aria-label="t.sentimentHint"></div></div>
    </main>
    <transition name="fade"><div class="drawer-overlay" v-if="drawerOpen" @click="closeRecords"></div></transition>
    <transition name="slide">
      <aside ref="recordsDialog" class="drawer-panel tech-card" v-if="drawerOpen" :style="{ borderLeftColor: settings.themeColor, boxShadow: `-10px 0 40px ${settings.themeColor}30` }" role="dialog" aria-modal="true" aria-labelledby="drawer-title" tabindex="-1" @keydown="onDrawerKeydown">
        <div class="drawer-header"><h3 id="drawer-title" :style="{ color: settings.themeColor }">{{ t.records }}</h3><button class="icon-btn close-btn" @click="closeRecords" :title="t.close" :aria-label="t.close" data-testid="drawer-close">✕</button></div>
        <p v-if="recordsError" class="interface-hint" role="alert">{{ t.recordsError }}</p>
        <div class="drawer-loading" v-if="recordsLoading && !records.items.length"><div class="cyber-spinner small-spinner"></div><p>{{ t.loading }}</p></div>
        <div class="drawer-content" v-else-if="records.items.length">
          <div class="drawer-section" v-if="recordTags.length"><h4 class="section-title">🧬 {{ t.kwTitle }}</h4><div class="tags-container"><span class="cyber-tag" v-for="tag in recordTags" :key="tag" :style="{ ...accentStyle, backgroundColor: `${settings.themeColor}15` }"># {{ tag }}</span></div></div>
          <div class="drawer-section"><h4 class="section-title">💊 {{ t.recordExplanation }}</h4><div class="rx-box" :style="{ borderLeftColor: settings.themeColor, backgroundColor: `${settings.themeColor}0A` }">{{ t.rawHint }}</div></div>
          <div class="drawer-section timeline-section"><h4 class="section-title">⏱️ {{ t.tlTitle }}</h4>
            <ul class="cyber-timeline"><li v-for="item in records.items" :key="item.id" data-testid="record-row">
              <div class="timeline-dot" :style="{ borderColor: settings.themeColor }"></div>
              <div class="timeline-content"><div class="timeline-meta"><time class="time">{{ formatTime(item.ts) }}</time></div>
                <a v-if="safeHttpUrl(item.url)" :href="safeHttpUrl(item.url)" target="_blank" rel="noopener noreferrer" class="title">{{ item.title || t.untitled }}</a><span v-else class="title">{{ item.title || t.untitled }}</span>
                <div class="timeline-tags-row"><span class="source-tag">{{ t.source }}: {{ sourceLabel(item.source) }}</span><span v-if="item.channel" class="source-tag">{{ t.channel }}: {{ item.channel }}</span></div>
              </div>
            </li></ul>
          </div>
        </div>
        <div class="drawer-empty" v-else-if="!recordsError"><svg viewBox="0 0 24 24" width="64" height="64" stroke="var(--text-muted)" stroke-width="1" fill="none"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line><line x1="11" y1="8" x2="11" y2="14"></line><line x1="8" y1="11" x2="14" y2="11"></line></svg><p>{{ t.noRecords }}</p><span class="empty-hint">{{ t.rawHint }}</span></div>
        <div class="pagination" v-if="records.total !== null"><p class="interface-hint">{{ t.loaded.replace('{n}', records.items.length).replace('{total}', records.total) }}</p><button v-if="records.hasMore" class="interface-btn" @click="fetchRecords(true)" :disabled="recordsLoading" data-testid="load-more">{{ recordsLoading ? t.loading : t.loadMore }}</button></div>
      </aside>
    </transition>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, onUnmounted, watch, nextTick } from 'vue'
import { createCyberCharts } from './cyber-charts.js'
import { createLocalApi } from './local-api.js'
import { analysisSnapshot, emptyAnalysis, recordsPage, mergeRecordsPage, safeHttpUrl } from './dashboard-data.js'

// Appearance is the owner's original 4d0bcd9 design; API lifecycle stays isolated.
const settings = reactive({ lang: 'zh', isLightMode: false, themeColor: '#00f3ff', fontFamily: "'Segoe UI', 'Microsoft YaHei', sans-serif" })
const lang = computed(() => settings.lang), showSettings = ref(false)
const copy = {
  zh: {
    connection: '连接本机服务', adminKey: '本机管理密钥（admin_token）', connect: '连接', disconnect: '断开并清除当前页面数据', connected: '已连接', disconnected: '尚未连接',
    keyHint: '从本机服务启动时提示的密钥文件复制管理密钥。密钥仅在当前页面内存中保存；刷新后需要重新连接。',
    dataManagement: '数据管理', backup: '下载页面记录备份', restoreFile: '选择备份文件', restore: '替换恢复', deleteAll: '删除全部已保存记录', deleteRecord: '删除此记录',
    backupHint: '备份是含网页内容的明文 JSON，最多 10,000 条 / 20 MiB。请保存在可信位置；备份不包含密钥、扩展队列或分析缓存。恢复会替换当前记录并重建向量。',
    deletionHint: '删除或恢复前，请先暂停扩展并清空其待同步队列，避免旧内容重新入库。分析缓存会一并清除；已下载备份需自行删除。',
    confirmDelete: '删除全部已保存页面记录和分析缓存？此操作不可撤销，请先按需下载备份。', confirmRecord: '删除此页面记录及全部分析缓存？', confirmRestore: '用所选备份替换当前全部页面记录？请先按需备份现有数据。',
    backupDone: '备份下载已发起。请检查浏览器下载结果并妥善保管。', deleteDone: '删除完成，分析缓存已清除。', restoreDone: '恢复完成，分析缓存已清除。',
    failedOperation: '操作结果未确认，请检查服务、备份文件并刷新核对数据；不要连续重复提交。', keyError: '连接失败：请使用有效的管理密钥，并检查本机接口和来源配置。', largeBackup: '备份文件超过 20 MiB。',
    subtitle: '本地页面记录与实验性文本统计', dark: '深色', light: '浅色',
    records: '已保存记录 · 全部', refresh: '刷新记录', viewRecords: '查看记录', savedPages: '条已保存页面记录',
    recordLimit: '当前记录按页面去重，不能代表访问次数、阅读时长或完整浏览历史。',
    loading: '正在读取…', recordsError: '读取记录失败，请检查本地服务后重试。已有显示值来自上次读取。', updated: '记录快照时间：', stale: '数据可能已过时',
    snapshotHint: '翻页保持同一份记录快照，重新打开抽屉或刷新记录可查看新增记录。', recordsRefreshed: '记录已发生变化，已重新读取第一页；实验图表已清空。',
    analysis: '近 7 天文本统计', experimental: '实验功能', run: '运行实验分析', running: '分析中…',
    analysisLimit: '仅在点击后分析最近 7×24 小时的已保存记录。结果不用于判断信息茧房、心理状态或健康程度。',
    states: { not_requested: '尚未运行实验分析。原始记录可独立查看。', loading: '正在分析，旧图表已清空。', empty: '所选时间范围内没有记录。', insufficient_data: '所选范围样本不足，至少需要 {n} 条记录；未生成分析结论。', unavailable: '实验分析暂不可用，请检查分析依赖和配置。原始记录仍可查看。', failed: '实验分析失败，未生成有效结果。', request_error: '分析请求失败或响应格式不兼容，请检查服务后重试。', ready: '实验统计已生成。图表展示本次快照；新记录不会自动加入，请按需重新运行。' },
    snapshot: '快照生成时间：', window: '分析范围：', truncated: '范围内共有 {total} 条记录，本次仅读取按时间升序排列的前 {n} 条，不能代表全部记录。', coverage: '本次处理记录 {n} 条；时间范围内已保存记录共 {total} 条。',
    metricCoverage: '分类有效 {categories}/{n} 条；情感有效 {sentiments}/{n} 条；日期有效 {timestamps}/{n} 条；相邻比较有效 {comparisons}/{pairs} 对。比较按后一条记录的 UTC 日期和分类归属。',
    charts: '实验统计图表', categories: '实验分类分布', categoriesHint: '同一次分析的有效样本数；点击分类可切换趋势。',
    readingGuide: '如何理解这些数据', guide1: '“没有数据”和“比例为零”含义不同，缺失日期保留为空白。', guide2: '相似度只比较相邻文本，不代表全部重复访问或全部重复内容。', guide3: '文本的积极、中性、消极标签不代表用户情绪或心理状态。',
    selectedCategory: '趋势分类', allCategories: '全部分类', repetition: '相邻文本高相似比例', repetitionHint: '实验指标：相邻文本相似度 ≥ 0.85；日期按 UTC。', sentiment: '文本情感标签比例', sentimentHint: '使用实际标签比例；日期按 UTC。',
    positive: '积极', neutral: '中性', negative: '消极', noData: '暂无可展示的实验数据', missingMetric: '所选分类缺少有效指标',
    cats: { entertainment: '娱乐', learning: '学习', news: '新闻', social: '社交', shopping: '购物', tools: '工具', other: '其他' },
    close: '关闭', rawHint: '这里展示全部已保存的原始页面记录，与上方实验分类筛选独立。接口尚未提供逐条情感、访问次数或缓存状态。', noRecords: '尚无已保存记录。', untitled: '未命名页面', unknown: '未知', source: '来源', channel: '采集频道', plugin: '扩展', imported: '导入', loaded: '已显示 {n} / {total} 条记录', loadMore: '加载更多',
  },
  en: {
    connection: 'Connect to local service', adminKey: 'Local admin key (admin_token)', connect: 'Connect', disconnect: 'Disconnect and clear page data', connected: 'Connected', disconnected: 'Disconnected',
    keyHint: 'Copy the admin key from the local file reported at service startup. It stays in page memory only; reconnect after refreshing.',
    dataManagement: 'Data management', backup: 'Download page-record backup', restoreFile: 'Select backup', restore: 'Replace from backup', deleteAll: 'Delete all saved records', deleteRecord: 'Delete this record',
    backupHint: 'Backups are plaintext JSON containing page content, up to 10,000 records / 20 MiB. Keep them private. Keys, extension queues and analysis caches are excluded. Restore replaces records and rebuilds vectors.',
    deletionHint: 'Pause the extension and clear its pending queue before deleting or restoring, to prevent old content being uploaded again. Analysis caches are cleared; downloaded backups must be deleted separately.',
    confirmDelete: 'Delete all saved records and analysis caches? This cannot be undone. Download a backup first if needed.', confirmRecord: 'Delete this page record and all analysis caches?', confirmRestore: 'Replace all current page records with this backup? Back up current data first if needed.',
    backupDone: 'Backup download started. Check the browser download and store it safely.', deleteDone: 'Deletion complete; analysis caches cleared.', restoreDone: 'Restore complete; analysis caches cleared.',
    failedOperation: 'Operation not confirmed. Check the service and backup, then refresh to verify the data before retrying.', keyError: 'Connection failed. Use a valid admin key and check the local API and origin configuration.', largeBackup: 'Backup exceeds 20 MiB.',
    subtitle: 'Local page records and experimental text statistics', dark: 'Dark', light: 'Light',
    records: 'Saved records · All', refresh: 'Refresh records', viewRecords: 'View records', savedPages: 'saved page records',
    recordLimit: 'Records are currently deduplicated by page. They do not measure visits, reading time, or complete browsing history.',
    loading: 'Loading…', recordsError: 'Could not read records. Check the local service and retry. Existing values are from the previous request.', updated: 'Records snapshot:', stale: 'May be outdated',
    snapshotHint: 'Pages share one snapshot. Reopen the drawer or refresh records to include new arrivals.', recordsRefreshed: 'Records changed. Reloaded the first page and cleared the experimental charts.',
    analysis: 'Text statistics · Last 7 days', experimental: 'Experimental', run: 'Run experimental analysis', running: 'Analyzing…',
    analysisLimit: 'Runs only on request, using saved records from the last 7×24 hours. These results do not assess echo chambers, mental state, or health.',
    states: { not_requested: 'Analysis has not been run. Raw records are available independently.', loading: 'Analyzing. Previous charts have been cleared.', empty: 'No records in this time window.', insufficient_data: 'At least {n} records are required in this window. No analysis was produced.', unavailable: 'Experimental analysis is unavailable. Check its dependencies and configuration. Raw records remain available.', failed: 'Experimental analysis failed. No valid result was produced.', request_error: 'Analysis request failed or the response is incompatible. Check the service and retry.', ready: 'This is an experimental snapshot. Run analysis again to include new records.' },
    snapshot: 'Snapshot generated:', window: 'Window:', truncated: 'The window contains {total} records. Only the first {n} in ascending timestamp order were read; this does not represent all records.', coverage: '{n} records processed; {total} saved records in the window.',
    metricCoverage: 'Valid categories: {categories}/{n} records; sentiment: {sentiments}/{n}; dates: {timestamps}/{n}; adjacent comparisons: {comparisons}/{pairs} pairs. Each comparison belongs to the later record’s UTC date and category.',
    charts: 'Experimental charts', categories: 'Experimental categories', categoriesHint: 'Valid sample counts from this analysis. Click a category to filter trends.',
    readingGuide: 'Reading these statistics', guide1: 'Missing values are different from zero. Dates without data remain blank.', guide2: 'Similarity compares adjacent texts only; it does not measure all repeated visits or content.', guide3: 'Positive, neutral and negative text labels do not describe your mood or mental state.',
    selectedCategory: 'Trend category', allCategories: 'All categories', repetition: 'High adjacent-text similarity', repetitionHint: 'Experimental: adjacent-text similarity ≥ 0.85. Dates use UTC.', sentiment: 'Text sentiment proportions', sentimentHint: 'Actual label proportions. Dates use UTC.',
    positive: 'Positive', neutral: 'Neutral', negative: 'Negative', noData: 'No experimental data to display', missingMetric: 'No valid metric for this category',
    cats: { entertainment: 'Entertainment', learning: 'Learning', news: 'News', social: 'Social', shopping: 'Shopping', tools: 'Tools', other: 'Other' },
    close: 'Close', rawHint: 'All saved raw page records, independent of the experimental category filter. Per-record sentiment, visit counts and cache status are not provided.', noRecords: 'No saved records.', untitled: 'Untitled page', unknown: 'Unknown', source: 'Source', channel: 'Captured channel', plugin: 'Extension', imported: 'Import', loaded: 'Showing {n} of {total} records', loadMore: 'Load more',
  },
}

const uiCopy = {
  zh: { sysUi: '系统 UI 引擎', langLabel: '系统语言 / Language', visMode: '视觉模式', color: '全局高亮色', font: '界面字体', report: '实验统计', hint: '(点击环形图下钻)', traceBtn: '深度溯源', connectHint: '请从右上角设置连接本机服务。', graph: '统计结构', structureHint: '(示意)', graphHint: '全局展示本次样本分类数量；分类视图展示实际日期的相邻文本高相似比例。这是统计结构示意，不是语义关系图。', graphRoot: '本次样本', returnGlobal: '返回全局', chooseRecord: '选择待删除记录', selectionHint: '这里只列出已加载记录；更多记录可在右侧抽屉加载。', kwTitle: '已加载记录标签', recordExplanation: '记录口径说明', tlTitle: '真实页面记录' },
  en: { sysUi: 'System UI Engine', langLabel: 'System Language', visMode: 'Visual Mode', color: 'Highlight Color', font: 'Interface Font', report: 'Experimental Statistics', hint: '(Click chart to explore)', traceBtn: 'Trace Records', connectHint: 'Connect to your local service in the top-right settings.', graph: 'Statistical Structure', structureHint: '(illustration)', graphHint: 'Global: category sample counts. Category: actual dated adjacent-text similarity proportions. This illustrates statistics, not semantic relationships.', graphRoot: 'Current samples', returnGlobal: 'Back to global', chooseRecord: 'Choose a record to delete', selectionHint: 'Only loaded records are listed. Load more in the right-hand drawer.', kwTitle: 'Loaded Record Tags', recordExplanation: 'About These Records', tlTitle: 'Saved Page Timeline' },
}
const t = computed(() => ({ ...copy[lang.value], ...uiCopy[lang.value] }))
watch(lang, value => { document.documentElement.lang = value === 'zh' ? 'zh-CN' : 'en' }, { immediate: true })
const customStyles = computed(() => ({
  '--primary-color': settings.themeColor, '--font-family': settings.fontFamily,
  '--bg-gradient': settings.isLightMode ? 'linear-gradient(135deg, #f0f2f5 0%, #e2e8f0 100%)' : 'radial-gradient(circle at center, #111827 0%, #030712 100%)',
  '--card-bg': settings.isLightMode ? 'rgba(255, 255, 255, 0.85)' : 'rgba(17, 25, 40, 0.65)',
  '--card-border': settings.isLightMode ? 'rgba(0, 0, 0, 0.08)' : 'rgba(255, 255, 255, 0.05)',
  '--text-main': settings.isLightMode ? '#1e293b' : '#f8fafc', '--text-muted': settings.isLightMode ? '#64748b' : '#94a3b8',
}))
const accentStyle = computed(() => ({ color: settings.themeColor, borderColor: settings.themeColor }))
const toggleTheme = () => { settings.isLightMode = !settings.isLightMode }
function closeSettings() { showSettings.value = false; settingsButton.value?.focus() }
let api
try { api = createLocalApi(import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000') } catch { /* Show connection failure without sending keys anywhere. */ }
const connected = ref(false), pairing = ref(false), keyDraft = ref(''), connectionMessage = ref('')
const maintenance = ref(false), maintenanceMessage = ref(''), restoreFile = ref(null)
let dataGeneration = 0
const emptyRecords = () => ({ items: [], total: null, page: 0, pageSize: 50, hasMore: false, nextCursor: null, snapshotAt: null })
const records = reactive(emptyRecords())
const recordsLoading = ref(false), recordsError = ref(false), recordsUpdatedAt = ref(null), recordsRefreshed = ref(false)
const analysis = ref(emptyAnalysis()), category = ref('global'), drawerOpen = ref(false)
const recordsDialog = ref(null), recordsButton = ref(null), settingsButton = ref(null)
const deleteRecordId = ref('')
const pieRef = ref(null), graphRef = ref(null), repeatRef = ref(null), sentimentRef = ref(null)
let recordController, analysisController, timer, charts
let recordBackground = false
let disposed = false
const analysisMessage = computed(() => {
  const state = analysis.value
  const message = (t.value.states[state.status] || t.value.states.request_error).replace('{n}', state.minimumRecords ?? '—')
  if (state.status !== 'ready' || !state.coverage) return message
  const coverage = state.coverage
  const values = { n: coverage.record_count, categories: coverage.category_count, sentiments: coverage.sentiment_count,
    timestamps: coverage.timestamp_count, comparisons: coverage.comparison_count, pairs: Math.max(0, coverage.record_count - 1) }
  return message + ' ' + t.value.metricCoverage.replace(/\{(\w+)\}/g, (_, key) => values[key])
})
const categoryLabel = key => key === 'global' ? t.value.allCategories : (t.value.cats[key] || key)
const sourceLabel = source => source === 'plugin' ? t.value.plugin : source === 'import' ? t.value.imported : (source || t.value.unknown)
const formatTime = value => typeof value === 'number' && Number.isFinite(new Date(value).getTime()) ? new Date(value).toLocaleString(lang.value === 'zh' ? 'zh-CN' : 'en-GB') : t.value.unknown
const formatUtc = value => typeof value === 'number' && Number.isFinite(new Date(value).getTime()) ? new Date(value).toISOString().replace('T', ' ').slice(0, 16) : t.value.unknown

function resetData() {
  dataGeneration++; recordController?.abort(); analysisController?.abort()
  recordController = undefined; recordBackground = false
  Object.assign(records, emptyRecords())
  recordsLoading.value = false; recordsError.value = false; recordsUpdatedAt.value = null; recordsRefreshed.value = false
  analysis.value = emptyAnalysis(); category.value = 'global'; deleteRecordId.value = ''
}
function disconnect() {
  api?.disconnect(); connected.value = false; keyDraft.value = ''; resetData()
  restoreFile.value = null; maintenanceMessage.value = ''; drawerOpen.value = false; showSettings.value = true
}
async function connect() {
  if (pairing.value || connected.value) return
  pairing.value = true; connectionMessage.value = ''; resetData()
  try {
    if (!api) throw new Error('INVALID_LOCAL_API')
    await api.pair(keyDraft.value.trim()); connected.value = true; keyDraft.value = ''; await fetchRecords()
  } catch { connected.value = false; connectionMessage.value = t.value.keyError }
  finally { pairing.value = false }
}
function selectBackup(event) {
  const file = event.target.files?.[0]
  restoreFile.value = file && file.size <= 20 * 1024 * 1024 ? file : null
  maintenanceMessage.value = file && !restoreFile.value ? t.value.largeBackup : ''
}
async function dataAction(task, success) {
  if (!connected.value || maintenance.value) return
  maintenance.value = true; maintenanceMessage.value = ''; resetData()
  const epoch = dataGeneration
  try { await task(); if (epoch === dataGeneration) maintenanceMessage.value = success }
  catch (error) {
    if (epoch === dataGeneration) {
      if (error.message === 'HTTP_401') { disconnect(); connectionMessage.value = t.value.keyError }
      else maintenanceMessage.value = t.value.failedOperation
    }
  }
  finally { maintenance.value = false; if (connected.value) await fetchRecords() }
}
async function downloadBackup() {
  await dataAction(async () => {
    const { data } = await api.get('/data/backup', { responseType: 'blob' })
    const url = URL.createObjectURL(data), link = document.createElement('a')
    link.href = url; link.download = 'idm-pages-backup.json'; link.click(); setTimeout(() => URL.revokeObjectURL(url), 10000)
  }, t.value.backupDone)
}
async function restoreBackup() {
  if (!restoreFile.value || !window.confirm(t.value.confirmRestore)) return
  const file = restoreFile.value
  await dataAction(() => api.post('/data/restore', file, { headers: { 'X-IDM-Confirm': 'replace-records' } }), t.value.restoreDone)
}
async function deleteAll() {
  if (window.confirm(t.value.confirmDelete)) await dataAction(() => api.delete('/data', { headers: { 'X-IDM-Confirm': 'delete-all' } }), t.value.deleteDone)
}
async function deleteRecord(id) {
  if (!records.items.some(item => String(item.id) === String(id))) return
  if (window.confirm(t.value.confirmRecord)) await dataAction(() => api.delete('/items/' + id, { headers: { 'X-IDM-Confirm': 'delete-record' } }), t.value.deleteDone)
}

async function fetchRecords(append = false, background = false) {
  if (disposed || !connected.value || maintenance.value || (append && !records.hasMore)) return
  if (recordController) {
    // Explicit reads take priority over polling. Foreground reads remain
    // serialized so two page requests cannot append the same cursor.
    if (background || !recordBackground) return
    recordController.abort()
  }
  recordsLoading.value = !background
  const controller = new AbortController()
  let epoch = dataGeneration
  recordController = controller; recordBackground = background
  const current = () => !disposed && epoch === dataGeneration && recordController === controller && !controller.signal.aborted
  try {
    let response, refreshed = false
    try {
      response = await api.get('/items', { params: { pagination: 'cursor', page_size: 50,
        ...(append ? { cursor: records.nextCursor } : {}) }, signal: controller.signal })
    } catch (error) {
      if (!append || error.code !== 'items_snapshot_expired' || !current()) throw error
      // The old set can contain deleted or replaced records. Clear it before
      // the single recovery request; never merge across snapshots or retry in a loop.
      dataGeneration++; epoch = dataGeneration; analysisController?.abort()
      Object.assign(records, emptyRecords()); recordsUpdatedAt.value = null
      deleteRecordId.value = ''; analysis.value = emptyAnalysis(); category.value = 'global'
      recordsRefreshed.value = false; refreshed = true; append = false
      response = await api.get('/items', { params: { pagination: 'cursor', page_size: 50 }, signal: controller.signal })
    }
    // An already-started poll must not replace loaded pages while the user
    // is choosing a record in settings or reading the drawer.
    if (!current() || (background && (showSettings.value || drawerOpen.value))) return
    const page = recordsPage(response.data)
    if (!append && page.page !== 1) throw new Error('Invalid first records page')
    Object.assign(records, append ? mergeRecordsPage(records, page) : page)
    if (!append) recordsRefreshed.value = refreshed
    if (!records.items.some(item => String(item.id) === deleteRecordId.value)) deleteRecordId.value = ''
    recordsUpdatedAt.value = Date.now()
    recordsError.value = false
  } catch (error) {
    if (current() && error.message === 'HTTP_401') { disconnect(); connectionMessage.value = t.value.keyError }
    else if (current() && error.name !== 'AbortError') recordsError.value = true
  } finally {
    if (recordController === controller) {
      recordController = undefined; recordBackground = false; recordsLoading.value = false
    }
  }
}
async function runAnalysis() {
  if (!connected.value || maintenance.value || analysis.value.status === 'loading') return
  analysis.value = emptyAnalysis('loading')
  category.value = 'global'
  const epoch = dataGeneration
  analysisController = new AbortController()
  try {
    const response = await api.get('/dashboard/visualization', { params: { days: 7, force: true }, signal: analysisController.signal })
    if (epoch === dataGeneration) analysis.value = analysisSnapshot(response.data)
  } catch (error) {
    if (epoch === dataGeneration && error.message === 'HTTP_401') { disconnect(); connectionMessage.value = t.value.keyError }
    else if (epoch === dataGeneration && error.name !== 'AbortError') analysis.value = emptyAnalysis('request_error')
  }
}

const recordTags = computed(() => [...new Set(records.items.flatMap(item => Array.isArray(item.tags) ? item.tags.filter(tag => typeof tag === 'string' && tag.trim()) : []))].slice(0, 20))
let restoreBodyScroll
watch(drawerOpen, open => {
  if (open) {
    const body = document.body, overflow = body.style.overflow, padding = body.style.paddingRight
    const gutter = window.innerWidth - document.documentElement.clientWidth
    if (gutter > 0) body.style.paddingRight = `${parseFloat(getComputedStyle(body).paddingRight) + gutter}px`
    body.style.overflow = 'hidden'
    restoreBodyScroll = () => { body.style.overflow = overflow; body.style.paddingRight = padding }
  } else { restoreBodyScroll?.(); restoreBodyScroll = undefined }
})
async function openRecords() {
  if (!connected.value || maintenance.value) return
  showSettings.value = false; drawerOpen.value = true
  await nextTick(); recordsDialog.value?.focus(); fetchRecords()
}
async function closeRecords() { drawerOpen.value = false; await nextTick(); recordsButton.value?.focus() }
function onDrawerKeydown(event) {
  if (event.key === 'Escape') { event.preventDefault(); closeRecords(); return }
  if (event.key !== 'Tab') return
  const focusable = [...recordsDialog.value.querySelectorAll('button:not(:disabled), a[href], [tabindex="0"]')]
  const first = focusable[0], last = focusable.at(-1)
  if (!first) { event.preventDefault(); recordsDialog.value.focus(); return }
  if (event.shiftKey && (document.activeElement === first || document.activeElement === recordsDialog.value)) { event.preventDefault(); last.focus() }
  else if (!event.shiftKey && (document.activeElement === last || document.activeElement === recordsDialog.value)) { event.preventDefault(); first.focus() }
}
function updateCharts() {
  if (disposed || !charts) return
  charts.render({ analysis: analysis.value, category: category.value, settings, labels: {
    categoryName: categoryLabel, noData: t.value.noData, missingMetric: t.value.missingMetric,
    returnGlobal: t.value.returnGlobal, sent: [t.value.positive, t.value.neutral, t.value.negative], graphRoot: t.value.graphRoot,
  } })
}
watch([analysis, category, settings], () => nextTick(updateCharts), { deep: true })
const resize = () => charts?.resize()
onMounted(() => {
  charts = createCyberCharts({ pie: pieRef.value, graph: graphRef.value, repetition: repeatRef.value, sentiment: sentimentRef.value }, key => { category.value = key })
  updateCharts(); window.addEventListener('resize', resize)
  timer = setInterval(() => { if (!drawerOpen.value && !showSettings.value) fetchRecords(false, true) }, 30000)
})
onUnmounted(() => {
  disposed = true; api?.disconnect(); clearInterval(timer); recordController?.abort(); analysisController?.abort()
  restoreBodyScroll?.()
  window.removeEventListener('resize', resize); charts?.dispose()
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
.tech-card:not(.drawer-panel):hover { transform: translateY(-3px); box-shadow: 0 15px 40px rgba(0,0,0,0.15); border-color: rgba(255,255,255,0.1); }
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

/* Interface additions stay within the original settings panel. */
.settings-panel { max-height: calc(100dvh - 160px); overflow-y: auto; overscroll-behavior: contain; }
.api-settings { border-top: 1px solid var(--card-border); padding-top: 1rem; margin-top: 1rem; }
.interface-hint { color: var(--text-muted); font-size: 0.85rem; line-height: 1.6; overflow-wrap: anywhere; }
.api-settings input { box-sizing: border-box; width: 100%; min-width: 0; color: var(--text-main); background: var(--card-bg); border: 1px solid var(--card-border); border-radius: 6px; padding: 0.6rem; }
.interface-btn { color: var(--primary-color); background: transparent; border: 1px solid var(--primary-color); border-radius: 20px; padding: 6px 16px; cursor: pointer; font: inherit; }
button:disabled { opacity: 0.55; cursor: not-allowed; }
button:focus-visible, a:focus-visible, input:focus-visible, select:focus-visible, [role="switch"]:focus-visible { outline: 2px solid var(--primary-color); outline-offset: 3px; }
.drawer-panel { max-height: calc(100dvh - 4rem); overscroll-behavior: contain; }
.drawer-panel:focus { outline: none; }
.drawer-section { margin-bottom: 2rem; }
.timeline-content, .summary-content, .dashboard-grid > .tech-card { min-width: 0; }
.timeline-content { overflow-wrap: anywhere; }
.cyber-tag { max-width: 100%; box-sizing: border-box; overflow-wrap: anywhere; }
.pagination { margin-top: auto; padding-bottom: 1rem; }
.mock-hint { color: var(--text-muted); font-size: 0.75rem; font-weight: normal; }
@media (max-width: 900px) { .tech-header, .card-title-row { flex-wrap: wrap; gap: 1rem; } .settings-panel { right: auto; left: 0; max-width: calc(100vw - 7rem); } .dashboard-grid { grid-template-columns: minmax(0, 1fr); } .card-actions { flex-wrap: wrap; } .drawer-panel { max-width: calc(100vw - 4rem - 4px); } }
</style>

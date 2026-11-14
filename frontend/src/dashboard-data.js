// Boundary between API payloads and chart values. Missing measurements stay null.
export const CATEGORY_KEYS = ['entertainment', 'learning', 'news', 'social', 'shopping', 'tools', 'other']

export function ratioPercent(value) {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1
    ? Math.round(value * 1000000) / 10000
    : null
}

function dateNumber(value) {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return null
  const ms = Date.parse(`${value}T00:00:00Z`)
  return Number.isFinite(ms) && new Date(ms).toISOString().slice(0, 10) === value ? ms : null
}

export function dailySeries(rows = [], window = {}) {
  const byDate = new Map(rows.filter(row => dateNumber(row.date) !== null).map(row => [row.date, row]))
  const suppliedDates = [...byDate.keys()].sort()
  const start = Number.isFinite(window.from_ts) ? dateNumber(new Date(window.from_ts).toISOString().slice(0, 10)) : dateNumber(suppliedDates[0])
  const end = Number.isFinite(window.to_ts) ? dateNumber(new Date(window.to_ts).toISOString().slice(0, 10)) : dateNumber(suppliedDates.at(-1))
  const dates = []
  if (start !== null && end !== null && end >= start && end - start <= 91 * 86400000) {
    for (let day = start; day <= end; day += 86400000) dates.push(new Date(day).toISOString().slice(0, 10))
  }
  const result = { dates, repeat: [], positive: [], neutral: [], negative: [] }
  for (const date of dates) {
    const row = byDate.get(date)
    const hasSamples = Number.isInteger(row?.count) && row.count > 0
    result.repeat.push(hasSamples ? ratioPercent(row.repeat_ratio) : null)
    const proportions = ['positive_ratio', 'neutral_ratio', 'negative_ratio'].map(key => row?.[key])
    const valid = hasSamples && proportions.every(value => ratioPercent(value) !== null)
      && Math.abs(proportions.reduce((sum, value) => sum + value, 0) - 1) <= 0.00001
    ;['positive', 'neutral', 'negative'].forEach((key, index) => {
      result[key].push(valid ? ratioPercent(proportions[index]) : null)
    })
  }
  return result
}

export function emptyAnalysis(status = 'not_requested') {
  return { status, window: {}, generatedAt: null, minimumRecords: null, counts: {}, series: {}, coverage: null, warning: false }
}

export function analysisSnapshot(payload) {
  const statuses = ['empty', 'insufficient_data', 'unavailable', 'failed', 'ready']
  if (!payload || !statuses.includes(payload.analysis_status) || !payload.window) {
    throw new Error('Unsupported analysis response')
  }
  const snapshot = {
    ...emptyAnalysis(payload.analysis_status), window: { ...payload.window },
    generatedAt: payload.generated_at, minimumRecords: payload.minimum_records,
    warning: Boolean(payload.pipeline_warning),
  }
  if (Object.hasOwn(payload, 'coverage')) {
    const coverage = payload.coverage
    const fields = ['record_count', 'timestamp_count', 'category_count', 'comparison_count', 'sentiment_count']
    if (!coverage || typeof coverage !== 'object' || Array.isArray(coverage)
      || fields.some(key => !Number.isSafeInteger(coverage[key]) || coverage[key] < 0)
      || fields.some(key => coverage[key] > coverage.record_count)
      || coverage.comparison_count > Math.max(0, coverage.record_count - 1)) {
      throw new Error('Invalid analysis coverage')
    }
    snapshot.coverage = Object.fromEntries(fields.map(key => [key, coverage[key]]))
  }
  // A warning must never become a successful-looking measurement.
  if (snapshot.warning && snapshot.status === 'ready') snapshot.status = 'failed'
  if (snapshot.status !== 'ready') return snapshot
  if (!payload.category_counts || !Array.isArray(payload.global?.time_series)) {
    throw new Error('Incomplete analysis response')
  }
  for (const [key, count] of Object.entries(payload.category_counts)) {
    if (!Number.isInteger(count) || count < 0) throw new Error('Invalid category count')
    snapshot.counts[key] = count
  }
  snapshot.series.global = dailySeries(payload.global.time_series, payload.window)
  for (const [key, category] of Object.entries(payload.categories || {})) {
    snapshot.series[key] = dailySeries(category.time_series, payload.window)
  }
  return snapshot
}

export function recordsPage(payload) {
  if (!payload || payload.pagination !== 'cursor' || !Array.isArray(payload.items)
    || !Number.isSafeInteger(payload.total) || payload.total < 0
    || !Number.isSafeInteger(payload.page) || payload.page < 1
    || !Number.isSafeInteger(payload.page_size) || payload.page_size < 1 || payload.page_size > 200
    || typeof payload.has_more !== 'boolean'
    || !Number.isSafeInteger(payload.snapshot_at) || payload.snapshot_at < 0
    || !Number.isFinite(new Date(payload.snapshot_at).getTime())
    || (payload.has_more ? typeof payload.next_cursor !== 'string' || !payload.next_cursor.length : payload.next_cursor !== null)) {
    throw new Error('Invalid records response')
  }
  const offset = (payload.page - 1) * payload.page_size
  const expectedCount = Math.min(payload.page_size, Math.max(0, payload.total - offset))
  if (payload.items.length !== expectedCount || payload.has_more !== (offset + payload.items.length < payload.total)
    || payload.items.some((item, index) => !Number.isSafeInteger(item?.id) || item.id < 1
      || (index > 0 && item.id >= payload.items[index - 1].id))) {
    throw new Error('Inconsistent records response')
  }
  return { total: payload.total, page: payload.page, pageSize: payload.page_size, items: payload.items,
    hasMore: payload.has_more, nextCursor: payload.next_cursor, snapshotAt: payload.snapshot_at }
}

export function mergeRecordsPage(current, incoming) {
  if (incoming.page !== current.page + 1 || incoming.pageSize !== current.pageSize
    || incoming.total !== current.total || incoming.snapshotAt !== current.snapshotAt
    || !current.hasMore || current.items.length !== current.page * current.pageSize
    || (incoming.items.length && incoming.items[0].id >= current.items.at(-1)?.id)) {
    throw new Error('Mismatched records snapshot')
  }
  return { ...incoming, items: [...current.items, ...incoming.items] }
}

export function safeHttpUrl(value) {
  try {
    const url = new URL(value)
    return ['http:', 'https:'].includes(url.protocol) ? url.href : null
  } catch {
    return null
  }
}

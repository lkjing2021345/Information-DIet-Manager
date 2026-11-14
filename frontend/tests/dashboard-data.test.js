import test from 'node:test'
import assert from 'node:assert/strict'
import { dailySeries, ratioPercent, analysisSnapshot, emptyAnalysis, recordsPage, mergeRecordsPage, safeHttpUrl } from '../src/dashboard-data.js'

const day = (date, values = {}) => ({ date, count: 5, repeat_ratio: 0, positive_ratio: 1, negative_ratio: 0, neutral_ratio: 0, ...values })
const ready = () => ({ analysis_status: 'ready', window: { input_count: 5, processed_count: 5, available_count: 5 }, minimum_records: 5, generated_at: 1000, category_counts: { tools: 3, shopping: 2 }, global: { time_series: [day('2026-09-24')] }, categories: { tools: { time_series: [day('2026-09-24')] } } })

test('100% positive stays 100/0/0; actual zeroes are preserved', () => {
  const data = dailySeries([day('2026-09-24')])
  assert.deepEqual([data.positive, data.neutral, data.negative, data.repeat], [[100], [0], [0], [0]])
})
test('missing and invalid sentiment are unknown, not neutral or a default percentage', () => {
  for (const invalid of [undefined, null, -1, 2, '0', NaN]) {
    const result = dailySeries([day('2026-09-24', { negative_ratio: invalid })])
    assert.deepEqual([result.positive, result.neutral, result.negative], [[null], [null], [null]])
  }
  assert.deepEqual(dailySeries([day('2026-09-24', { neutral_ratio: 0.5 })]).positive, [null])
  assert.deepEqual(dailySeries([day('2026-09-24', { count: 0 })]).positive, [null])
})
test('aligns actual UTC dates and leaves missing dates blank', () => {
  const data = dailySeries([day('2026-09-24')], { from_ts: Date.parse('2026-09-22T23:30Z'), to_ts: Date.parse('2026-09-24T01:00Z') })
  assert.deepEqual(data.dates, ['2026-09-22', '2026-09-23', '2026-09-24'])
  assert.deepEqual(data.positive, [null, null, 100])
  assert.deepEqual(data.repeat, [null, null, 0])
})
test('sorts dates and rejects impossible dates without fabricating points', () => {
  assert.deepEqual(dailySeries([day('2026-02-30')]).dates, [])
  const data = dailySeries([day('2026-09-24'), day('2026-09-22')])
  assert.deepEqual(data.positive, [100, null, 100])
  assert.equal(ratioPercent(null), null)
})
test('tools and shopping remain separate categories from the same analysis', () => {
  assert.deepEqual(analysisSnapshot(ready()).counts, { tools: 3, shopping: 2 })
})

test('metric coverage preserves independent valid counts while legacy responses stay unknown', () => {
  const coverage = { record_count: 5, timestamp_count: 4, category_count: 3, comparison_count: 2, sentiment_count: 1 }
  assert.deepEqual(analysisSnapshot({ ...ready(), coverage }).coverage, coverage)
  assert.equal(analysisSnapshot(ready()).coverage, null)
  assert.equal(emptyAnalysis().coverage, null)
  const zeros = Object.fromEntries(Object.keys(coverage).map(key => [key, 0]))
  assert.deepEqual(analysisSnapshot({ ...ready(), coverage: zeros }).coverage, zeros)
})

test('invalid metric coverage cannot produce a successful-looking snapshot', () => {
  const coverage = { record_count: 5, timestamp_count: 5, category_count: 5, comparison_count: 4, sentiment_count: 5 }
  for (const invalid of [null, [], {}, { ...coverage, record_count: '5' }, { ...coverage, timestamp_count: 6 },
    { ...coverage, category_count: -1 }, { ...coverage, comparison_count: 5 }, { ...coverage, sentiment_count: 0.5 },
    { ...coverage, record_count: Number.MAX_SAFE_INTEGER + 1 }, { ...coverage, sentiment_count: undefined }]) {
    assert.throws(() => analysisSnapshot({ ...ready(), coverage: invalid }), /Invalid analysis coverage/)
  }
})
test('new snapshots replace absent categories rather than retaining stale series', () => {
  let state = analysisSnapshot(ready())
  assert.ok(state.series.tools)
  const next = ready(); next.categories = {}; next.category_counts = { other: 5 }
  state = analysisSnapshot(next)
  assert.equal(state.series.tools, undefined)
  state = analysisSnapshot({ ...next, analysis_status: 'empty' })
  assert.deepEqual(state.series, {})
  assert.deepEqual(state.counts, {})
})
test('failure, insufficient data and warnings never provide chart values', () => {
  for (const status of ['failed', 'unavailable', 'insufficient_data', 'empty']) {
    const state = analysisSnapshot({ ...ready(), analysis_status: status })
    assert.deepEqual(state.counts, {})
    assert.deepEqual(state.series, {})
  }
  assert.equal(analysisSnapshot({ ...ready(), pipeline_warning: 'dependency missing' }).status, 'failed')
  assert.equal(emptyAnalysis().status, 'not_requested')
})
test('incompatible responses fail visibly instead of being interpreted as empty', () => {
  assert.throws(() => analysisSnapshot({ global: { time_series: [] } }))
  assert.throws(() => analysisSnapshot({ ...ready(), category_counts: { tools: -1 } }))
  assert.throws(() => recordsPage({ items: [], total: null }))
})
test('cursor pagination preserves the supplied total, page size and empty snapshot', () => {
  const page = recordsPage({ pagination: 'cursor', items: [{ id: 2 }, { id: 1 }], page: 2, page_size: 50, total: 52, has_more: false, next_cursor: null, snapshot_at: 1000 })
  assert.equal(page.total, 52)
  assert.equal(page.page, 2)
  assert.equal(page.pageSize, 50)
  assert.equal(recordsPage({ pagination: 'cursor', items: [], total: 0, page: 1, page_size: 50, has_more: false, next_cursor: null, snapshot_at: 1000 }).total, 0)
})

const firstRecords = () => recordsPage({ pagination: 'cursor', items: [{ id: 4 }, { id: 3 }], page: 1,
  page_size: 2, total: 4, has_more: true, next_cursor: 'synthetic-next', snapshot_at: 1000 })
const lastRecords = () => recordsPage({ pagination: 'cursor', items: [{ id: 2 }, { id: 1 }], page: 2,
  page_size: 2, total: 4, has_more: false, next_cursor: null, snapshot_at: 1000 })

test('cursor pages retain snapshot identity and terminate only on the server final page', () => {
  const first = firstRecords()
  assert.equal(first.nextCursor, 'synthetic-next')
  assert.equal(first.hasMore, true)
  const complete = mergeRecordsPage(first, lastRecords())
  assert.deepEqual(complete.items.map(item => item.id), [4, 3, 2, 1])
  assert.equal(complete.total, 4)
  assert.equal(complete.snapshotAt, first.snapshotAt)
  assert.equal(complete.hasMore, false)
  assert.equal(complete.nextCursor, null)
})

test('incompatible, malformed and incomplete pagination cannot look like a complete snapshot', () => {
  const valid = { pagination: 'cursor', items: [{ id: 2 }], total: 2, page: 1, page_size: 1,
    has_more: true, next_cursor: 'synthetic-next', snapshot_at: 1000 }
  for (const change of [{ pagination: undefined }, { next_cursor: null }, { next_cursor: '' },
    { has_more: false }, { snapshot_at: Infinity }, { items: [] }, { items: [{ id: null }] },
    { page_size: 201 }, { items: [{ id: 1 }, { id: 2 }], page_size: 2, has_more: false, next_cursor: null },
    { items: [{ id: 2 }, { id: 2 }], page_size: 2, has_more: false, next_cursor: null }]) {
    assert.throws(() => recordsPage({ ...valid, ...change }))
  }
})

test('append rejects different snapshots, skipped pages and duplicate boundaries instead of hiding missing rows', () => {
  for (const change of [{ total: 5 }, { snapshotAt: 2000 }, { pageSize: 3 }, { page: 3 }, { items: [{ id: 3 }, { id: 1 }] }]) {
    assert.throws(() => mergeRecordsPage(firstRecords(), { ...lastRecords(), ...change }))
  }
  assert.throws(() => mergeRecordsPage({ ...firstRecords(), hasMore: false }, lastRecords()))
})
test('raw record links are restricted to HTTP(S)', () => {
  for (const value of ['javascript:alert(1)', 'data:text/html,x', 'file:///c:/test', '/relative', null]) assert.equal(safeHttpUrl(value), null)
  assert.equal(safeHttpUrl('https://example.invalid/page'), 'https://example.invalid/page')
})

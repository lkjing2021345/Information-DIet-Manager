const test = require('node:test');
const assert = require('node:assert/strict');
const { createCollector, STATE_KEY, normalizeItem, normalizeUrl, validateSettings, allowedUrl } = require('../collector-core.js');

function memory(initial = {}) {
  let data = structuredClone(initial);
  return {
    failSet: false,
    async get(keys) { await Promise.resolve(); return structuredClone(Object.fromEntries(keys.map(key => [key, data[key]]))); },
    async set(update) { await Promise.resolve(); if (this.failSet) throw new Error('quota'); Object.assign(data, structuredClone(update)); },
    async remove(keys) { keys.forEach(key => delete data[key]); },
    data: () => structuredClone(data),
  };
}
const record = (id = 1) => ({ url: `https://example.test/article/${id}`, title: `Article ${id}`, text: 'Synthetic article. '.repeat(20), ts: 1790208000000 });
const response = (body = { inserted: 1, duplicates: 0, failed: 0 }, status = 200) => ({ ok: status >= 200 && status < 300, status, text: async () => JSON.stringify(body) });
function fixture(options = {}) {
  const storage = options.storage || memory();
  let id = 0, now = 1790208000000;
  const collector = createCollector({ storage, fetchImpl: async () => response(), uuid: () => `id-${++id}`,
    clock: () => now, random: () => 0, version: 'test', ...options });
  return { collector, storage, advance: ms => { now += ms; } };
}
async function active(options) { const f = fixture(options); await f.collector.initialize(); await f.collector.updateSettings({ enabled: true, apiToken: 'c'.repeat(43) }); return f; }
function gate() { let resolve; return { promise: new Promise(r => { resolve = r; }), resolve: value => resolve(value) }; }
async function until(check) { for (let i = 0; i < 200; i++) { if (check()) return; await new Promise(setImmediate); } throw new Error('condition not reached'); }

test('new install is off; explicit existing choice survives migration and restart', async () => {
  const fresh = fixture(); assert.equal((await fresh.collector.settings()).enabled, false);
  const storage = memory({ idm_settings: { enabled: true }, idm_queue: [record()], idm_stats: { lastSuccessAt: 123 } });
  const f = fixture({ storage }); await f.collector.initialize();
  assert.equal((await f.collector.settings()).enabled, true);
  assert.equal((await f.collector.status()).queueSize, 1);
  assert.equal((await f.collector.status()).stats.lastSuccessAt, null);
  assert.equal(storage.data().idm_queue, undefined);
  assert.equal((await fixture({ storage }).collector.status()).queueSize, 1);
});
test('migration preserves malformed legacy records as blocked, not silently discarded', async () => {
  const f = fixture({ storage: memory({ idm_queue: [record(), { title: 'broken' }] }) });
  const status = await f.collector.status(); assert.equal(status.queueSize, 2); assert.equal(status.blockedCount, 1);
});
test('failed migration keeps the legacy backup', async () => {
  const storage = memory({ idm_queue: [record()] }); storage.failSet = true;
  await assert.rejects(fixture({ storage }).collector.initialize());
  assert.equal(storage.data().idm_queue.length, 1);
});
test('invalid legacy settings migrate to a paused, editable configuration', async () => {
  const f = fixture({ storage: memory({ idm_settings: { enabled: true, endpoint: 'https://remote.test/collect', blockedDomains: 'bad', minTextLength: 200 } }) });
  await f.collector.initialize();
  const settings = await f.collector.settings();
  assert.equal(settings.enabled, false); assert.equal(settings.minTextLength, 200);
  assert.deepEqual(settings.blockedDomains, ['localhost', '127.0.0.1', '::1']);
  assert.equal((await f.collector.status()).stats.lastError, 'migrated_settings_reset');
  await f.collector.updateSettings({ enabled: true }); assert.equal((await f.collector.status()).enabled, true);
});
test('corrupt persisted setting or retry metadata fails closed without overwriting data', async () => {
  const f = await active(); await f.collector.enqueue(record());
  const original = f.storage.data();
  for (const corrupt of [state => { state.settings.enabled = 'false'; }, state => { delete state.queue[0].nextAttemptAt; }]) {
    const broken = structuredClone(original); corrupt(broken[STATE_KEY]);
    const storage = memory(broken), restarted = fixture({ storage });
    await assert.rejects(restarted.collector.enqueue(record(2)), /invalid_storage/);
    await assert.rejects(restarted.collector.flush(), /invalid_storage/);
    assert.deepEqual(storage.data(), broken);
  }
});
test('100 concurrent enqueues persist all unique records', async () => {
  const f = await active();
  const results = await Promise.all(Array.from({ length: 100 }, (_, i) => f.collector.enqueue(record(i))));
  assert.ok(results.every(r => r.ok)); assert.equal((await f.collector.status()).queueSize, 100);
  assert.equal(new Set(f.storage.data()[STATE_KEY].queue.map(entry => entry.id)).size, 100);
});
test('duplicate in-flight page is acknowledged locally without consuming capacity', async () => {
  const { collector } = await active();
  await collector.enqueue(record()); assert.equal((await collector.enqueue(record())).reason, 'duplicate');
  assert.equal((await collector.status()).queueSize, 1);
});
test('capacity rejects newest and counts rejection without evicting old data', async () => {
  const { collector, storage } = await active(); await collector.updateSettings({ maxQueueSize: 1 });
  await collector.enqueue(record(1)); assert.equal((await collector.enqueue(record(2))).reason, 'queue_full');
  assert.equal(storage.data()[STATE_KEY].queue[0].item.url, record(1).url);
  assert.equal((await collector.status()).stats.rejectedFull, 1);
});
test('storage failure cannot acknowledge enqueue success', async () => {
  const f = await active(); f.storage.failSet = true;
  await assert.rejects(f.collector.enqueue(record()));
  f.storage.failSet = false; assert.equal((await f.collector.status()).queueSize, 0);
});
test('singleflight sync and concurrent enqueue do not overwrite one another', async () => {
  const wait = gate(); let calls = 0;
  const f = await active({ batchLimit: 1, fetchImpl: async () => { calls++; return wait.promise; } });
  await f.collector.enqueue(record(1));
  const flush = f.collector.flush(); const joined = f.collector.flush(); assert.equal(flush, joined);
  await until(() => calls === 1); await f.collector.enqueue(record(2)); wait.resolve(response());
  assert.equal((await flush).inserted, 1);
  assert.equal((await f.collector.status()).queueSize, 1);
  assert.equal(f.storage.data()[STATE_KEY].queue[0].item.url, record(2).url);
});
for (const operation of ['clear', 'pause', 'endpoint']) test(`${operation} cancels stale sync writes and prevents the next request`, async () => {
  const wait = gate(); let calls = 0;
  const f = await active({ fetchImpl: async () => { calls++; return wait.promise; } });
  await f.collector.enqueue(record(1)); await f.collector.enqueue(record(2));
  const flush = f.collector.flush(); await until(() => calls === 1);
  if (operation === 'clear') await f.collector.clear();
  else await f.collector.updateSettings(operation === 'pause' ? { enabled: false } : { endpoint: 'http://localhost:8001/collect' });
  wait.resolve(response()); const result = await flush;
  assert.equal(result.cancelled, true); assert.equal(calls, 1);
  assert.equal((await f.collector.status()).queueSize, operation === 'clear' ? 0 : 2);
  assert.equal((await f.collector.status()).stats.lastSuccessAt, null);
});
test('clear while network later fails never resurrects a record', async () => {
  const wait = gate(); let called = false;
  const f = await active({ fetchImpl: async () => { called = true; await wait.promise; throw new Error('offline'); } });
  await f.collector.enqueue(record()); const flush = f.collector.flush(); await until(() => called);
  await f.collector.clear(); wait.resolve(); await flush; assert.equal((await f.collector.status()).queueSize, 0);
});
for (const body of [{}, { inserted: 1, duplicates: 0, failed: 1 }, { inserted: 0, duplicates: 0, failed: 0 }, { inserted: 2, duplicates: 0, failed: 0 }, { inserted: '1', duplicates: 0, failed: 0 }]) {
  test(`malformed or negative acknowledgement retains record: ${JSON.stringify(body)}`, async () => {
    const f = await active({ fetchImpl: async () => response(body) }); await f.collector.enqueue(record());
    assert.equal((await f.collector.flush()).ok, false);
    const status = await f.collector.status(); assert.equal(status.queueSize, 1); assert.equal(status.stats.lastSuccessAt, null);
  });
}
test('duplicate acknowledgement confirms existing page, without claiming an insert', async () => {
  const f = await active({ fetchImpl: async () => response({ inserted: 0, duplicates: 1, failed: 0 }) });
  await f.collector.enqueue(record()); const result = await f.collector.flush();
  assert.equal(result.inserted, 0); assert.equal(result.duplicates, 1); assert.equal(result.queueSize, 0);
});
test('offline retries persist beyond four attempts and survive worker restart', async () => {
  const fetchImpl = async () => { throw new Error('offline'); };
  const f = await active({ fetchImpl }); await f.collector.enqueue(record());
  for (let i = 0; i < 6; i++) { assert.equal((await f.collector.flush({ force: true })).ok, false); }
  const restarted = fixture({ storage: f.storage, fetchImpl });
  assert.equal((await restarted.collector.status()).queueSize, 1);
  assert.equal((await restarted.collector.flush()).attempted, 0);
  assert.equal(f.storage.data()[STATE_KEY].queue[0].attempts, 6);
});
test('request timeout aborts and retains data', async () => {
  const f = await active({ requestTimeoutMs: 10, fetchImpl: async (_url, { signal }) => new Promise((_resolve, reject) => signal.addEventListener('abort', () => reject(new Error('aborted')))) });
  await f.collector.enqueue(record()); assert.equal((await f.collector.flush()).failed, 1);
  assert.equal((await f.collector.status()).queueSize, 1);
});
test('422 is blocked, other records can proceed, manual retry can recover', async () => {
  let bad = true;
  const f = await active({ fetchImpl: async (_url, options) => JSON.parse(options.body).url.endsWith('/1') && bad ? response({}, 422) : response() });
  await f.collector.enqueue(record(1)); await f.collector.enqueue(record(2));
  const result = await f.collector.flush(); assert.equal(result.inserted, 1); assert.equal(result.blocked, 1);
  assert.equal((await f.collector.flush({ force: true })).attempted, 0);
  bad = false; await f.collector.retryBlocked(); assert.equal((await f.collector.flush()).queueSize, 0);
});
test('429 respects persisted backoff and does not hammer remaining queue', async () => {
  let calls = 0;
  const f = await active({ fetchImpl: async () => { calls++; return response({}, 429); } });
  await f.collector.enqueue(record()); await f.collector.flush();
  assert.equal((await f.collector.flush()).attempted, 0); assert.equal(calls, 1);
  f.advance(5001); assert.equal((await f.collector.flush()).attempted, 1);
});
test('lost local deletion after server acknowledgement is safe to retry', async () => {
  const f = await active({ fetchImpl: async () => { f.storage.failSet = true; return response(); } });
  await f.collector.enqueue(record()); await assert.rejects(f.collector.flush()); f.storage.failSet = false;
  assert.equal((await f.collector.status()).queueSize, 1);
  assert.equal((await f.collector.status()).stats.lastSuccessAt, null);
});
test('privacy normalization removes secrets, fragment, credentials and arbitrary metadata', () => {
  const item = normalizeItem({ ...record(), url: 'https://user:pass@example.test/a?token=secret&mode=read#private', meta: { referrer: 'secret' } }, 'v');
  assert.equal(item.url, 'https://example.test/a?mode=read'); assert.deepEqual(item.meta, { pluginVersion: 'v' });
  assert.throws(() => normalizeItem({ ...record(), title: '' }), /invalid_record/);
  assert.throws(() => normalizeUrl('file:///private'), /invalid_url/);
});
test('settings enforce loopback-only ingestion and typed bounds', () => {
  for (const endpoint of ['https://remote.test/collect', 'http://localhost:8000/other', 'http://localhost/collect?token=x', 'http://user:pass@127.0.0.1/collect']) assert.throws(() => validateSettings({ endpoint }));
  for (const patch of [{ enabled: 'true' }, { maxQueueSize: 0 }, { minTextLength: 1 }, { flushIntervalMinutes: 0 }, { unknown: true }]) assert.throws(() => validateSettings(patch));
  assert.equal(validateSettings({ endpoint: 'http://[::1]:8000/collect' }).endpoint, 'http://[::1]:8000/collect');
  const settings = validateSettings({ blockedDomains: ['example.test'] });
  assert.equal(allowedUrl('https://sub.example.test', settings), false);
  assert.equal(allowedUrl('https://sub.example.test.', settings), false);
  assert.equal(allowedUrl('https://example.test', validateSettings({ blockedDomains: ['example.test.'] })), false);
  assert.equal(allowedUrl('https://notexample.test', settings), true);
  assert.equal(allowedUrl('http://127.0.0.1', settings), false);
  for (const url of ['http://127.0.0.2/a', 'http://[::ffff:127.0.0.1]/a', 'http://x.localhost/a', 'http://0.0.0.0/a', 'http://localhost./a', 'http://sub.localhost./a']) {
    assert.equal(allowedUrl(url, validateSettings({ blockedDomains: [] })), false);
  }
});
test('disabled and short-content collection do not enqueue; empty status has no fictitious retry time', async () => {
  const f = fixture(); assert.equal((await f.collector.enqueue(record())).reason, 'disabled');
  await f.collector.updateSettings({ enabled: true });
  assert.equal((await f.collector.enqueue({ ...record(), text: 'short' })).reason, 'filtered');
  assert.equal((await f.collector.status()).nextRetryAt, null);
});
test('unpaired queue is retained without sending a request', async () => {
  let calls = 0;
  const f = await active({ fetchImpl: async () => { calls++; return response(); } });
  await f.collector.updateSettings({ apiToken: '' }); await f.collector.enqueue(record());
  const result = await f.collector.flush();
  assert.equal(result.reason, 'not_paired'); assert.equal(calls, 0); assert.equal(result.queueSize, 1);
});
test('collector sends scoped key in header, hides it from status and recovers after rejected key replacement', async () => {
  let sentKey;
  const f = await active({ fetchImpl: async (_url, options) => {
    sentKey = options.headers.Authorization;
    assert.equal(options.body.includes('c'.repeat(43)), false);
    return sentKey === 'Bearer ' + 'd'.repeat(43) ? response() : response({}, 401);
  } });
  await f.collector.enqueue(record()); assert.equal((await f.collector.flush()).blocked, 1);
  assert.equal(JSON.stringify(await f.collector.status()).includes('c'.repeat(43)), false);
  await f.collector.updateSettings({ apiToken: 'd'.repeat(43) });
  assert.equal((await f.collector.flush()).queueSize, 0);
  assert.equal(sentKey, 'Bearer ' + 'd'.repeat(43));
  await assert.rejects(f.collector.updateSettings({ apiToken: 'short' }), /invalid_token/);
});

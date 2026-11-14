/* Shared by the Manifest V3 worker and isolated Node behavior tests. */
globalThis.IDMCollector = (() => {
  'use strict';
  const STATE_KEY = 'idm_state_v1';
  const LEGACY_KEYS = ['idm_queue', 'idm_settings', 'idm_stats'];
  const DEFAULT_SETTINGS = Object.freeze({
    enabled: false, endpoint: 'http://127.0.0.1:8000/collect',
    apiToken: '',
    flushIntervalMinutes: 1, maxQueueSize: 300, minTextLength: 80,
    blockedDomains: ['localhost', '127.0.0.1', '::1'],
  });
  const PRIVATE_PARAMS = /^(access_token|refresh_token|id_token|token|code|auth|authorization|password|passwd|api_key|apikey|key|session|sessionid|sid)$/i;
  const clone = value => structuredClone(value);
  const text = value => typeof value === 'string' ? value.replace(/\s+/g, ' ').trim() : '';
  function failure(code) { const error = new Error(code); error.code = code; return error; }
  function endpoint(value) {
    let url;
    try { url = new URL(value); } catch { throw failure('invalid_endpoint'); }
    if (!['http:', 'https:'].includes(url.protocol) || !['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)
      || url.username || url.password || url.search || url.hash || url.pathname !== '/collect') throw failure('invalid_endpoint');
    return url.href;
  }
  function normalizeUrl(value) {
    let url;
    try { url = new URL(value); } catch { throw failure('invalid_url'); }
    if (!['http:', 'https:'].includes(url.protocol)) throw failure('invalid_url');
    url.hash = ''; url.username = ''; url.password = '';
    for (const key of [...url.searchParams.keys()]) if (PRIVATE_PARAMS.test(key)) url.searchParams.delete(key);
    if (url.href.length > 2048) throw failure('invalid_url');
    return url.href;
  }
  function validateSettings(patch, current = DEFAULT_SETTINGS) {
    if (!patch || typeof patch !== 'object' || Array.isArray(patch)) throw failure('invalid_settings');
    const result = { ...clone(current) };
    for (const key of Object.keys(patch)) {
      if (!Object.hasOwn(DEFAULT_SETTINGS, key)) throw failure('invalid_settings');
      result[key] = clone(patch[key]);
    }
    if (typeof result.enabled !== 'boolean') throw failure('invalid_settings');
    result.endpoint = endpoint(result.endpoint);
    if (typeof result.apiToken !== 'string' || (result.apiToken && !/^[A-Za-z0-9_-]{43,128}$/.test(result.apiToken))) throw failure('invalid_token');
    for (const [key, min, max] of [['flushIntervalMinutes', 1, 30], ['maxQueueSize', 1, 1000], ['minTextLength', 20, 500]]) {
      if (!Number.isInteger(result[key]) || result[key] < min || result[key] > max) throw failure('invalid_settings');
    }
    if (!Array.isArray(result.blockedDomains) || result.blockedDomains.length > 100) throw failure('invalid_settings');
    result.blockedDomains = [...new Set(result.blockedDomains.map(domain => {
      if (typeof domain !== 'string') throw failure('invalid_settings');
      const clean = domain.trim().toLowerCase().replace(/^\./, '').replace(/\.$/, '');
      if (!clean || clean.length > 253 || !/^[a-z0-9.:-]+$/.test(clean)) throw failure('invalid_settings');
      return clean;
    }))];
    return result;
  }
  function allowedUrl(value, settings) {
    try {
      const url = new URL(normalizeUrl(value));
      const host = url.hostname.toLowerCase().replace(/^\[|\]$/g, '').replace(/\.$/, '');
      if (['localhost', '0.0.0.0', '::1'].includes(host) || host.endsWith('.localhost')
        || /^127\./.test(host) || /^::ffff:7f[0-9a-f]{2}:/.test(host)) return false;
      return !settings.blockedDomains.some(domain => host === domain || host.endsWith('.' + domain));
    } catch { return false; }
  }
  function normalizeItem(raw, version = 'unknown') {
    if (!raw || typeof raw !== 'object' || Array.isArray(raw)) throw failure('invalid_record');
    const url = normalizeUrl(raw.url), title = text(raw.title).slice(0, 300), body = text(raw.text).slice(0, 1000);
    if (!title || !Number.isSafeInteger(raw.ts) || raw.ts < 0 || raw.ts > 253402300799999) throw failure('invalid_record');
    const bounded = (value, size) => text(value).slice(0, size) || null;
    return {
      url, title, text: body, ts: raw.ts, source: 'plugin', lang: bounded(raw.lang, 16),
      channel: bounded(raw.channel, 32) || 'web', author: bounded(raw.author, 120),
      tags: Array.isArray(raw.tags) ? raw.tags.filter(value => typeof value === 'string').slice(0, 10).map(value => text(value).slice(0, 40)) : [],
      // Do not copy arbitrary page-provided metadata, referrer or credentials.
      meta: { pluginVersion: version },
    };
  }
  function fingerprint(item) { return JSON.stringify([item.url, item.title, item.text, Math.floor(item.ts / 60000)]); }
  function acknowledgement(data) {
    return data && ['inserted', 'duplicates', 'failed'].every(key => Number.isInteger(data[key]) && data[key] >= 0)
      && data.failed === 0 && data.inserted + data.duplicates === 1;
  }
  function createCollector({ storage, fetchImpl, clock = Date.now, uuid = () => crypto.randomUUID(),
    random = Math.random, requestTimeoutMs = 10000, batchLimit = 20, batchBudgetMs = 20000, version = 'unknown' }) {
    let serial = Promise.resolve(), flight = null, generation = 0, controller = null;
    function exclusive(task) {
      const result = serial.then(task);
      serial = result.catch(() => {});
      return result;
    }
    async function load() {
      const stored = await storage.get([STATE_KEY, ...LEGACY_KEYS]);
      if (stored[STATE_KEY]) {
        const state = stored[STATE_KEY];
        if (state.version !== 1 || !Array.isArray(state.queue) || !state.settings || !state.stats
          || state.queue.some(entry => !entry || typeof entry.id !== 'string' || !entry.id)
          || new Set(state.queue.map(entry => entry.id)).size !== state.queue.length) throw failure('invalid_storage');
        try { state.settings = validateSettings(state.settings); } catch { throw failure('invalid_storage'); }
        const counter = value => Number.isSafeInteger(value) && value >= 0;
        if (['accepted', 'acknowledged', 'existingPages', 'rejectedFull', 'rejectedInvalid'].some(key => !counter(state.stats[key]))
          || state.queue.some(entry => !counter(entry.attempts) || !counter(entry.nextAttemptAt) || !counter(entry.createdAt)
            || !(entry.blocked === null || typeof entry.blocked === 'string')
            || !(entry.error === null || typeof entry.error === 'string')
            || !(entry.fp === null || typeof entry.fp === 'string'))) throw failure('invalid_storage');
        return state;
      }
      const legacy = stored.idm_settings || {};
      let settings = { ...clone(DEFAULT_SETTINGS) };
      // Preserve valid old choices; invalid values fall back safely and require a fresh opt-in.
      let settingsError = null;
      if (!legacy || typeof legacy !== 'object' || Array.isArray(legacy)) settingsError = 'migrated_settings_reset';
      else for (const key of Object.keys(DEFAULT_SETTINGS)) if (Object.hasOwn(legacy, key)) {
        try { settings = validateSettings({ [key]: legacy[key] }, settings); } catch { settingsError = 'migrated_settings_reset'; }
      }
      if (settingsError) settings.enabled = false;
      const state = { version: 1, settings, queue: [], stats: { ...stored.idm_stats,
        accepted: 0, acknowledged: 0, existingPages: 0, rejectedFull: 0, rejectedInvalid: 0,
        lastSuccessAt: null, lastError: settingsError } };
      if (stored.idm_queue != null && !Array.isArray(stored.idm_queue)) throw failure('invalid_storage');
      for (const old of stored.idm_queue || []) {
        let item, blocked = null;
        try { item = normalizeItem(old, version); } catch { item = old; blocked = 'invalid_record'; }
        state.queue.push({ id: uuid(), item, fp: blocked ? null : fingerprint(item), createdAt: clock(),
          attempts: Number.isSafeInteger(old?._retry) ? Math.max(0, old._retry) : 0, nextAttemptAt: 0, blocked, error: blocked });
      }
      await storage.set({ [STATE_KEY]: state });
      // Migration must be durable before removing old keys. A removal failure leaves a safe duplicate backup.
      if (storage.remove) await storage.remove(LEGACY_KEYS);
      return state;
    }
    function mutate(action) {
      return exclusive(async () => { const state = await load(); const result = action(state); await storage.set({ [STATE_KEY]: state }); return clone(result); });
    }
    function cancel() { generation++; controller?.abort(); }
    async function settings() { return exclusive(async () => clone((await load()).settings)); }
    async function updateSettings(patch) {
      return mutate(state => {
        const next = validateSettings(patch, state.settings);
        if (next.apiToken !== state.settings.apiToken || next.endpoint !== state.settings.endpoint) {
          for (const entry of state.queue) if (['http_401', 'http_403'].includes(entry.blocked)) {
            entry.blocked = null; entry.error = null; entry.nextAttemptAt = 0;
          }
        }
        cancel(); state.settings = next;
        if (['invalid_settings', 'migrated_settings_reset'].includes(state.stats.lastError)) state.stats.lastError = null;
        return next;
      });
    }
    async function enqueue(raw) {
      return mutate(state => {
        if (!state.settings.enabled) return { ok: false, reason: 'disabled' };
        let item;
        try { item = normalizeItem(raw, version); } catch { state.stats.rejectedInvalid++; return { ok: false, reason: 'invalid_record' }; }
        if (!allowedUrl(item.url, state.settings) || item.text.length < state.settings.minTextLength) return { ok: false, reason: 'filtered' };
        const fp = fingerprint(item);
        if (state.queue.some(entry => entry.fp === fp)) return { ok: true, reason: 'duplicate' };
        if (state.queue.length >= state.settings.maxQueueSize) {
          state.stats.rejectedFull++; state.stats.lastError = 'queue_full';
          return { ok: false, reason: 'queue_full' };
        }
        state.queue.push({ id: uuid(), fp, item, attempts: 0, nextAttemptAt: 0, blocked: null, error: null, createdAt: clock() });
        state.stats.accepted++;
        return { ok: true, reason: 'queued', size: state.queue.length };
      });
    }
    async function clear() { return mutate(state => { cancel(); const removed = state.queue.length; state.queue = []; state.stats.lastError = null; return { ok: true, removed }; }); }
    async function retryBlocked() {
      return mutate(state => { cancel(); for (const item of state.queue) { item.blocked = null; item.nextAttemptAt = 0; } return { ok: true }; });
    }
    async function status() {
      return exclusive(async () => {
        const state = await load();
        return clone({ enabled: state.settings.enabled, endpoint: state.settings.endpoint, queueSize: state.queue.length,
          paired: Boolean(state.settings.apiToken),
          blockedCount: state.queue.filter(item => item.blocked).length,
          retryCount: state.queue.filter(item => !item.blocked && item.attempts > 0).length,
          nextRetryAt: state.queue.some(item => !item.blocked)
            ? Math.min(...state.queue.filter(item => !item.blocked).map(item => item.nextAttemptAt)) : null,
          syncing: Boolean(flight), stats: state.stats });
      });
    }
    async function performFlush(force) {
      const startedAt = clock(), epoch = generation, attempted = new Set();
      const result = { ok: true, inserted: 0, duplicates: 0, failed: 0, attempted: 0, cancelled: false };
      await mutate(state => { state.stats.lastFlushAt = clock(); return null; });
      while (result.attempted < batchLimit && clock() - startedAt < batchBudgetMs) {
        const candidate = await mutate(state => {
          if (epoch !== generation || !state.settings.enabled || !state.settings.apiToken) return null;
          endpoint(state.settings.endpoint);
          const entry = state.queue.find(item => !item.blocked && (force || item.nextAttemptAt <= clock()) && !attempted.has(item.id));
          if (!entry) return null;
          entry.attempts++;
          const delay = Math.min(3600000, 5000 * 2 ** Math.min(entry.attempts - 1, 10));
          entry.nextAttemptAt = clock() + delay + Math.floor(random() * delay * 0.2);
          return { entry, endpoint: state.settings.endpoint, apiToken: state.settings.apiToken };
        });
        if (!candidate || epoch !== generation) break;
        attempted.add(candidate.entry.id); result.attempted++;
        let ack = null, error = null, permanent = false;
        const requestController = new AbortController();
        controller = requestController;
        const timer = setTimeout(() => requestController.abort(), requestTimeoutMs);
        try {
          const item = normalizeItem(candidate.entry.item, version);
          const response = await fetchImpl(candidate.endpoint, { method: 'POST', headers: { 'Content-Type': 'application/json', 'Authorization': 'Bearer ' + candidate.apiToken },
            body: JSON.stringify(item), redirect: 'error', credentials: 'omit', signal: controller.signal });
          if (!response.ok) {
            error = `http_${response.status}`;
            permanent = response.status >= 400 && response.status < 500 && ![408, 429].includes(response.status);
          } else {
            const body = await response.text();
            if (body.length > 4096) throw failure('invalid_ack');
            try { ack = JSON.parse(body); } catch { throw failure('invalid_ack'); }
            if (!acknowledgement(ack)) throw failure('invalid_ack');
          }
        } catch (caught) {
          error = ['invalid_record', 'invalid_url', 'invalid_ack'].includes(caught.code) ? caught.code : 'network_error';
          permanent = error === 'invalid_record' || error === 'invalid_url';
        } finally { clearTimeout(timer); controller = null; }
        if (epoch !== generation) { result.cancelled = true; break; }
        await mutate(state => {
          if (epoch !== generation) { result.cancelled = true; return null; }
          const index = state.queue.findIndex(entry => entry.id === candidate.entry.id);
          if (index < 0) return null;
          if (!error && acknowledgement(ack)) {
            state.queue.splice(index, 1);
            result.inserted += ack.inserted; result.duplicates += ack.duplicates;
            state.stats.acknowledged++; state.stats.existingPages += ack.duplicates;
            state.stats.lastSuccessAt = clock();
          } else {
            state.queue[index].error = error; state.queue[index].blocked = permanent ? error : null;
            state.stats.lastError = error; result.failed++;
          }
          return null;
        });
        // A connection outage should not hammer the same unavailable local server for the entire queue.
        if (error === 'network_error' || error === 'http_429' || (error && Number(error.slice(5)) >= 500)) break;
      }
      return mutate(state => {
        result.cancelled ||= epoch !== generation;
        result.queueSize = state.queue.length;
        result.blocked = state.queue.filter(item => item.blocked).length;
        result.ok = result.failed === 0 && !result.cancelled && result.blocked === 0 && state.settings.enabled && Boolean(state.settings.apiToken);
        if (!state.settings.enabled) result.reason = 'disabled';
        else if (!state.settings.apiToken) result.reason = 'not_paired';
        if (result.attempted === 0 && state.queue.length === 0) result.empty = true;
        if (!state.queue.length && !result.failed && !result.cancelled) state.stats.lastError = null;
        state.stats.lastFlushResult = result;
        return result;
      });
    }
    function flush({ force = false } = {}) {
      if (flight) return flight;
      flight = performFlush(force).finally(() => { flight = null; });
      return flight;
    }
    return { initialize: () => exclusive(load), settings, updateSettings, enqueue, clear, retryBlocked, status, flush };
  }
  return { STATE_KEY, DEFAULT_SETTINGS, normalizeUrl, normalizeItem, validateSettings, allowedUrl, acknowledgement, createCollector };
})();
if (typeof module !== 'undefined' && module.exports) module.exports = globalThis.IDMCollector;

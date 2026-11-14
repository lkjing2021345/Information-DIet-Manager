const test = require('node:test');
const assert = require('node:assert/strict');
const vm = require('node:vm');
const fs = require('node:fs');
const path = require('node:path');
const core = require('../collector-core.js');
const source = file => fs.readFileSync(path.join(__dirname, '..', file), 'utf8');
const tick = () => new Promise(setImmediate);

async function worker() {
  let listener, accessLevel, stored = {};
  const chrome = {
    runtime: { id: 'id', getURL: name => `chrome-extension://id/${name}`, getManifest: () => ({ version: 'test' }),
      onMessage: { addListener: value => { listener = value; } }, onInstalled: { addListener() {} }, onStartup: { addListener() {} } },
    alarms: { get: async () => null, create: async () => {}, onAlarm: { addListener() {} } },
    storage: { local: {
      get: async keys => structuredClone(Object.fromEntries(keys.map(key => [key, stored[key]]))),
      set: async value => Object.assign(stored, structuredClone(value)), remove: async keys => keys.forEach(key => delete stored[key]),
      setAccessLevel: async value => { accessLevel = value.accessLevel; },
    } },
    tabs: { get: async () => ({ url: 'https://example.test/article', incognito: false }), query: async () => [], sendMessage: async () => {} },
  };
  vm.runInNewContext(source('background.js'), { chrome, IDMCollector: core, importScripts() {}, fetch, URL });
  const send = (message, sender) => new Promise(resolve => { assert.equal(listener(message, sender, resolve), true); });
  const popup = { id: 'id', url: chrome.runtime.getURL('popup.html') };
  const content = { id: 'id', tab: { id: 1, incognito: false }, frameId: 0, url: 'https://example.test/article' };
  await send({ type: 'GET_STATUS' }, popup);
  return { send, popup, content, accessLevel, setTabUrl: url => { chrome.tabs.get = async () => ({ url, incognito: false }); } };
}
test('worker storage is trusted-only; content gets policy but cannot read/control queue', async () => {
  const w = await worker(); assert.equal(w.accessLevel, 'TRUSTED_CONTEXTS');
  for (const type of ['GET_STATUS', 'GET_SETTINGS', 'UPDATE_SETTINGS', 'CLEAR_QUEUE', 'FLUSH_NOW', 'RETRY_BLOCKED']) {
    assert.equal((await w.send({ type }, w.content)).reason, 'unauthorized');
  }
  const policy = await w.send({ type: 'CAN_COLLECT' }, w.content);
  assert.deepEqual(Object.keys(policy).sort(), ['allowed', 'enabled', 'minTextLength', 'ok']);
});
test('foreign sender, subframe and incognito cannot ingest', async () => {
  const w = await worker();
  for (const sender of [{ ...w.content, id: 'other' }, { ...w.content, frameId: 2 }, { ...w.content, tab: { incognito: true } }, { ...w.popup, url: 'chrome-extension://id/other.html' }]) {
    assert.equal((await w.send({ type: 'INGEST_ITEM' }, sender)).reason, 'unauthorized');
  }
});
test('worker rejects payload origin spoofing and validates popup settings', async () => {
  const w = await worker();
  assert.equal((await w.send({ type: 'UPDATE_SETTINGS', payload: { endpoint: 'https://remote.test/collect' } }, w.popup)).reason, 'invalid_endpoint');
  assert.equal((await w.send({ type: 'UPDATE_SETTINGS', payload: { enabled: true } }, w.popup)).ok, true);
  assert.equal((await w.send({ type: 'INGEST_ITEM', payload: { url: 'https://other.test/' } }, w.content)).reason, 'invalid_url');
});
test('worker uses current browser tab URL after SPA navigation instead of stale sender URL', async () => {
  const w = await worker();
  await w.send({ type: 'UPDATE_SETTINGS', payload: { enabled: true } }, w.popup);
  const payload = { url: 'https://example.test/spa', title: 'SPA', text: 'Synthetic '.repeat(20), ts: 1790208000000 };
  w.setTabUrl(payload.url);
  assert.equal((await w.send({ type: 'INGEST_ITEM', payload }, w.content)).reason, 'queued');
  assert.equal((await w.send({ type: 'INGEST_ITEM', payload: { ...payload, url: w.content.url } }, w.content)).reason, 'invalid_url');
});

function contentFixture(reply) {
  let now = 100000, nextId = 0, clones = 0, mutation;
  const timers = new Map(), intervals = [], requests = [], listeners = {};
  const location = { href: 'https://example.test/first' };
  const cloned = { querySelectorAll: () => [], querySelector: () => ({ textContent: 'Synthetic article. '.repeat(20) }) };
  const document = { visibilityState: 'visible', title: 'Synthetic title', documentElement: { lang: 'en' },
    querySelector: () => null, cloneNode: () => { clones++; return cloned; }, addEventListener: (name, fn) => { listeners[name] = fn; } };
  const chrome = { runtime: { onMessage: { addListener: fn => { listeners.settings = fn; } },
    sendMessage: (message, callback) => { requests.push(message); reply(message, callback); } } };
  class Clock extends Date { static now() { return now; } }
  vm.runInNewContext(source('content.js'), {
    chrome, location, document, navigator: { language: 'en' }, Date: Clock,
    window: { addEventListener: (name, fn) => { listeners[name] = fn; } },
    MutationObserver: class { constructor(callback) { mutation = callback; } observe() {} },
    setTimeout: (fn, delay) => { const id = ++nextId; timers.set(id, { fn, due: now + delay }); return id; },
    clearTimeout: id => timers.delete(id), setInterval: fn => intervals.push(fn),
  });
  return {
    requests, location, clones: () => clones, mutation: () => mutation(), settings: () => listeners.settings({ type: 'COLLECTION_SETTINGS_CHANGED' }),
    poll: () => intervals.forEach(fn => fn()),
    async advance(ms) {
      now += ms;
      for (const [id, timer] of [...timers]) if (timer.due <= now) { timers.delete(id); timer.fn(); }
      await tick();
    },
  };
}
test('static document is scheduled immediately; disabled policy prevents DOM extraction', async () => {
  const c = contentFixture((_message, callback) => callback({ ok: true, enabled: false, allowed: true, minTextLength: 80 }));
  await c.advance(301);
  assert.equal(c.requests.length, 1); assert.equal(c.requests[0].type, 'CAN_COLLECT'); assert.equal(c.clones(), 0);
});
test('rejected enqueue can retry unchanged text; only acknowledged fingerprint suppresses duplicates', async () => {
  let ingests = 0;
  const c = contentFixture((message, callback) => callback(message.type === 'CAN_COLLECT'
    ? { ok: true, enabled: true, allowed: true, minTextLength: 80 }
    : (++ingests === 1 ? { ok: false, reason: 'queue_full' } : { ok: true, reason: 'queued' })));
  await c.advance(301); assert.equal(ingests, 1);
  await c.advance(5001); assert.equal(ingests, 2);
  c.mutation(); await c.advance(5001); assert.equal(ingests, 2);
});
test('slow policy plus SPA navigation retains pending capture after busy finishes', async () => {
  let initialPolicy, calls = 0, ingests = 0;
  const policy = { ok: true, enabled: true, allowed: true, minTextLength: 80 };
  const c = contentFixture((message, callback) => {
    if (message.type === 'CAN_COLLECT') { if (++calls === 1) initialPolicy = callback; else callback(policy); }
    else { ingests++; callback({ ok: true, reason: 'queued' }); }
  });
  await c.advance(301); c.location.href = 'https://example.test/new-spa'; c.poll(); await c.advance(5001);
  initialPolicy(policy); await tick(); await c.advance(301);
  assert.equal(calls, 2); assert.equal(ingests, 1);
  assert.equal(c.requests.find(message => message.type === 'INGEST_ITEM').payload.url, c.location.href);
});
test('continuous DOM mutations cannot indefinitely postpone initial capture', async () => {
  let calls = 0;
  const c = contentFixture((_message, callback) => { calls++; callback({ ok: true, enabled: false, allowed: true }); });
  for (let i = 0; i < 11; i++) { c.mutation(); await c.advance(500); }
  assert.ok(calls >= 1);
});

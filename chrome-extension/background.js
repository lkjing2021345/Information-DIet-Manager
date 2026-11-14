/* Manifest V3 worker: persist state before acknowledging any queue mutation. */
importScripts('collector-core.js');

const collector = IDMCollector.createCollector({
  storage: chrome.storage.local, fetchImpl: fetch,
  version: chrome.runtime.getManifest().version,
});
const ALARM_NAME = 'idm_flush_alarm';
let ready;

async function ensureAlarm() {
  const settings = await collector.settings();
  const period = Number.isInteger(settings.flushIntervalMinutes) && settings.flushIntervalMinutes >= 1
    ? Math.min(30, settings.flushIntervalMinutes) : 1;
  const alarm = await chrome.alarms.get(ALARM_NAME);
  if (!alarm || alarm.periodInMinutes !== period) await chrome.alarms.create(ALARM_NAME, { periodInMinutes: period });
}
function initialize() {
  if (!ready) ready = (async () => {
    await chrome.storage.local.setAccessLevel({ accessLevel: 'TRUSTED_CONTEXTS' });
    await collector.initialize();
    await ensureAlarm();
  })().catch(error => { ready = null; throw error; });
  return ready;
}
function isPopup(sender) {
  return sender.id === chrome.runtime.id && sender.url === chrome.runtime.getURL('popup.html');
}
function isContent(sender) {
  return sender.id === chrome.runtime.id && sender.tab && !sender.tab.incognito
    && sender.frameId === 0 && /^https?:\/\//.test(sender.url || '');
}
async function notifyContent() {
  const tabs = await chrome.tabs.query({});
  await Promise.allSettled(tabs.map(tab => chrome.tabs.sendMessage(tab.id, { type: 'COLLECTION_SETTINGS_CHANGED' })));
}
async function handleMessage(message, sender) {
  if (!message || typeof message.type !== 'string') return { ok: false, reason: 'invalid_message' };
  const contentMessage = ['CAN_COLLECT', 'INGEST_ITEM'].includes(message.type);
  if (contentMessage ? !isContent(sender) : !isPopup(sender)) return { ok: false, reason: 'unauthorized' };
  await initialize();
  if (contentMessage) {
    // sender.url can remain the original document URL after history.pushState.
    // Query the browser's current top-level tab URL (host permissions cover this read).
    const tab = await chrome.tabs.get(sender.tab.id);
    if (tab.incognito || !tab.url) return { ok: false, reason: 'filtered' };
    const currentUrl = tab.url;
    const settings = await collector.settings();
    const allowed = IDMCollector.allowedUrl(currentUrl, settings);
    if (message.type === 'CAN_COLLECT') return { ok: true, enabled: settings.enabled, allowed, minTextLength: settings.minTextLength };
    if (!settings.enabled || !allowed) return { ok: false, reason: settings.enabled ? 'filtered' : 'disabled' };
    // Reject stale/spoofed payloads; a content script cannot label text with another URL.
    try {
      if (IDMCollector.normalizeUrl(message.payload?.url) !== IDMCollector.normalizeUrl(currentUrl)) return { ok: false, reason: 'invalid_url' };
    } catch { return { ok: false, reason: 'invalid_url' }; }
    return collector.enqueue({ ...message.payload, url: currentUrl });
  }
  switch (message.type) {
    case 'GET_STATUS': return { ok: true, ...await collector.status() };
    case 'GET_SETTINGS': return { ok: true, settings: await collector.settings() };
    case 'UPDATE_SETTINGS': {
      const settings = await collector.updateSettings(message.payload);
      await ensureAlarm();
      await notifyContent();
      return { ok: true, settings };
    }
    case 'FLUSH_NOW': return collector.flush({ force: true });
    case 'RETRY_BLOCKED': await collector.retryBlocked(); return collector.flush({ force: true });
    case 'CLEAR_QUEUE': return collector.clear();
    default: return { ok: false, reason: 'invalid_message' };
  }
}
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  handleMessage(message, sender).then(sendResponse).catch(error => {
    const reason = ['invalid_settings', 'invalid_endpoint', 'invalid_storage', 'invalid_token'].includes(error.code) ? error.code : 'storage_error';
    sendResponse({ ok: false, reason });
  });
  return true;
});
chrome.runtime.onInstalled.addListener(() => { initialize().catch(() => {}); });
chrome.runtime.onStartup.addListener(() => { initialize().catch(() => {}); });
chrome.alarms.onAlarm.addListener(alarm => {
  if (alarm.name === ALARM_NAME) initialize().then(() => collector.flush()).catch(() => {});
});
initialize().catch(() => {});

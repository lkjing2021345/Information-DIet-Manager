const DEFAULT_SETTINGS = {
  enabled: true,
  endpoint: "http://127.0.0.1:8000/collect",
  flushIntervalMinutes: 1,
  maxQueueSize: 300,
  minTextLength: 80,
  allowedProtocols: ["http:", "https:"],
  blockedDomains: ["localhost", "127.0.0.1"],
};

const STORAGE_KEYS = {
  queue: "idm_queue",
  settings: "idm_settings",
  stats: "idm_stats",
};

function now() {
  return Date.now();
}

async function getSettings() {
  const data = await chrome.storage.local.get(STORAGE_KEYS.settings);
  return { ...DEFAULT_SETTINGS, ...(data[STORAGE_KEYS.settings] || {}) };
}

async function setSettings(nextSettings) {
  await chrome.storage.local.set({
    [STORAGE_KEYS.settings]: nextSettings,
  });
  return nextSettings;
}

async function getQueue() {
  const data = await chrome.storage.local.get(STORAGE_KEYS.queue);
  return data[STORAGE_KEYS.queue] || [];
}

async function setQueue(queue) {
  await chrome.storage.local.set({
    [STORAGE_KEYS.queue]: queue,
  });
}

async function getStats() {
  const data = await chrome.storage.local.get(STORAGE_KEYS.stats);
  return (
    data[STORAGE_KEYS.stats] || {
      lastFlushAt: null,
      lastSuccessAt: null,
      lastError: null,
      lastFlushResult: null,
    }
  );
}

async function setStats(stats) {
  await chrome.storage.local.set({
    [STORAGE_KEYS.stats]: stats,
  });
}

function normalizeUrl(rawUrl) {
  try {
    const u = new URL(rawUrl);
    u.hash = "";
    return u.toString();
  } catch {
    return rawUrl || "";
  }
}

function stableText(text) {
  return String(text || "")
    .replace(/\s+/g, " ")
    .trim();
}

function minuteBucket(ts) {
  return Math.floor(ts / 60000);
}

function calcFingerprint(item) {
  const url = normalizeUrl(item.url);
  const title = stableText(item.title).slice(0, 120);
  const text = stableText(item.text).slice(0, 200);
  return `${url}::${title}::${text}::${minuteBucket(item.ts)}`;
}

function isAllowedPage(item, settings) {
  try {
    const u = new URL(item.url);
    if (!settings.allowedProtocols.includes(u.protocol)) return false;
    if (settings.blockedDomains.includes(u.hostname)) return false;
  } catch {
    return false;
  }

  if (!item.title || !String(item.title).trim()) return false;

  const text = stableText(item.text || "");
  if (text.length < settings.minTextLength && !item.title) return false;

  return true;
}

function buildChannel(url, meta = {}) {
  const host = (() => {
    try {
      return new URL(url).hostname;
    } catch {
      return "";
    }
  })();

  if (/bilibili\.com/.test(host)) return "video";
  if (/weibo\.com/.test(host)) return "social";
  if (/zhihu\.com/.test(host)) return "social";
  if (/toutiao\.com|thepaper\.cn|ifeng\.com|163\.com|qq\.com|news/.test(host))
    return "news";
  if (meta.pageType) return meta.pageType;
  return "web";
}

function normalizeItem(raw) {
  const title = stableText(raw.title || document?.title || "");
  const text = stableText(raw.text || title).slice(0, 1000);
  const ts = Number(raw.ts || now());

  const item = {
    url: normalizeUrl(raw.url),
    title: title || "Untitled",
    text: text || title || "Untitled",
    ts,
    source: "plugin",
    lang: raw.lang || navigator.language || "zh-CN",
    channel: raw.channel || buildChannel(raw.url, raw.meta),
    author: raw.author || null,
    tags: Array.isArray(raw.tags) ? raw.tags.slice(0, 10) : [],
    meta: {
      ...(raw.meta || {}),
      pluginVersion: chrome.runtime.getManifest().version,
      collectedAt: now(),
    },
  };

  return item;
}

async function enqueueItem(raw) {
  const settings = await getSettings();
  if (!settings.enabled) {
    return { ok: false, reason: "disabled" };
  }

  const item = normalizeItem(raw);
  if (!isAllowedPage(item, settings)) {
    return { ok: false, reason: "filtered" };
  }

  const queue = await getQueue();
  const fp = calcFingerprint(item);

  const duplicated = queue.some((existing) => existing._fp === fp);
  if (duplicated) {
    return { ok: true, reason: "duplicate" };
  }

  const nextQueue = [
    ...queue,
    {
      ...item,
      _fp: fp,
      _retry: 0,
      _createdAt: now(),
    },
  ].slice(-settings.maxQueueSize);

  await setQueue(nextQueue);
  return { ok: true, reason: "queued", size: nextQueue.length };
}

async function flushQueue() {
  const settings = await getSettings();
  const stats = await getStats();

  if (!settings.enabled) {
    const next = {
      ...stats,
      lastFlushAt: now(),
      lastError: "插件已关闭",
    };
    await setStats(next);
    return { ok: false, reason: "disabled" };
  }

  const queue = await getQueue();
  if (!queue.length) {
    const next = {
      ...stats,
      lastFlushAt: now(),
      lastFlushResult: { inserted: 0, duplicates: 0, failed: 0, queueSize: 0 },
    };
    await setStats(next);
    return { ok: true, empty: true };
  }

  let inserted = 0;
  let duplicates = 0;
  let failed = 0;
  const remain = [];

  for (const item of queue) {
    try {
      const payload = {
        url: item.url,
        title: item.title,
        text: item.text,
        ts: item.ts,
        source: "plugin",
        lang: item.lang || null,
        channel: item.channel || null,
        author: item.author || null,
        tags: item.tags || [],
        meta: item.meta || {},
      };

      const res = await fetch(settings.endpoint, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }

      const data = await res.json();
      inserted += Number(data.inserted || 0);
      duplicates += Number(data.duplicates || 0);
      failed += Number(data.failed || 0);
    } catch (err) {
      const retry = Number(item._retry || 0) + 1;
      if (retry <= 3) {
        remain.push({ ...item, _retry: retry });
      } else {
        failed += 1;
      }
    }
  }

  await setQueue(remain);

  const nextStats = {
    ...stats,
    lastFlushAt: now(),
    lastSuccessAt: now(),
    lastError: remain.length ? "部分记录发送失败，已保留待重试" : null,
    lastFlushResult: {
      inserted,
      duplicates,
      failed,
      queueSize: remain.length,
    },
  };

  await setStats(nextStats);
  return { ok: true, ...nextStats.lastFlushResult };
}

async function buildStatus() {
  const queue = await getQueue();
  const settings = await getSettings();
  const stats = await getStats();

  return {
    enabled: settings.enabled,
    endpoint: settings.endpoint,
    queueSize: queue.length,
    stats,
  };
}

chrome.runtime.onInstalled.addListener(async () => {
  const current = await getSettings();
  await setSettings(current);
  chrome.alarms.create("idm_flush_alarm", {
    periodInMinutes: current.flushIntervalMinutes,
  });
});

chrome.runtime.onStartup.addListener(async () => {
  const current = await getSettings();
  chrome.alarms.create("idm_flush_alarm", {
    periodInMinutes: current.flushIntervalMinutes,
  });
});

chrome.alarms.onAlarm.addListener(async (alarm) => {
  if (alarm.name === "idm_flush_alarm") {
    await flushQueue();
  }
});

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  (async () => {
    switch (message.type) {
      case "INGEST_ITEM": {
        const result = await enqueueItem(message.payload);
        sendResponse(result);
        break;
      }
      case "GET_STATUS": {
        sendResponse(await buildStatus());
        break;
      }
      case "FLUSH_NOW": {
        sendResponse(await flushQueue());
        break;
      }
      case "CLEAR_QUEUE": {
        await setQueue([]);
        sendResponse({ ok: true });
        break;
      }
      case "GET_SETTINGS": {
        sendResponse(await getSettings());
        break;
      }
      case "UPDATE_SETTINGS": {
        const oldSettings = await getSettings();
        const next = { ...oldSettings, ...(message.payload || {}) };
        await setSettings(next);
        chrome.alarms.create("idm_flush_alarm", {
          periodInMinutes: next.flushIntervalMinutes,
        });
        sendResponse({ ok: true, settings: next });
        break;
      }
      case "TOGGLE_ENABLED": {
        const oldSettings = await getSettings();
        const next = { ...oldSettings, enabled: !oldSettings.enabled };
        await setSettings(next);
        sendResponse({ ok: true, enabled: next.enabled });
        break;
      }
      default:
        sendResponse({ ok: false, error: "unknown_message_type" });
    }
  })();

  return true;
});

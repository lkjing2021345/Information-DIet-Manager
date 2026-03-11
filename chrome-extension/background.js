const QUEUE_KEY = "idm_ingest_queue";
const SETTINGS_KEY = "idm_settings";
const LAST_SENT_BY_TAB_KEY = "idm_last_sent_by_tab";

const DEFAULT_SETTINGS = {
  ingestEndpoint: "http://127.0.0.1:8000/collect",
  flushIntervalMin: 1,
  maxBatchSize: 100,
  maxQueueSize: 5000,
};

function isWebUrl(url) {
  return /^https?:\/\//i.test(url || "");
}

function clamp(str, maxLen) {
  const s = (str || "").toString();
  return s.length > maxLen ? s.slice(0, maxLen) : s;
}

async function getLocal(key, fallback) {
  const obj = await chrome.storage.local.get(key);
  return obj[key] ?? fallback;
}

async function setLocal(key, value) {
  await chrome.storage.local.set({ [key]: value });
}

async function getSettings() {
  const s = await getLocal(SETTINGS_KEY, {});
  return { ...DEFAULT_SETTINGS, ...s };
}

function normalizeItem(raw) {
  // 严格对齐 schema
  const item = {
    url: raw?.url || "",
    title: clamp(raw?.title || "", 300),
    text: clamp(raw?.text || "", 2000),
    ts: Number(raw?.ts || Date.now()),
    source: raw?.source === "import" ? "import" : "plugin",
  };

  if (raw?.lang) item.lang = clamp(raw.lang, 16);
  if (raw?.channel) item.channel = clamp(raw.channel, 32);
  if (raw?.author) item.author = clamp(raw.author, 120);
  if (Array.isArray(raw?.tags)) {
    item.tags = raw.tags
      .map((t) => clamp(String(t), 40))
      .filter(Boolean)
      .slice(0, 20);
  }
  if (raw?.meta && typeof raw.meta === "object") item.meta = raw.meta;

  // 最小校验
  if (!isWebUrl(item.url)) return null;
  if (!item.title.trim()) return null;
  if (!Number.isInteger(item.ts)) item.ts = Date.now();
  return item;
}

function fingerprint(item) {
  // 简单去重：url + title + 分钟
  const minBucket = Math.floor(item.ts / 60000);
  return `${item.url}|${item.title}|${minBucket}`;
}

// 1) 页面完成后注入
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status !== "complete" || !isWebUrl(tab.url)) return;

  try {
    await chrome.scripting.executeScript({
      target: { tabId },
      files: ["readability.js", "content.js"],
    });
  } catch (e) {
    console.warn("[IDM] inject failed:", e?.message || e);
  }
});

// 2) 收到 item -> 入队
chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg?.type !== "INGEST_ITEM") return;

  (async () => {
    const settings = await getSettings();
    const queue = await getLocal(QUEUE_KEY, []);
    const lastByTab = await getLocal(LAST_SENT_BY_TAB_KEY, {});

    const item = normalizeItem(msg.payload);
    if (!item) {
      sendResponse({ ok: false, reason: "invalid_item" });
      return;
    }

    // 同 tab 同 url 短时间重复抑制
    const tabId = sender?.tab?.id;
    if (typeof tabId === "number") {
      const last = lastByTab[tabId];
      if (
        last &&
        last.url === item.url &&
        Math.abs(item.ts - last.ts) < 10000
      ) {
        sendResponse({ ok: true, skipped: "duplicate_tab_event" });
        return;
      }
      lastByTab[tabId] = { url: item.url, ts: item.ts };
      await setLocal(LAST_SENT_BY_TAB_KEY, lastByTab);
    }

    const fp = fingerprint(item);
    const exists = queue.some((q) => fingerprint(q) === fp);
    if (!exists) queue.push(item);

    if (queue.length > settings.maxQueueSize) {
      queue.splice(0, queue.length - settings.maxQueueSize); // 丢最旧
    }

    await setLocal(QUEUE_KEY, queue);
    sendResponse({ ok: true, queued: queue.length });
  })();

  return true;
});

async function flushQueue() {
  const settings = await getSettings();
  const queue = await getLocal(QUEUE_KEY, []);
  if (!queue.length) return;

  const batch = queue.slice(0, settings.maxBatchSize);
  let sentCount = 0;

  for (const item of batch) {
    try {
      const resp = await fetch(settings.ingestEndpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(item), // 单条对象
      });

      if (!resp.ok) {
        const txt = await resp.text();
        console.warn("[IDM] flush failed:", resp.status, txt);
        break; // 停止本轮，避免疯狂请求
      }

      sentCount += 1;
    } catch (e) {
      console.warn("[IDM] flush error:", e?.message || e);
      break;
    }
  }

  if (sentCount > 0) {
    const rest = queue.slice(sentCount);
    await setLocal(QUEUE_KEY, rest);
    console.log(`[IDM] flushed ${sentCount}, remain ${rest.length}`);
  }
}

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === "idm_flush") flushQueue();
});

async function ensureAlarm() {
  const settings = await getSettings();
  await chrome.alarms.create("idm_flush", {
    periodInMinutes: settings.flushIntervalMin,
  });
}

chrome.runtime.onInstalled.addListener(ensureAlarm);
chrome.runtime.onStartup.addListener(ensureAlarm);

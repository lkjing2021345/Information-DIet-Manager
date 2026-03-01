// background.js — Service worker for Information Diet Manager Chrome Extension
// Tracks tab activation times to compute dwell time, then sends data to backend.

const DEFAULT_API = "http://127.0.0.1:8000";
const MIN_DWELL_MS = 2000;

// In-memory map: tabId -> { url, title, activatedAt }
const activeTabs = new Map();

function getApiBase() {
  return new Promise((resolve) => {
    chrome.storage.local.get({ apiBase: DEFAULT_API, enabled: true }, (cfg) => {
      resolve(cfg);
    });
  });
}

async function sendItem(payload) {
  const { apiBase, enabled } = await getApiBase();
  if (!enabled) return;

  const endpoint = apiBase.replace(/\/+$/, "") + "/collect";
  try {
    const resp = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (!resp.ok) {
      console.warn("[IDM] Backend returned", resp.status);
    }
  } catch (err) {
    console.warn("[IDM] Failed to send data:", err.message);
  }
}

function flushTab(tabId) {
  const entry = activeTabs.get(tabId);
  if (!entry) return;
  activeTabs.delete(tabId);

  const dwellMs = Date.now() - entry.activatedAt;
  // Ignore very short visits to reduce noise
  if (dwellMs < MIN_DWELL_MS) return;

  // Skip browser internal pages
  if (!entry.url || !/^https?:\/\//.test(entry.url)) return;

  const payload = {
    url: entry.url,
    title: entry.title || "(no title)",
    text: entry.text || null,
    ts: entry.activatedAt,
    source: "plugin",
    meta: {
      dwell_ms: dwellMs,
    },
  };
  sendItem(payload);
}

// Track tab activation
chrome.tabs.onActivated.addListener(async (activeInfo) => {
  // Flush previously active tab in this window
  for (const [tid, data] of activeTabs) {
    if (data.windowId === activeInfo.windowId) {
      flushTab(tid);
    }
  }

  try {
    const tab = await chrome.tabs.get(activeInfo.tabId);
    activeTabs.set(activeInfo.tabId, {
      url: tab.url,
      title: tab.title,
      text: null,
      activatedAt: Date.now(),
      windowId: activeInfo.windowId,
    });
  } catch {
    // Tab may have been closed already
  }
});

// Update URL/title when navigation completes in active tab
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  const entry = activeTabs.get(tabId);
  if (!entry) return;
  if (changeInfo.url) entry.url = changeInfo.url;
  if (changeInfo.title) entry.title = changeInfo.title;
});

// Flush when tab is closed
chrome.tabs.onRemoved.addListener((tabId) => {
  flushTab(tabId);
});

// Listen for content script messages (page text extraction)
chrome.runtime.onMessage.addListener((message, sender) => {
  if (message.type === "PAGE_TEXT" && sender.tab) {
    const entry = activeTabs.get(sender.tab.id);
    if (entry) {
      entry.text = message.text || null;
    }
  }
});

// background.js — 信息饮食管理器 Chrome 插件 Service Worker
// 跟踪标签页激活时间以计算停留时长，然后将数据发送到后端。

const DEFAULT_API = "http://127.0.0.1:8000";
const MIN_DWELL_MS = 2000;

// 内存中的映射表：tabId -> { url, title, activatedAt }
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
  // 忽略停留时间过短的访问，减少噪声数据
  if (dwellMs < MIN_DWELL_MS) return;

  // 跳过浏览器内部页面
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

// 跟踪标签页激活事件
chrome.tabs.onActivated.addListener(async (activeInfo) => {
  // 刷新当前窗口中先前活跃的标签页
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
    // 标签页可能已经关闭
  }
});

// 当活跃标签页导航完成时更新 URL/标题
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  const entry = activeTabs.get(tabId);
  if (!entry) return;
  if (changeInfo.url) entry.url = changeInfo.url;
  if (changeInfo.title) entry.title = changeInfo.title;
});

// 标签页关闭时刷新数据
chrome.tabs.onRemoved.addListener((tabId) => {
  flushTab(tabId);
});

// 监听 Content Script 消息（页面文本提取）
chrome.runtime.onMessage.addListener((message, sender) => {
  if (message.type === "PAGE_TEXT" && sender.tab) {
    const entry = activeTabs.get(sender.tab.id);
    if (entry) {
      entry.text = message.text || null;
    }
  }
});

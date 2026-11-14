(() => {
  'use strict';
  let lastAcknowledged = '', busy = false, pending = false, timer = null, firstScheduled = 0;
  let lastAttempt = 0, retryDelay = 5000, observedUrl = location.href;
  const clean = value => String(value || '').replace(/\s+/g, ' ').trim();
  function message(value) {
    return new Promise(resolve => {
      try {
        chrome.runtime.sendMessage(value, response => {
          resolve(chrome.runtime.lastError ? null : response);
        });
      } catch { resolve(null); }
    });
  }
  function schedule(delay = 1200) {
    if (document.visibilityState !== 'visible') return;
    const now = Date.now();
    if (!firstScheduled) firstScheduled = now;
    clearTimeout(timer);
    // Debounce a changing DOM, but do not starve indefinitely on busy pages.
    const due = Math.max(lastAttempt + 5000, Math.min(now + delay, firstScheduled + 5000));
    timer = setTimeout(capture, Math.max(0, due - now));
  }
  function extract() {
    // Skip password pages and remove form/editable content before extracting text.
    if (document.querySelector('input[type="password"]')) return null;
    const cloned = document.cloneNode(true);
    cloned.querySelectorAll('form,input,textarea,select,[contenteditable],[role="textbox"],script,style,noscript').forEach(node => node.remove());
    let article;
    try { if (typeof Readability !== 'undefined') article = new Readability(cloned.cloneNode(true)).parse(); } catch { /* bounded fallback below */ }
    const fallback = cloned.querySelector('article,main,[role="main"]') || cloned.body;
    return {
      url: location.href, title: clean(article?.title || document.title).slice(0, 300),
      text: clean(article?.textContent || fallback?.textContent).slice(0, 1000), ts: Date.now(),
      lang: document.documentElement.lang || navigator.language || 'zh-CN', channel: 'web',
    };
  }
  async function capture() {
    timer = null; firstScheduled = 0;
    if (busy) { pending = true; return; }
    if (document.visibilityState !== 'visible') return;
    busy = true; lastAttempt = Date.now();
    let retry = false;
    const url = location.href;
    try {
      const policy = await message({ type: 'CAN_COLLECT' });
      if (!policy?.ok) { retry = true; return; }
      // Ask the worker before reading page content, even on an initial static document.
      if (!policy.enabled || !policy.allowed || url !== location.href || document.visibilityState !== 'visible') return;
      const payload = extract();
      if (!payload?.title || payload.text.length < policy.minTextLength) return;
      const fp = JSON.stringify([payload.url.split('#')[0], payload.title, payload.text]);
      if (fp === lastAcknowledged) return;
      const response = await message({ type: 'INGEST_ITEM', payload });
      if (response?.ok && ['queued', 'duplicate'].includes(response.reason)) {
        if (url === location.href) lastAcknowledged = fp;
        retryDelay = 5000;
      } else retry = !['disabled', 'filtered', 'invalid_record', 'invalid_url'].includes(response?.reason);
    } finally {
      busy = false;
      if (pending || url !== location.href) {
        pending = false; schedule(300);
      } else if (retry) {
        clearTimeout(timer); firstScheduled = 0;
        timer = setTimeout(capture, retryDelay);
        retryDelay = Math.min(60000, retryDelay * 2);
      }
    }
  }
  chrome.runtime.onMessage.addListener(message => {
    if (message?.type === 'COLLECTION_SETTINGS_CHANGED') schedule(0);
  });
  document.addEventListener('visibilitychange', () => { if (document.visibilityState === 'visible') schedule(300); });
  window.addEventListener('pageshow', () => schedule(300));
  window.addEventListener('popstate', () => schedule(300));
  new MutationObserver(() => schedule()).observe(document.documentElement, { childList: true, subtree: true, characterData: true });
  // Content scripts run in an isolated world: patching history here does not intercept page-world SPA navigation.
  setInterval(() => {
    if (document.visibilityState === 'visible' && observedUrl !== location.href) {
      observedUrl = location.href; schedule(300);
    }
  }, 1000);
  schedule(300);
})();

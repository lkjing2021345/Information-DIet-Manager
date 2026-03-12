(() => {
  let lastSentFingerprint = "";
  let routeChangeTimer = null;

  function stableText(text) {
    return String(text || "")
      .replace(/\s+/g, " ")
      .trim();
  }

  function detectLang() {
    return document.documentElement.lang || navigator.language || "zh-CN";
  }

  function extractMeta(name, attr = "name") {
    const el = document.querySelector(`meta[${attr}="${name}"]`);
    return el ? el.getAttribute("content") : "";
  }

  function extractTextFallback() {
    const candidates = [
      document.querySelector("article"),
      document.querySelector("main"),
      document.querySelector("[role='main']"),
      document.querySelector(".article"),
      document.querySelector(".content"),
      document.body,
    ].filter(Boolean);

    for (const node of candidates) {
      const text = stableText(node.innerText || "");
      if (text.length > 120) {
        return text.slice(0, 1000);
      }
    }

    return stableText(document.body?.innerText || "").slice(0, 1000);
  }

  function extractTags() {
    const keywords = extractMeta("keywords");
    if (!keywords) return [];
    return keywords
      .split(/[，,|]/)
      .map((x) => x.trim())
      .filter(Boolean)
      .slice(0, 10);
  }

  function extractAuthor() {
    return (
      extractMeta("author") || extractMeta("article:author", "property") || ""
    );
  }

  function detectPageType() {
    const host = location.hostname;
    if (/bilibili\.com/.test(host)) return "video";
    if (/weibo\.com/.test(host)) return "social";
    if (/zhihu\.com/.test(host)) return "social";
    if (/news|163\.com|qq\.com|ifeng\.com|thepaper\.cn/.test(host))
      return "news";
    return "web";
  }

  function extractReadable() {
    try {
      if (typeof Readability !== "undefined") {
        const cloned = document.cloneNode(true);
        const article = new Readability(cloned).parse();
        if (article && stableText(article.textContent).length > 80) {
          return {
            title: stableText(article.title || document.title),
            text: stableText(article.textContent).slice(0, 1000),
          };
        }
      }
    } catch (e) {}

    return {
      title: stableText(document.title),
      text: extractTextFallback(),
    };
  }

  function buildPayload() {
    const readable = extractReadable();
    const title =
      readable.title ||
      extractMeta("og:title", "property") ||
      extractMeta("twitter:title", "name") ||
      stableText(document.title);

    const text =
      readable.text ||
      extractMeta("description") ||
      extractMeta("og:description", "property") ||
      extractTextFallback();

    return {
      url: location.href,
      title,
      text,
      ts: Date.now(),
      lang: detectLang(),
      author: extractAuthor() || null,
      tags: extractTags(),
      channel: detectPageType(),
      meta: {
        host: location.hostname,
        pathname: location.pathname,
        referrer: document.referrer || "",
        pageType: detectPageType(),
      },
    };
  }

  function fingerprint(payload) {
    return [
      payload.url.split("#")[0],
      stableText(payload.title).slice(0, 120),
      stableText(payload.text).slice(0, 120),
    ].join("::");
  }

  function sendPayload() {
    const payload = buildPayload();

    if (!payload.title && !payload.text) return;
    if ((payload.text || "").length < 30 && !payload.title) return;

    const fp = fingerprint(payload);
    if (fp === lastSentFingerprint) return;
    lastSentFingerprint = fp;

    chrome.runtime.sendMessage(
      {
        type: "INGEST_ITEM",
        payload,
      },
      () => void chrome.runtime.lastError,
    );
  }

  function scheduleSend(delay = 1200) {
    clearTimeout(routeChangeTimer);
    routeChangeTimer = setTimeout(sendPayload, delay);
  }

  const originalPushState = history.pushState;
  const originalReplaceState = history.replaceState;

  history.pushState = function (...args) {
    originalPushState.apply(this, args);
    scheduleSend(800);
  };

  history.replaceState = function (...args) {
    originalReplaceState.apply(this, args);
    scheduleSend(800);
  };

  window.addEventListener("popstate", () => scheduleSend(800));
  window.addEventListener("load", () => scheduleSend(1200));
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") {
      scheduleSend(600);
    }
  });

  const observer = new MutationObserver(() => {
    scheduleSend(1500);
  });

  observer.observe(document.documentElement, {
    childList: true,
    subtree: true,
  });
})();

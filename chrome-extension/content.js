(() => {
  if (window.__IDM_COLLECTED__) return;
  window.__IDM_COLLECTED__ = true;

  function clamp(str, maxLen) {
    const s = (str || "").trim();
    return s.length > maxLen ? s.slice(0, maxLen) : s;
  }

  function getMetaMap() {
    const metas = document.querySelectorAll("meta");
    const result = {};
    metas.forEach((m) => {
      const k = m.getAttribute("name") || m.getAttribute("property");
      const v = m.getAttribute("content");
      if (k && v) result[k.toLowerCase()] = v;
    });
    return result;
  }

  function parseArticleText() {
    try {
      const article = new Readability(document.cloneNode(true)).parse();
      return {
        text: article?.textContent || "",
        byline: article?.byline || "",
        siteName: article?.siteName || "",
        excerpt: article?.excerpt || "",
        publishedTime: article?.publishedTime || "",
      };
    } catch (e) {
      return {
        text: "",
        byline: "",
        siteName: "",
        excerpt: "",
        publishedTime: "",
      };
    }
  }

  function inferChannel(url, meta) {
    const host = new URL(url).hostname.toLowerCase();
    const full =
      `${host} ${meta["og:type"] || ""} ${meta["keywords"] || ""}`.toLowerCase();

    if (/twitter|x\.com|weibo|zhihu|reddit|facebook|instagram/.test(full))
      return "social";
    if (/youtube|bilibili|douyin|tiktok/.test(full)) return "short_video";
    if (/news|bbc|cnn|nytimes|xinhuanet|people\.com/.test(full)) return "news";
    return "other";
  }

  function splitTags(meta) {
    const raw = meta["keywords"] || "";
    return raw
      .split(/[,，;；]/)
      .map((s) => s.trim())
      .filter(Boolean)
      .slice(0, 20)
      .map((s) => s.slice(0, 40));
  }

  const meta = getMetaMap();
  const article = parseArticleText();

  const url = location.href;
  const title = clamp(
    document.title ||
      meta["og:title"] ||
      meta["twitter:title"] ||
      location.hostname,
    300,
  );

  const text = clamp(
    article.text ||
      meta["description"] ||
      meta["og:description"] ||
      article.excerpt ||
      "",
    2000,
  );

  const item = {
    url,
    title: title || location.hostname,
    text, // 可空
    ts: Date.now(),
    source: "plugin",
    lang: clamp(document.documentElement.lang || "", 16) || undefined,
    channel: clamp(inferChannel(url, meta), 32),
    author: clamp(meta["author"] || article.byline || "", 120) || undefined,
    tags: splitTags(meta),
    meta: {
      referrer: document.referrer || "",
      site_name: article.siteName || meta["og:site_name"] || "",
      published_time:
        article.publishedTime || meta["article:published_time"] || "",
      og_type: meta["og:type"] || "",
    },
  };

  // 去掉 undefined 字段，避免脏数据
  Object.keys(item).forEach((k) => item[k] === undefined && delete item[k]);

  chrome.runtime.sendMessage({ type: "INGEST_ITEM", payload: item }, () => {
    // 忽略 callback 错误，后台会记录
  });
})();

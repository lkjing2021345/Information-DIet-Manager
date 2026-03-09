// content.js — 提取页面 meta description 和截断后的正文，
// 然后发送给 background service worker。

(function () {
  "use strict";

  const MAX_TEXT_LENGTH = 2000;

  function getMeta(name) {
    const el =
      document.querySelector('meta[name="' + name + '"]') ||
      document.querySelector('meta[property="og:' + name + '"]');
    return el ? el.getAttribute("content") || "" : "";
  }

  function getBodyText() {
    const body = document.body;
    if (!body) return "";
    // 克隆 DOM 以避免修改真实页面
    const clone = body.cloneNode(true);
    // 移除 script、style、noscript、nav、footer、header 等元素
    const removeTags = ["script", "style", "noscript", "nav", "footer", "header"];
    removeTags.forEach((tag) => {
      clone.querySelectorAll(tag).forEach((el) => el.remove());
    });
    const raw = clone.innerText || clone.textContent || "";
    // 合并连续空白字符
    return raw.replace(/\s+/g, " ").trim();
  }

  try {
    const description = getMeta("description");
    const bodySnippet = getBodyText().slice(0, MAX_TEXT_LENGTH);
    const text = description || bodySnippet;

    chrome.runtime.sendMessage({
      type: "PAGE_TEXT",
      text: text,
    });
  } catch {
    // 在受限页面中静默忽略错误
  }
})();

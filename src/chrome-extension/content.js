// content.js — Extracts meta description and truncated body text,
// then sends to the background service worker.

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
    // Clone to avoid modifying the real DOM
    const clone = body.cloneNode(true);
    // Remove script, style, noscript, nav, footer, header elements
    const removeTags = ["script", "style", "noscript", "nav", "footer", "header"];
    removeTags.forEach((tag) => {
      clone.querySelectorAll(tag).forEach((el) => el.remove());
    });
    const raw = clone.innerText || clone.textContent || "";
    // Collapse whitespace
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
    // Silently ignore errors in restricted pages
  }
})();

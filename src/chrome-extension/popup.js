// popup.js — 信息饮食管理器插件的设置界面

(function () {
  "use strict";

  const DEFAULT_API = "http://127.0.0.1:8000";

  const apiInput = document.getElementById("apiBase");
  const enabledBox = document.getElementById("enabled");
  const saveBtn = document.getElementById("save");
  const statusEl = document.getElementById("status");

  // 加载已保存的设置
  chrome.storage.local.get({ apiBase: DEFAULT_API, enabled: true }, (cfg) => {
    apiInput.value = cfg.apiBase;
    enabledBox.checked = cfg.enabled;
  });

  saveBtn.addEventListener("click", () => {
    const apiBase = apiInput.value.trim() || DEFAULT_API;
    const enabled = enabledBox.checked;
    chrome.storage.local.set({ apiBase, enabled }, () => {
      statusEl.textContent = "设置已保存 ✓";
      setTimeout(() => {
        statusEl.textContent = "";
      }, 2000);
    });
  });
})();

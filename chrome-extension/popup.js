function fmt(ts) {
  if (!ts) return "-";
  const d = new Date(ts);
  return d.toLocaleString();
}

function sendMessage(type, payload) {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage({ type, payload }, (res) => {
      resolve(res);
    });
  });
}

async function refresh() {
  const status = await sendMessage("GET_STATUS");

  document.getElementById("queueSize").textContent = status.queueSize ?? "-";
  document.getElementById("lastFlushAt").textContent = fmt(
    status.stats?.lastFlushAt,
  );
  document.getElementById("lastSuccessAt").textContent = fmt(
    status.stats?.lastSuccessAt,
  );

  const toggleBtn = document.getElementById("toggleBtn");
  toggleBtn.textContent = status.enabled ? "已开启" : "已关闭";

  const settings = await sendMessage("GET_SETTINGS");
  document.getElementById("endpoint").value = settings.endpoint || "";
  document.getElementById("flushIntervalMinutes").value =
    settings.flushIntervalMinutes || 1;
  document.getElementById("minTextLength").value = settings.minTextLength || 80;

  const resultBox = document.getElementById("result");
  resultBox.textContent = status.stats?.lastError || "";
}

document.getElementById("toggleBtn").addEventListener("click", async () => {
  await sendMessage("TOGGLE_ENABLED");
  await refresh();
});

document.getElementById("saveBtn").addEventListener("click", async () => {
  const endpoint = document.getElementById("endpoint").value.trim();
  const flushIntervalMinutes = Number(
    document.getElementById("flushIntervalMinutes").value || 1,
  );
  const minTextLength = Number(
    document.getElementById("minTextLength").value || 80,
  );

  await sendMessage("UPDATE_SETTINGS", {
    endpoint,
    flushIntervalMinutes,
    minTextLength,
  });

  document.getElementById("result").textContent = "配置已保存";
  await refresh();
});

document.getElementById("flushBtn").addEventListener("click", async () => {
  const res = await sendMessage("FLUSH_NOW");
  document.getElementById("result").textContent =
    `同步完成：inserted=${res.inserted || 0}, duplicates=${res.duplicates || 0}, failed=${res.failed || 0}`;
  await refresh();
});

document.getElementById("clearBtn").addEventListener("click", async () => {
  await sendMessage("CLEAR_QUEUE");
  document.getElementById("result").textContent = "队列已清空";
  await refresh();
});

refresh();

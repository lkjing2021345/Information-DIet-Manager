function fmt(ts) {
  if (!ts) return "-";
  const d = new Date(ts);
  return d.toLocaleString();
}

function sendMessage(type, payload) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage({ type, payload }, (res) => {
      if (chrome.runtime.lastError) {
        reject(new Error(chrome.runtime.lastError.message));
        return;
      }
      resolve(res);
    });
  });
}

function setResult(message) {
  const el = document.getElementById("result");
  el.textContent = message || "";
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

  if (status.stats?.lastError) {
    setResult(status.stats.lastError);
  }
}

document.getElementById("toggleBtn").addEventListener("click", async () => {
  try {
    await sendMessage("TOGGLE_ENABLED");
    await refresh();
    setResult("插件状态已切换");
  } catch (err) {
    setResult(`切换失败：${err.message}`);
  }
});

document.getElementById("saveBtn").addEventListener("click", async () => {
  try {
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

    await refresh();
    setResult("配置已保存");
  } catch (err) {
    setResult(`保存失败：${err.message}`);
  }
});

document.getElementById("flushBtn").addEventListener("click", async () => {
  try {
    const res = await sendMessage("FLUSH_NOW");
    await refresh();

    if (res?.empty) {
      setResult("队列为空，没有需要同步的数据");
      return;
    }

    if (!res?.ok) {
      setResult(`同步失败：${res?.reason || "未知错误"}`);
      return;
    }

    setResult(
      `同步完成：inserted=${res.inserted || 0}, duplicates=${res.duplicates || 0}, failed=${res.failed || 0}`,
    );
  } catch (err) {
    setResult(`同步失败：${err.message}`);
  }
});

document.getElementById("clearBtn").addEventListener("click", async () => {
  try {
    await sendMessage("CLEAR_QUEUE");
    await refresh();
    setResult("队列已清空");
  } catch (err) {
    setResult(`清空失败：${err.message}`);
  }
});

refresh().catch((err) => {
  setResult(`初始化失败：${err.message}`);
});

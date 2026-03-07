chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {

  // 页面加载完成才执行
  if (changeInfo.status === "complete" && tab.url?.startsWith("http")) {

    console.log("Injecting content script:", tab.url);

    chrome.scripting.executeScript({
      target: { tabId: tabId },
      files: ["readability.js", "content.js"]
    });
  }
});
(function () {
  console.log("Information Diet Collector running...");

  function getMeta() {
    const metas = document.querySelectorAll("meta");
    const result = {};

    metas.forEach((m) => {
      const name = m.getAttribute("name") || m.getAttribute("property");

      const content = m.getAttribute("content");

      if (name && content) {
        result[name] = content;
      }
    });

    return result;
  }

  function extractArticle() {
    try {
      const article = new Readability(document.cloneNode(true)).parse();

      return article?.textContent || "";
    } catch (e) {
      console.log("Readability failed:", e);
      return "";
    }
  }

  const data = {
    url: location.href,
    title: document.title,
    meta: getMeta(),
    text: extractArticle(),
    ts: Date.now(),
  };

  console.log("Collected data:", data);
})();

'use strict';
const $ = id => document.getElementById(id);
const reasons = {
  invalid_token: '密钥格式不正确，请复制完整的采集密钥', not_paired: '请先保存本机采集密钥，现有队列仍保留',
  http_401: '采集密钥无效，请重新配置；记录仍保留', http_403: '权限或来源被拒绝，请检查采集密钥和本地服务配置',
  disabled: '采集与同步已暂停', filtered: '页面被排除', queue_full: '队列已满，新增记录被拒收',
  invalid_record: '记录格式不合格', invalid_url: '页面地址不合格', invalid_ack: '服务响应不符合入库确认契约，记录仍保留',
  invalid_settings: '配置不合格，请检查数值和域名', invalid_endpoint: '仅允许本机 HTTP/HTTPS 地址，路径须为 /collect',
  invalid_storage: '本地队列格式异常，未覆盖已有数据', storage_error: '本地存储或扩展操作失败，请重试',
  unauthorized: '此页面无权执行该操作', network_error: '连接失败或超时，记录保留并等待重试',
  migrated_settings_reset: '旧配置部分无效，已暂停并恢复有效默认值；请检查配置再开启',
};
function explain(reason) { return reasons[reason] || (/^http_\d+$/.test(reason || '') ? '服务返回 HTTP ' + reason.slice(5) : reason || '未知错误'); }
function request(type, payload) {
  return new Promise((resolve, reject) => {
    chrome.runtime.sendMessage({ type, payload }, response => {
      if (chrome.runtime.lastError) reject(new Error('无法连接扩展后台，请重新打开弹窗'));
      else if (!response) reject(new Error('扩展没有返回结果'));
      else resolve(response);
    });
  });
}
let enabled = false;
const operations = new Set();
function updateButtons() {
  for (const id of ['toggleBtn', 'saveBtn']) $(id).disabled = operations.has('settings');
  for (const id of ['flushBtn', 'retryBtn']) $(id).disabled = operations.has('sync');
  $('clearBtn').disabled = operations.has('clear');
}
const formatTime = value => Number.isFinite(value) && value > 0 ? new Date(value).toLocaleString() : '尚无确认';
async function refresh() {
  const status = await request('GET_STATUS');
  if (!status.ok) throw new Error(explain(status.reason));
  enabled = status.enabled;
  $('toggleBtn').textContent = enabled ? '暂停' : '开启';
  updateButtons();
  $('queueSize').textContent = status.queueSize;
  $('blockedCount').textContent = status.blockedCount;
  $('rejectedFull').textContent = status.stats.rejectedFull || 0;
  $('lastFlushAt').textContent = formatTime(status.stats.lastFlushAt);
  $('lastSuccessAt').textContent = formatTime(status.stats.lastSuccessAt);
  $('lastError').textContent = status.stats.lastError ? '最近问题：' + explain(status.stats.lastError) : '';
  $('queueState').textContent = !enabled ? '已暂停；现有队列保留。'
    : !status.paired ? '尚未填写采集密钥，现有队列仅保存在扩展中。'
    : status.syncing ? '正在等待服务确认…'
    : status.queueSize === 0 ? '没有待确认记录。'
    : status.blockedCount === status.queueSize ? '记录需要检查服务或配置后手动重试。'
    : status.nextRetryAt > Date.now() ? '最早自动重试：' + formatTime(status.nextRetryAt) + '（由同步定时器唤醒）'
    : '等待下次同步，可点击立即同步。';
}
function populate(settings) {
  for (const key of ['endpoint', 'apiToken', 'flushIntervalMinutes', 'minTextLength']) $(key).value = settings[key];
  $('blockedDomains').value = settings.blockedDomains.join('\n');
}
async function action(task, group = 'settings') {
  if (operations.has(group)) return;
  operations.add(group); updateButtons();
  try { await task(); } catch (error) { $('result').textContent = error.message; }
  finally {
    operations.delete(group); updateButtons();
    await refresh().catch(error => { $('result').textContent = error.message; });
  }
}
async function update(patch) {
  const response = await request('UPDATE_SETTINGS', patch);
  if (!response.ok) throw new Error(explain(response.reason));
  return response.settings;
}
$('toggleBtn').addEventListener('click', () => action(async () => {
  const nextEnabled = !enabled;
  await update({ enabled: nextEnabled });
  $('result').textContent = nextEnabled ? '已开启。新网页会按排除规则采集。' : '已暂停采集与同步，队列保留。';
}));
$('saveBtn').addEventListener('click', () => action(async () => {
  populate(await update({
    endpoint: $('endpoint').value.trim(),
    apiToken: $('apiToken').value.trim(),
    flushIntervalMinutes: Number($('flushIntervalMinutes').value),
    minTextLength: Number($('minTextLength').value),
    blockedDomains: $('blockedDomains').value.split(/\r?\n/).map(value => value.trim()).filter(Boolean),
  }));
  $('result').textContent = '配置已保存。';
}));
for (const [id, type] of [['flushBtn', 'FLUSH_NOW'], ['retryBtn', 'RETRY_BLOCKED']]) {
  $(id).addEventListener('click', () => action(async () => {
    $('result').textContent = '正在等待服务确认…';
    const response = await request(type);
    if (response.reason) { $('result').textContent = explain(response.reason); return; }
    if (!Number.isInteger(response.attempted)) throw new Error(explain(response.reason));
    $('result').textContent = response.cancelled ? '同步已中断，请查看当前待确认队列。'
      : '本次确认新增 ' + response.inserted + '，已存在 ' + response.duplicates + '，失败 ' + response.failed + '；仍待确认 ' + response.queueSize + ' 条。';
  }, 'sync'));
}
$('clearBtn').addEventListener('click', () => action(async () => {
  if (!window.confirm('永久删除待确认队列？这不会删除服务端已经保存的记录。')) return;
  const response = await request('CLEAR_QUEUE');
  if (!response.ok) throw new Error(explain(response.reason));
  $('result').textContent = '已清空 ' + response.removed + ' 条待确认记录。';
}, 'clear'));
(async () => {
  try {
    const response = await request('GET_SETTINGS');
    if (!response.ok) throw new Error(explain(response.reason));
    populate(response.settings); await refresh();
  } catch (error) { $('result').textContent = error.message; }
})();
setInterval(() => { refresh().catch(error => { $('result').textContent = error.message; }); }, 2000);

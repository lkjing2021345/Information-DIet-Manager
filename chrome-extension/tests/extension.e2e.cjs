/* Real unpacked extension + real FastAPI, synthetic pages and a temporary profile/DB only.
 * Run: npm ci; npx playwright install chromium; set IDM_TEST_PYTHON; npm run test:e2e
 */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const http = require('node:http');
const { spawn } = require('node:child_process');
const { randomBytes } = require('node:crypto');
const { chromium } = require('playwright');

const extension = path.resolve(__dirname, '..');
const root = path.resolve(extension, '..');
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'idm-extension-e2e-'));
const profile = path.join(temp, 'profile');
const artifacts = path.join(root, 'output', 'playwright', 'round2');
fs.mkdirSync(artifacts, { recursive: true });
const checks = [];
const adminKey = randomBytes(32).toString('base64url'), collectorKey = randomBytes(32).toString('base64url');
const authHeaders = { Authorization: 'Bearer ' + adminKey };
const pass = name => { checks.push(name); console.log('PASS ' + name); };
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(check, label, timeout = 15000) {
  const start = Date.now();
  while (Date.now() - start < timeout) { if (await check()) return; await delay(100); }
  throw new Error('Timed out: ' + label);
}
async function listen(server) { await new Promise(resolve => server.listen(0, '127.0.0.1', resolve)); return server.address().port; }
async function close(server) { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
let context, backend, backendLog = '', stubMode = 'invalid', stubRequests = 0;
const pageServer = http.createServer((_request, response) => {
  response.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
  response.end(`<!doctype html><html lang="en"><head><title>Synthetic reading fixture</title></head><body>
    <main><h1>Synthetic reading fixture</h1><form><textarea>PRIVATE_FORM_SENTINEL</textarea></form>
    <div contenteditable="true">PRIVATE_EDITOR_SENTINEL</div>
    <p>${'This synthetic article tests reliable local collection and contains no personal data. '.repeat(30)}</p></main></body></html>`);
});
const stub = http.createServer((request, response) => {
  request.resume(); stubRequests++;
  if (stubMode === 'hold') return; // Deliberately unresolved: pause must abort the request.
  response.writeHead(stubMode === 'reject' ? 422 : 200, { 'Content-Type': 'application/json' });
  response.end(JSON.stringify(stubMode === 'ok' ? { inserted: 1, duplicates: 0, failed: 0 } : {}));
});
async function run() {
  const pagePort = await listen(pageServer), stubPort = await listen(stub);
  const portProbe = http.createServer(); const apiPort = await listen(portProbe); await close(portProbe);
  const api = `http://127.0.0.1:${apiPort}`;
  backend = spawn(process.env.IDM_TEST_PYTHON || 'python', ['-m', 'uvicorn', 'src.backend_api.app:app', '--host', '127.0.0.1', '--port', String(apiPort)], {
    cwd: root, windowsHide: true, env: { ...process.env, IDM_DB_PATH: path.join(temp, 'synthetic.sqlite3'), IDM_ADMIN_TOKEN: adminKey, IDM_COLLECTOR_TOKEN: collectorKey, HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1', PYTHONUTF8: '1' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  backend.stdout.on('data', data => { backendLog += data; }); backend.stderr.on('data', data => { backendLog += data; });
  backend.on('error', error => { backendLog += error.message; });
  await until(async () => { try { return (await fetch(api + '/health')).ok; } catch { return false; } }, 'API ready');
  assert.equal((await fetch(api + '/items')).status, 401);
  const options = { channel: 'chromium', headless: true, args: [
    `--disable-extensions-except=${extension}`, `--load-extension=${extension}`,
    '--host-resolver-rules=MAP fixture.test 127.0.0.1', '--no-proxy-server',
  ] };
  context = await chromium.launchPersistentContext(profile, options);
  const worker = context.serviceWorkers()[0] || await context.waitForEvent('serviceworker');
  const extensionId = new URL(worker.url()).host;
  const popupUrl = `chrome-extension://${extensionId}/popup.html`;
  let popup = await context.newPage(); await popup.goto(popupUrl);
  const errors = [];
  popup.on('pageerror', error => errors.push(error.message));
  const message = (type, payload) => popup.evaluate(({ type, payload }) => chrome.runtime.sendMessage({ type, payload }), { type, payload });
  const status = () => message('GET_STATUS');
  async function settings(endpoint, blocked = '') {
    await popup.bringToFront(); await popup.locator('#endpoint').fill(endpoint);
    await popup.locator('#apiToken').fill(collectorKey);
    await popup.locator('#blockedDomains').fill(blocked); await popup.locator('#saveBtn').click();
    await popup.waitForFunction(() => document.querySelector('#result').textContent === '配置已保存。' && !document.querySelector('#saveBtn').disabled);
  }
  async function flush() {
    await popup.bringToFront(); await popup.locator('#flushBtn').click();
    await popup.waitForFunction(() => !document.querySelector('#flushBtn').disabled && !document.querySelector('#result').textContent.includes('正在等待'));
  }
  assert.equal((await status()).enabled, false);
  let page = await context.newPage(); await page.goto(`http://fixture.test:${pagePort}/first?token=SECRET_QUERY&topic=tests`);
  await delay(1800); assert.equal((await status()).queueSize, 0);
  pass('fresh install stays disabled on a static page');
  await settings(api + '/collect'); await popup.locator('#toggleBtn').click();
  await until(async () => (await status()).enabled, 'enabled'); await page.bringToFront();
  await until(async () => (await status()).queueSize === 1, 'initial page collection');
  pass('explicit enable collects an already loaded static page');
  await flush(); assert.equal((await status()).queueSize, 0);
  let rows = await (await fetch(api + '/items', { headers: authHeaders })).json(); assert.equal(rows.total, 1);
  const saved = rows.items[0];
  assert.equal(saved.url.includes('SECRET_QUERY'), false); assert.equal(saved.url.includes('topic=tests'), true);
  assert.equal(saved.text.includes('PRIVATE_'), false); assert.deepEqual(Object.keys(saved.meta), ['pluginVersion']);
  pass('real API confirms saved record; form/editor text, token and referrer are absent');
  await page.bringToFront();
  await page.evaluate(() => { history.pushState({}, '', '/spa'); document.title = 'Synthetic SPA route'; });
  await until(async () => (await status()).queueSize === 1, 'SPA route collection');
  await flush(); rows = await (await fetch(api + '/items', { headers: authHeaders })).json(); assert.equal(rows.total, 2);
  pass('page-world SPA navigation is collected without reloading');
  await settings(api + '/collect', 'fixture.test'); await page.bringToFront(); await page.goto(`http://fixture.test:${pagePort}/blocked`);
  await delay(1800); assert.equal((await status()).queueSize, 0);
  pass('domain exclusion blocks fresh page collection');
  await settings(api + '/collect'); await page.bringToFront(); await page.goto(`http://fixture.test:${pagePort}/password`);
  await page.evaluate(() => { const input = document.createElement('input'); input.type = 'password'; document.body.appendChild(input); });
  await delay(1800); assert.equal((await status()).queueSize, 0);
  pass('password page is skipped');

  // Offline endpoint uses an ephemeral port whose probe has just been closed.
  const offlineProbe = http.createServer(); const offlinePort = await listen(offlineProbe); await close(offlineProbe);
  await settings(`http://127.0.0.1:${offlinePort}/collect`);
  await page.bringToFront(); await page.goto(`http://fixture.test:${pagePort}/offline`);
  await until(async () => (await status()).queueSize === 1, 'offline queued'); await flush();
  assert.equal((await status()).queueSize, 1); assert.equal((await status()).stats.lastError, 'network_error');
  await context.close(); context = await chromium.launchPersistentContext(profile, options);
  popup = await context.newPage(); await popup.goto(popupUrl);
  assert.equal((await status()).queueSize, 1); assert.equal((await status()).enabled, true);
  await settings(api + '/collect'); await flush();
  assert.equal((await status()).queueSize, 0); rows = await (await fetch(api + '/items', { headers: authHeaders })).json(); assert.equal(rows.total, 3);
  pass('offline data and opt-in survive browser/worker restart, then reach real API');

  await settings(`http://127.0.0.1:${stubPort}/collect`);
  page = await context.newPage(); await page.goto(`http://fixture.test:${pagePort}/bad-ack`);
  await until(async () => (await status()).queueSize === 1, 'bad ack page queued'); await flush();
  assert.equal((await status()).queueSize, 1); assert.equal((await status()).stats.lastError, 'invalid_ack');
  pass('HTTP 200 with invalid acknowledgement keeps the record');
  stubMode = 'reject'; await flush(); assert.equal((await status()).blockedCount, 1);
  stubMode = 'ok'; await popup.locator('#retryBtn').click();
  await until(async () => (await status()).queueSize === 0, 'manual retry recovers');
  pass('422 quarantines a record; explicit retry can recover');
  stubMode = 'hold';
  await page.bringToFront(); await page.goto(`http://fixture.test:${pagePort}/pause-in-flight`);
  await until(async () => (await status()).queueSize === 1, 'in-flight page queued');
  const before = stubRequests; await popup.bringToFront(); await popup.locator('#flushBtn').click();
  await until(() => stubRequests > before, 'request reached holding server');
  assert.equal(await popup.locator('#toggleBtn').isEnabled(), true);
  await popup.locator('#toggleBtn').click(); await until(async () => !(await status()).enabled, 'pause during sync');
  await until(async () => !(await status()).syncing, 'in-flight sync cancelled');
  assert.equal((await status()).queueSize, 1);
  popup.once('dialog', dialog => dialog.accept()); await popup.locator('#clearBtn').click();
  await until(async () => (await status()).queueSize === 0, 'clear queue');
  pass('popup pause stays usable during request and clear removes only pending queue');
  await popup.screenshot({ path: path.join(artifacts, 'popup.png'), fullPage: true });
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(artifacts, 'verification.json'), JSON.stringify({ browser: context.browser()?.version(), checks, temporaryData: true }, null, 2));
  console.log(JSON.stringify({ passed: checks.length, artifacts, temp }, null, 2));
}
run().catch(error => { console.error(error); console.error(backendLog.slice(-3000)); process.exitCode = 1; }).finally(async () => {
  if (context) await context.close();
  if (backend && backend.exitCode === null) backend.kill();
  await close(pageServer); await close(stub);
  // Keep the isolated synthetic fixture on disk for failure diagnosis; never delete a user profile.
});

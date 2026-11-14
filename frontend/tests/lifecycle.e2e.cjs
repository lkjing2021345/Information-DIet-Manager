/* Production UI + real protected API, temporary data and synthetic capability keys. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const http = require('node:http');
const { spawn } = require('node:child_process');
const { randomBytes } = require('node:crypto');
const { chromium } = require('../../chrome-extension/node_modules/playwright');

const root = path.resolve(__dirname, '../..');
const temp = fs.mkdtempSync(path.join(os.tmpdir(), 'idm-lifecycle-e2e-'));
const database = path.join(temp, 'synthetic.sqlite3');
const legacyText = '历史'.repeat(750);
const dist = path.join(temp, 'dist');
const artifacts = path.join(root, 'output/playwright/display-correctness');
fs.mkdirSync(artifacts, { recursive: true });
const admin = randomBytes(32).toString('base64url'), collector = randomBytes(32).toString('base64url');
const headers = { Authorization: 'Bearer ' + admin };
let browser, backend, backendLog = '';
const checks = [], errors = [];
const pass = label => { checks.push(label); console.log('PASS ' + label); };
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(check, name, timeout = 15000) {
  const started = Date.now();
  while (Date.now() - started < timeout) { if (await check()) return; await delay(100); }
  throw new Error('Timed out: ' + name);
}
async function listen(server) { await new Promise(resolve => server.listen(0, '127.0.0.1', resolve)); return server.address().port; }
async function close(server) { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
const ui = http.createServer((request, response) => {
  const pathname = new URL(request.url, 'http://localhost').pathname;
  const filename = path.resolve(dist, '.' + decodeURIComponent(pathname === '/' ? '/index.html' : pathname));
  if (!filename.startsWith(dist + path.sep) || !fs.existsSync(filename) || !fs.statSync(filename).isFile()) { response.writeHead(404); response.end(); return; }
  response.setHeader('Content-Type', filename.endsWith('.js') ? 'text/javascript' : filename.endsWith('.css') ? 'text/css' : filename.endsWith('.svg') ? 'image/svg+xml' : 'text/html');
  response.end(fs.readFileSync(filename));
});
function command(args, options, executable = process.execPath) {
  return new Promise((resolve, reject) => {
    const child = spawn(executable, args, { windowsHide: true, ...options });
    let output = '';
    child.stdout.on('data', chunk => { output += chunk; }); child.stderr.on('data', chunk => { output += chunk; });
    child.on('error', reject); child.on('exit', code => code === 0 ? resolve() : reject(new Error(output)));
  });
}
async function run() {
  const uiPort = await listen(ui), uiOrigin = `http://127.0.0.1:${uiPort}`;
  const probe = http.createServer(); const apiPort = await listen(probe); await close(probe);
  const api = `http://127.0.0.1:${apiPort}`;
  backend = spawn(process.env.IDM_TEST_PYTHON || 'python', ['-m', 'uvicorn', 'src.backend_api.app:app', '--host', '127.0.0.1', '--port', String(apiPort)], {
    cwd: root, windowsHide: true, env: { ...process.env, IDM_DB_PATH: database, IDM_ADMIN_TOKEN: admin,
      IDM_COLLECTOR_TOKEN: collector, IDM_FRONTEND_ORIGINS: uiOrigin, HF_HUB_OFFLINE: '1', TRANSFORMERS_OFFLINE: '1', PYTHONUTF8: '1' },
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  backend.stdout.on('data', data => { backendLog += data; }); backend.stderr.on('data', data => { backendLog += data; });
  backend.on('error', error => { backendLog += error.message; });
  await until(async () => { try { return (await fetch(api + '/health')).ok; } catch { return false; } }, 'backend');
  for (let i = 0; i < 2; i++) assert.equal((await fetch(api + '/collect', { method: 'POST', headers: { Authorization: 'Bearer ' + collector, 'Content-Type': 'application/json' },
    body: JSON.stringify({ url: 'https://example.invalid/' + i, title: 'Synthetic page ' + i, text: 'Synthetic content', tags: ['W'.repeat(40)], ts: Date.now(), source: 'plugin' }) })).status, 200);
  // Seed legal historical values directly in this disposable database: normal
  // collection truncates text, so it cannot create the long-text restore case.
  await command(['-c', [
    'import sqlite3, sys',
    'from src.hyh.utils import normalize_text, sha256_hex',
    'with sqlite3.connect(sys.argv[1]) as conn:',
    '    for index, text in enumerate((sys.argv[2], None)):',
    '        title = "Synthetic page " + str(index)',
    '        result = conn.execute("UPDATE items SET text = ?, content_hash = ? WHERE url = ?", (text, sha256_hex(normalize_text(title, text)), "https://example.invalid/" + str(index)))',
    '        assert result.rowcount == 1',
  ].join('\n'), database, legacyText], { cwd: root, env: { ...process.env, PYTHONUTF8: '1' } }, process.env.IDM_TEST_PYTHON || 'python');
  await command([path.join(root, 'frontend/node_modules/vite/bin/vite.js'), 'build', '--outDir', dist], {
    cwd: path.join(root, 'frontend'), env: { ...process.env, VITE_API_BASE_URL: api },
  });
  browser = await chromium.launch({ channel: 'chromium', headless: true });
  const context = await browser.newContext({ acceptDownloads: true, viewport: { width: 1280, height: 1000 } });
  const page = await context.newPage(); page.on('pageerror', error => errors.push(error.message));
  await page.goto(uiOrigin); await page.locator('.tech-header h1').waitFor();
  await page.evaluate(() => document.fonts.ready);
  const settingsPanel = page.locator('.settings-panel');
  const savedTotal = page.getByTestId('saved-total');
  const showSettings = async () => {
    if (!await settingsPanel.isVisible()) await page.getByTestId('settings-toggle').click();
    await settingsPanel.waitFor({ state: 'visible' });
  };
  const hideSettings = async () => {
    if (await settingsPanel.isVisible()) await page.getByTestId('settings-toggle').click();
    await settingsPanel.waitFor({ state: 'hidden' });
  };
  assert.equal(await page.getByTestId('records-toggle').isDisabled(), true);
  assert.equal(await savedTotal.textContent(), '—');
  assert.equal(await settingsPanel.isVisible(), false);
  assert.equal(await page.locator('.dashboard-grid > .tech-card').count(), 4);
  assert.equal(await page.locator('.dashboard-grid .chart-container').count(), 4);
  const chartBounds = await page.locator('.dashboard-grid > .tech-card').evaluateAll(cards => cards.map(card => {
    const rect = card.getBoundingClientRect(); return { x: rect.x, y: rect.y };
  }));
  assert.ok(chartBounds[0].x < chartBounds[1].x && Math.abs(chartBounds[0].y - chartBounds[1].y) < 2);
  assert.ok(chartBounds[2].y > chartBounds[0].y && Math.abs(chartBounds[2].x - chartBounds[0].x) < 2);
  await page.screenshot({ path: path.join(artifacts, 'disconnected.png'), fullPage: true });
  pass('production page renders disconnected; no private records are loaded');

  // Focus/hover auto-scrolling can change viewport coordinates without a
  // layout change (including the dashboard's overflow container on Windows).
  // Compare all three siblings in their shared layout coordinate system;
  // actual repositioning or resizing still fails these exact assertions.
  const homeLayout = () => page.locator('.tech-header, .summary-card, .dashboard-grid').evaluateAll(elements => elements.map(element => ({
    x: element.offsetLeft, y: element.offsetTop, width: element.offsetWidth, height: element.offsetHeight,
  })));
  const gridBeforeSettings = await homeLayout();
  await showSettings();
  assert.equal(await settingsPanel.getByTestId('admin-key').count(), 1);
  assert.deepEqual(await homeLayout(), gridBeforeSettings, 'Opening settings must not reposition or resize the home layout');
  await hideSettings();
  assert.equal(await page.getByTestId('admin-key').isVisible(), false);
  const gridAfterSettings = await homeLayout();
  assert.deepEqual(gridAfterSettings, gridBeforeSettings);
  pass('original four-chart cyber dashboard remains in place; settings toggle does not relayout the home page');

  await showSettings();
  const language = settingsPanel.locator('#ui-language');
  const font = settingsPanel.locator('#ui-font');
  const color = settingsPanel.locator('#ui-color');
  await color.evaluate(input => { input.value = '#19d8ff'; input.dispatchEvent(new Event('input', { bubbles: true })); });
  await until(async () => await page.locator('.dashboard-container').evaluate(element => getComputedStyle(element).getPropertyValue('--primary-color').trim()) === '#19d8ff', 'custom theme color');
  await font.selectOption("'Courier New', Courier, monospace");
  await until(async () => (await page.locator('.dashboard-container').evaluate(element => getComputedStyle(element).fontFamily)).includes('Courier'), 'custom font');
  await language.selectOption('en');
  await settingsPanel.getByRole('heading', { name: 'System UI Engine', exact: true }).waitFor();
  await language.selectOption('zh');
  await settingsPanel.getByTestId('theme-toggle').click();
  await until(async () => await page.locator('.dashboard-container.light-mode').count() === 1, 'light mode');
  await hideSettings();
  await page.screenshot({ path: path.join(artifacts, 'light-home.png'), fullPage: true });
  await showSettings();
  await settingsPanel.getByTestId('theme-toggle').click();
  await until(async () => await page.locator('.dashboard-container.light-mode').count() === 0, 'dark mode');
  await color.evaluate(input => { input.value = '#00f3ff'; input.dispatchEvent(new Event('input', { bubbles: true })); });
  await font.selectOption("'Segoe UI', 'Microsoft YaHei', sans-serif");
  await settingsPanel.evaluate(element => { element.scrollTop = 0; });
  await delay(600); // Let the original theme transition finish before the visual review screenshot.
  await page.screenshot({ path: path.join(artifacts, 'original-settings.png'), fullPage: true });
  pass('original theme color, font, language and light/dark controls still update the dashboard');

  await page.getByTestId('admin-key').fill(collector); await page.getByTestId('connect').click();
  await page.getByText('连接失败：', { exact: false }).waitFor();
  assert.equal(await page.getByTestId('disconnect').count(), 0);
  pass('collector key cannot unlock the management UI');
  await page.getByTestId('admin-key').fill(admin); await page.getByTestId('connect').click();
  await until(async () => await savedTotal.textContent() === '2', 'paired records');
  assert.equal(await page.getByTestId('admin-key').count(), 0);
  assert.equal(await page.evaluate(key => JSON.stringify({ local: { ...localStorage }, session: { ...sessionStorage }, url: location.href }).includes(key), admin), false);
  pass('admin key pairs, loads real records and is absent from browser storage and URL');
  const managementControls = ['disconnect', 'backup', 'restore-file', 'restore', 'delete-all', 'refresh-records', 'delete-record-id', 'delete-record'];
  for (const control of managementControls) {
    assert.equal(await settingsPanel.getByTestId(control).count(), 1);
  }
  await hideSettings();
  for (const control of managementControls) {
    assert.equal(await page.getByTestId(control).isVisible(), false);
  }
  await page.screenshot({ path: path.join(artifacts, 'connected-home.png'), fullPage: true });
  await showSettings();
  pass('connection and maintenance controls are contained in the original settings panel and hidden on the home page');

  // A real SQLite publication fault, with fewer than five records so optional
  // inference is never loaded. The API response itself is not intercepted.
  const publicationState = path.join(temp, 'publication-state.json');
  const publicationFault = action => command(['-c', [
    'import json, pathlib, sqlite3, sys',
    'state_file = pathlib.Path(sys.argv[3])',
    'with sqlite3.connect(sys.argv[1]) as conn:',
    '    if sys.argv[2] == "install":',
    '        state_file.write_text(json.dumps([row[0] for row in conn.execute("SELECT id FROM analysis_jobs")]), encoding="utf-8")',
    '        conn.execute("CREATE TRIGGER synthetic_ui_publication_failure BEFORE UPDATE ON analysis_jobs WHEN NEW.status = \'completed\' BEGIN SELECT RAISE(FAIL, \'SYNTHETIC_PRIVATE_PUBLICATION_ERROR\'); END")',
    '    elif sys.argv[2] == "inspect":',
    '        before = set(json.loads(state_file.read_text(encoding="utf-8")))',
    '        created = [row[0] for row in conn.execute("SELECT id FROM analysis_jobs") if row[0] not in before]',
    '        assert len(created) == 1, "The browser must create exactly one analysis job"',
    '        state_file.write_text(json.dumps({"job_id": created[0]}), encoding="utf-8")',
    '    else:',
    '        conn.execute("DROP TRIGGER synthetic_ui_publication_failure")',
  ].join('\n'), database, action, publicationState], { cwd: root, env: { ...process.env, PYTHONUTF8: '1' } }, process.env.IDM_TEST_PYTHON || 'python');
  const historyBeforeFailure = await (await fetch(api + '/analyze/history', { headers })).json();
  await publicationFault('install');
  try {
    await hideSettings();
    const failedResponse = page.waitForResponse(response => response.request().method() === 'GET'
      && response.url().startsWith(api + '/dashboard/visualization?'));
    await page.getByTestId('run-analysis').click();
    const response = await failedResponse;
    assert.equal(response.status(), 500);
    assert.equal((await response.allHeaders())['access-control-allow-origin'], uiOrigin);
    await until(async () => (await page.getByTestId('analysis-status').textContent()).includes('分析请求失败')
      && !(await page.getByTestId('run-analysis').isDisabled()), 'real publication failure rendered');
    // The client intentionally ignores 500 bodies; Chromium may discard that
    // unread body. Identify this request's job from the disposable DB, then
    // inspect its public API representation instead of depending on DevTools.
    await publicationFault('inspect');
    const { job_id: failedJobId } = JSON.parse(fs.readFileSync(publicationState, 'utf8'));
    const failedJobResponse = await fetch(api + '/analyze/jobs/' + failedJobId, { headers });
    assert.equal(failedJobResponse.status, 200);
    const failedJob = await failedJobResponse.json();
    assert.equal(failedJob.status, 'failed');
    assert.equal(failedJob.result_payload, null);
    assert.equal(failedJob.error, 'Visualization analysis failed; no valid result was produced.');
    assert.equal(JSON.stringify(failedJob).includes('SYNTHETIC_PRIVATE_PUBLICATION_ERROR'), false);
    assert.equal((await page.locator('body').textContent()).includes('SYNTHETIC_PRIVATE_PUBLICATION_ERROR'), false);
    assert.deepEqual(await (await fetch(api + '/analyze/history', { headers })).json(), historyBeforeFailure);
    assert.equal(await savedTotal.textContent(), '2');
  } finally {
    await publicationFault('remove');
  }
  const recoveredResponse = page.waitForResponse(response => response.request().method() === 'GET'
    && response.url().startsWith(api + '/dashboard/visualization?'));
  await page.getByTestId('run-analysis').click();
  assert.equal((await recoveredResponse).status(), 200);
  await until(async () => (await page.getByTestId('analysis-status').textContent()).includes('样本不足'), 'publication retry recovers without inventing analysis');
  await showSettings();
  pass('real analysis publication failure shows no result, persists a failed job without private errors, and permits an honest retry');

  // Synthetic analysis response checks rendering only. No model inference is
  // exercised by this fixture; lifecycle requests below still use the real API.
  const fixtureCategories = ['entertainment', 'learning', 'news', 'social', 'shopping', 'tools', 'other'];
  const fixtureRows = count => [
    { date: '2026-09-22', count, repeat_ratio: 0, positive_ratio: 0.5, neutral_ratio: 0.5, negative_ratio: 0 },
    { date: '2026-09-24', count, repeat_ratio: 0.25, positive_ratio: 0, neutral_ratio: 0.75, negative_ratio: 0.25 },
  ];
  const visualizationFixture = {
    analysis_status: 'ready', minimum_records: 5, generated_at: Date.UTC(2026, 8, 24, 23, 59),
    window: { from_ts: Date.UTC(2026, 8, 22), to_ts: Date.UTC(2026, 8, 24, 23, 59), input_count: 56, available_count: 56, processed_count: 56, truncated: false },
    coverage: { record_count: 56, timestamp_count: 56, category_count: 56, comparison_count: 48, sentiment_count: 52 },
    category_counts: Object.fromEntries(fixtureCategories.map(key => [key, 8])),
    global: { time_series: fixtureRows(28) },
    categories: Object.fromEntries(fixtureCategories.map(key => [key, { time_series: fixtureRows(4) }])),
  };
  const canvasImages = () => page.locator('.chart-container').evaluateAll(elements => elements.map(element => element.querySelector('canvas')?.toDataURL() || null));
  const emptyCanvases = await canvasImages();
  await page.route(api + '/dashboard/visualization?**', route => route.fulfill({ json: visualizationFixture }));
  await hideSettings(); await page.getByTestId('run-analysis').click();
  await until(async () => (await page.getByTestId('analysis-status').textContent()).includes('实验统计已生成'), 'fixture analysis rendering');
  const coverageText = await page.getByTestId('analysis-status').textContent();
  assert.ok(coverageText.includes('分类有效 56/56 条') && coverageText.includes('情感有效 52/56 条')
    && coverageText.includes('日期有效 56/56 条') && coverageText.includes('相邻比较有效 48/55 对')
    && coverageText.includes('后一条记录的 UTC 日期和分类归属'));
  await until(async () => (await canvasImages()).every((image, index) => image && image !== emptyCanvases[index]), 'all four charts paint the supplied fixture');
  assert.equal(await page.locator('.chart-container canvas').count(), 4);
  assert.equal(await page.locator('.dashboard-grid h2').count(), 4);
  await delay(1000);
  await page.screenshot({ path: path.join(artifacts, 'fixture-ready-charts.png'), fullPage: true });
  await page.unroute(api + '/dashboard/visualization?**');
  await showSettings();
  pass('synthetic analysis fixture renders all four charts and independent metric coverage with the UTC later-record comparison definition; this does not verify model inference');

  const downloadPromise = page.waitForEvent('download'); await page.getByTestId('backup').click();
  const download = await downloadPromise, backupPath = path.join(temp, 'backup.json'); await download.saveAs(backupPath);
  const backup = JSON.parse(fs.readFileSync(backupPath, 'utf8')); assert.equal(backup.items.length, 2);
  assert.equal(backup.items.find(item => item.url === 'https://example.invalid/0').text, legacyText);
  assert.equal(backup.items.find(item => item.url === 'https://example.invalid/1').text, null);
  assert.equal(fs.readFileSync(backupPath, 'utf8').includes(admin), false);
  await until(async () => !(await page.getByTestId('backup').isDisabled()), 'backup complete');
  await page.screenshot({ path: path.join(artifacts, 'connected.png'), fullPage: true });
  pass('authenticated backup downloads two page records without capability keys');

  // Change only a disposable fixture's URL/hash; compare every table before cleanup.
  const conflictState = path.join(temp, 'backup-conflict-state.json');
  const conflictFixture = [
    'import hashlib, json, sqlite3, sys',
    'from pathlib import Path',
    'database, state_path, action = sys.argv[1:]',
    'state_file = Path(state_path)',
    'def snapshot(conn):',
    '    return hashlib.sha256("\\n".join(conn.iterdump()).encode("utf-8")).hexdigest()',
    'with sqlite3.connect(database) as conn:',
    '    if action == "seed":',
    '        row = conn.execute("SELECT id, url, url_hash FROM items WHERE url = ?", ("https://example.invalid/1",)).fetchone()',
    '        assert row is not None',
    '        revision = conn.execute("SELECT revision FROM items_revision WHERE singleton = 1").fetchone()[0]',
    '        state = {"row": row, "revision": revision, "before": snapshot(conn)}',
    '        conn.execute("UPDATE items SET url = ?, url_hash = NULL WHERE id = ?", ("https://EXAMPLE.invalid:443/0#backup-409-fixture", row[0]))',
    '        conn.commit()',
    '        state["conflict"] = snapshot(conn)',
    '        state_file.write_text(json.dumps(state), encoding="utf-8")',
    '    else:',
    '        state = json.loads(state_file.read_text(encoding="utf-8"))',
    '        if action == "verify":',
    '            assert snapshot(conn) == state["conflict"], "Rejected backup changed the database"',
    '        elif action == "cleanup":',
    '            row = state["row"]',
    '            assert conn.execute("UPDATE items SET url = ?, url_hash = ? WHERE id = ?", (row[1], row[2], row[0])).rowcount == 1',
    '            conn.execute("UPDATE items_revision SET revision = ? WHERE singleton = 1 AND revision = ?", (state["revision"], state["revision"] + 2))',
    '            conn.commit()',
    '            assert snapshot(conn) == state["before"], "Fixture cleanup did not restore the original database"',
    '        else:',
    '            raise AssertionError("Unknown fixture action")',
  ].join('\n');
  const runConflictFixture = action => command(['-c', conflictFixture, database, conflictState, action],
    { cwd: root, env: { ...process.env, PYTHONUTF8: '1' } }, process.env.IDM_TEST_PYTHON || 'python');
  let rejectedBackupDownloads = 0;
  const onRejectedBackupDownload = () => { rejectedBackupDownloads += 1; };
  await runConflictFixture('seed');
  page.on('download', onRejectedBackupDownload);
  try {
    const rejectedResponse = page.waitForResponse(response => response.url() === api + '/data/backup'
      && response.request().method() === 'GET');
    await page.getByTestId('backup').click();
    const response = await rejectedResponse;
    assert.equal(response.status(), 409);
    assert.equal(await response.finished(), null);
    await until(async () => !(await page.getByTestId('backup').isDisabled())
      && !(await page.getByTestId('refresh-records').isDisabled()), 'conflicting backup rejected and maintenance released');
    assert.equal(await page.getByTestId('data-result').textContent(), '操作结果未确认，请检查服务、备份文件并刷新核对数据；不要连续重复提交。');
    assert.equal(rejectedBackupDownloads, 0, 'Rejected backup must not initiate a download');
    await runConflictFixture('verify');
  } finally {
    page.off('download', onRejectedBackupDownload);
    await runConflictFixture('cleanup');
  }
  await page.getByTestId('refresh-records').click();
  await until(async () => await savedTotal.textContent() === '2'
    && !(await page.getByTestId('refresh-records').isDisabled()), 'original records refreshed after fixture cleanup');
  pass('real duplicate-URL backup returns 409, shows failure without downloading, preserves every database table, and releases maintenance');
  await hideSettings();
  const originalBodyOverflow = await page.evaluate(() => document.body.style.overflow);
  await page.getByTestId('records-toggle').click();
  const drawer = page.locator('.drawer-panel');
  await drawer.waitFor({ state: 'visible' });
  await until(async () => {
    const bounds = await drawer.boundingBox();
    return bounds && bounds.x > 640 && Math.abs(bounds.x + bounds.width - 1280) < 2 && Math.abs(bounds.y) < 2;
  }, 'right-side drawer layout after slide transition');
  assert.equal(await drawer.evaluate(element => getComputedStyle(element).position), 'fixed');
  assert.equal(await drawer.locator('.cyber-tag').count(), 1);
  assert.equal(await drawer.evaluate(element => element.scrollWidth <= element.clientWidth + 1), true, 'A maximum-length unbroken tag must not overflow the drawer');
  const lockedScrollY = await page.evaluate(() => scrollY);
  await page.mouse.move(10, 500); await page.mouse.wheel(0, 500); await delay(250);
  assert.equal(await page.evaluate(() => scrollY), lockedScrollY, 'Scrolling the overlay must not move the background page');
  assert.equal(await drawer.evaluate(element => document.activeElement === element), true);
  await page.keyboard.press('Tab');
  assert.equal(await page.getByTestId('drawer-close').evaluate(element => document.activeElement === element), true);
  await page.keyboard.press('Shift+Tab');
  assert.equal(await page.getByTestId('record-row').locator('a').last().evaluate(element => document.activeElement === element), true);
  await page.keyboard.press('Tab');
  assert.equal(await page.getByTestId('drawer-close').evaluate(element => document.activeElement === element), true);
  await page.keyboard.press('Escape'); await drawer.waitFor({ state: 'hidden' });
  assert.equal(await page.getByTestId('records-toggle').evaluate(element => document.activeElement === element), true);
  assert.equal(await page.evaluate(() => document.body.style.overflow), originalBodyOverflow);
  await page.getByTestId('records-toggle').click(); await drawer.waitFor({ state: 'visible' });
  await delay(500);
  await drawer.hover({ position: { x: 30, y: 100 } });
  await delay(350);
  const hoveredBounds = await drawer.boundingBox();
  assert.ok(Math.abs(hoveredBounds.y) < 2 && Math.abs(hoveredBounds.x + hoveredBounds.width - 1280) < 2, 'Hover must not translate the fixed drawer');
  await drawer.evaluate(element => { element.scrollTop = 0; });
  await page.screenshot({ path: path.join(artifacts, 'records-drawer-top.png'), fullPage: true });
  pass('drawer traps keyboard focus, Escape restores focus and body scrolling, and long tags and overlay scrolling stay contained');
  await page.getByTestId('drawer-close').click();
  await page.waitForFunction(() => {
    const element = document.querySelector('.drawer-panel');
    return element && new DOMMatrixReadOnly(getComputedStyle(element).transform).m41 > 1;
  }, undefined, { timeout: 1000, polling: 'raf' });
  await drawer.waitFor({ state: 'hidden' });
  assert.equal(await page.getByTestId('records-toggle').evaluate(element => document.activeElement === element), true);
  pass('hover leaves the drawer fixed and does not suppress its rightward exit animation');
  await page.getByTestId('records-toggle').click();
  await drawer.waitFor({ state: 'visible' });
  await page.locator('.drawer-overlay').click({ position: { x: 10, y: 10 } });
  await drawer.waitFor({ state: 'hidden' });
  assert.equal(await page.getByTestId('records-toggle').evaluate(element => document.activeElement === element), true);
  await page.getByTestId('records-toggle').click();
  await drawer.waitFor({ state: 'visible' });
  pass('records open in the original fixed right-side drawer; close button and backdrop both dismiss it');
  assert.equal(await drawer.getByTestId('delete-record').count(), 0);
  await page.getByTestId('drawer-close').click();
  await drawer.waitFor({ state: 'hidden' });
  await showSettings();
  const recordToDelete = (await (await fetch(api + '/items', { headers })).json()).items[0];
  await page.getByTestId('delete-record-id').selectOption(String(recordToDelete.id));
  page.once('dialog', dialog => dialog.accept()); await page.getByTestId('delete-record').click();
  await until(async () => await savedTotal.textContent() === '1', 'single deletion');
  const remainingRecords = await (await fetch(api + '/items', { headers })).json();
  assert.equal(remainingRecords.total, 1);
  assert.equal(remainingRecords.items.some(item => item.id === recordToDelete.id), false);
  pass('record deletion in settings removes the chosen real ID and updates both database and visible count');

  const damagedPath = path.join(temp, 'damaged.json'); fs.writeFileSync(damagedPath, JSON.stringify({ ...backup, sha256: '0'.repeat(64) }));
  await page.getByTestId('restore-file').setInputFiles(damagedPath); page.once('dialog', dialog => dialog.accept()); await page.getByTestId('restore').click();
  await until(async () => (await page.getByTestId('data-result').textContent()).includes('操作') && !(await page.getByTestId('restore').isDisabled()), 'damaged restore rejected');
  assert.equal((await (await fetch(api + '/items', { headers })).json()).total, 1);
  pass('damaged backup is rejected through real upload; current data survives');
  await page.getByTestId('restore-file').setInputFiles(backupPath); page.once('dialog', dialog => dialog.accept()); await page.getByTestId('restore').click();
  await until(async () => await savedTotal.textContent() === '2', 'restore count');
  await until(async () => !(await page.getByTestId('backup').isDisabled()), 'restored backup available');
  const restoredDownloadPromise = page.waitForEvent('download'); await page.getByTestId('backup').click();
  const restoredDownload = await restoredDownloadPromise, restoredBackupPath = path.join(temp, 'restored-backup.json');
  await restoredDownload.saveAs(restoredBackupPath);
  const restoredBackup = JSON.parse(fs.readFileSync(restoredBackupPath, 'utf8'));
  assert.deepEqual(restoredBackup.items, backup.items);
  pass('UI backup, deletion, restore and second download preserve every page field, including 1500-character text and null');

  // Hold a real background response, then collect a page that only a later
  // foreground request can see. Keep the browser's real 30-second interval.
  const recordsFirstPage = url => url.origin === api && url.pathname === '/items' && !url.searchParams.has('cursor');
  let releaseBackground, backgroundFinished = false;
  await until(async () => !(await page.getByTestId('refresh-records').isDisabled()), 'restore refresh finished');
  await page.route(recordsFirstPage, async route => {
    const response = await route.fetch();
    assert.equal((await response.json()).total, 2);
    await new Promise(resolve => { releaseBackground = resolve; });
    await route.fulfill({ response }).catch(() => {});
    backgroundFinished = true;
  }, { times: 1 });
  await hideSettings();
  await until(() => Boolean(releaseBackground), 'real background poll in flight', 35000);
  const duringPollUrl = 'https://example.invalid/arrived-during-background-poll';
  const duringPollCollection = await fetch(api + '/collect', { method: 'POST', headers: { Authorization: 'Bearer ' + collector, 'Content-Type': 'application/json' },
    body: JSON.stringify({ url: duringPollUrl, title: 'Page collected after background snapshot', text: 'Synthetic polling race', ts: Date.now(), source: 'plugin' }) });
  assert.equal(duringPollCollection.status, 200);
  assert.equal((await duringPollCollection.json()).inserted, 1);
  try {
    await page.getByTestId('records-toggle').click();
    await until(async () => await savedTotal.textContent() === '3' && await page.getByTestId('record-row').count() === 3,
      'drawer opening supersedes pending background poll', 5000);
    assert.equal(await page.getByTestId('record-row').locator('a').first().getAttribute('href'), duringPollUrl);
  } finally { releaseBackground(); }
  await until(() => backgroundFinished, 'obsolete background response released');
  assert.equal(await savedTotal.textContent(), '3', 'Late background response cannot overwrite the fresh drawer snapshot');
  pass('opening the drawer supersedes a pending real background poll and includes pages collected after its snapshot');
  await page.getByTestId('drawer-close').click(); await drawer.waitFor({ state: 'hidden' });
  await showSettings();
  const transientRecord = (await (await fetch(api + '/items', { headers })).json()).items.find(item => item.url === duringPollUrl);
  assert.equal((await fetch(api + '/items/' + transientRecord.id, { method: 'DELETE', headers: { ...headers, 'X-IDM-Confirm': 'delete-record' } })).status, 200);
  await page.getByTestId('refresh-records').click();
  await until(async () => await savedTotal.textContent() === '2', 'polling fixture cleanup');

  // An explicit settings refresh must also supersede polling. Hold both real
  // responses to ensure the cancelled poll cannot end the new loading state.
  releaseBackground = undefined; backgroundFinished = false;
  await until(async () => !(await page.getByTestId('refresh-records').isDisabled()), 'cleanup refresh finished');
  await page.route(recordsFirstPage, async route => {
    const response = await route.fetch();
    assert.equal((await response.json()).total, 2);
    await new Promise(resolve => { releaseBackground = resolve; });
    await route.fulfill({ response }).catch(() => {});
    backgroundFinished = true;
  }, { times: 1 });
  await hideSettings();
  await until(() => Boolean(releaseBackground), 'second real background poll in flight', 35000);
  assert.equal((await fetch(api + '/collect', { method: 'POST', headers: { Authorization: 'Bearer ' + collector, 'Content-Type': 'application/json' },
    body: JSON.stringify({ url: duringPollUrl, title: 'Page collected before manual refresh', text: 'Synthetic polling race', ts: Date.now(), source: 'plugin' }) })).status, 200);
  let releaseManualRefresh;
  const holdManualRefresh = async route => {
    const response = await route.fetch();
    assert.equal((await response.json()).total, 3);
    await new Promise(resolve => { releaseManualRefresh = resolve; });
    await route.fulfill({ response });
  };
  // Keep interception registered until both held responses finish. Two
  // expiring routes let Playwright disable interception when the old handler
  // ends, which can resume the new request before its test gate is released.
  await page.route(recordsFirstPage, holdManualRefresh);
  await showSettings();
  assert.equal(await page.getByTestId('refresh-records').isDisabled(), false, 'Background polling must not disable explicit refresh');
  await page.getByTestId('refresh-records').click();
  await until(() => Boolean(releaseManualRefresh), 'manual refresh supersedes background poll');
  assert.equal(await page.getByTestId('refresh-records').isDisabled(), true);
  releaseBackground();
  await until(() => backgroundFinished, 'second obsolete background response released');
  assert.equal(await page.getByTestId('refresh-records').isDisabled(), true, 'Old request cleanup must not end the foreground loading state');
  releaseManualRefresh();
  await until(async () => await savedTotal.textContent() === '3' && !(await page.getByTestId('refresh-records').isDisabled()), 'manual refresh rendered');
  await page.unroute(recordsFirstPage, holdManualRefresh);
  pass('explicit settings refresh supersedes polling and its loading state survives late background completion');
  const manualTransientRecord = (await (await fetch(api + '/items', { headers })).json()).items.find(item => item.url === duringPollUrl);
  assert.equal((await fetch(api + '/items/' + manualTransientRecord.id, { method: 'DELETE', headers: { ...headers, 'X-IDM-Confirm': 'delete-record' } })).status, 200);
  await page.getByTestId('refresh-records').click();
  await until(async () => await savedTotal.textContent() === '2', 'manual polling fixture cleanup');

  // A second page must come from the protected API, not a local UI fixture.
  for (let i = 2; i < 53; i++) {
    const response = await fetch(api + '/collect', { method: 'POST', headers: { Authorization: 'Bearer ' + collector, 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: 'https://example.invalid/' + i, title: 'Synthetic page ' + i, text: 'Synthetic pagination content', ts: Date.now(), source: 'plugin' }) });
    assert.equal(response.status, 200); assert.equal((await response.json()).inserted, 1);
  }
  const seededPageResponse = page.waitForResponse(response => {
    const url = new URL(response.url());
    return response.request().method() === 'GET' && url.origin === api && url.pathname === '/items' && !url.searchParams.has('cursor');
  });
  await page.getByTestId('refresh-records').click();
  const seededPage = await (await seededPageResponse).json();
  await until(async () => await savedTotal.textContent() === '53', 'pagination seed count');
  // Keep settings open until openRecords closes them, so background polling
  // cannot consume the one-shot gate intended for the drawer's fresh page.
  const previousSnapshotStatus = await page.getByTestId('records-status').textContent();
  // Reopening requests a new first page while the previous 50 rows remain
  // visible. Hold that real request so stale row counts cannot satisfy the
  // readiness check; do not manufacture an API response or relax timestamps.
  let releaseDrawerRefresh;
  const firstPageUrl = url => url.origin === api && url.pathname === '/items' && !url.searchParams.has('cursor');
  await page.route(firstPageUrl, async route => {
    await new Promise(resolve => { releaseDrawerRefresh = resolve; });
    await route.fulfill({ response: await route.fetch() });
  }, { times: 1 });
  const drawerPageResponse = page.waitForResponse(response => response.request().method() === 'GET' && firstPageUrl(new URL(response.url())));
  await page.getByTestId('records-toggle').click();
  await until(() => Boolean(releaseDrawerRefresh), 'held real drawer refresh');
  assert.equal(await page.getByTestId('record-row').count(), 50, 'Old rows remain visible while the fresh first page is pending');
  assert.equal(await page.getByTestId('load-more').isDisabled(), true, 'Pagination cannot use the previous cursor while refreshing');
  assert.equal(await page.getByTestId('records-status').textContent(), previousSnapshotStatus);
  // Make the two real snapshots observably different even on a fast runner.
  await until(() => Date.now() >= (Math.floor(seededPage.snapshot_at / 1000) + 1) * 1000, 'next snapshot display second');
  releaseDrawerRefresh();
  const drawerPage = await (await drawerPageResponse).json();
  assert.ok(Math.floor(drawerPage.snapshot_at / 1000) > Math.floor(seededPage.snapshot_at / 1000));
  await until(async () => await page.getByTestId('record-row').count() === 50 &&
    !(await page.getByTestId('load-more').isDisabled()) &&
    await page.getByTestId('records-status').textContent() !== previousSnapshotStatus, 'fresh first drawer page rendered');
  pass('drawer refresh keeps previous rows visible but blocks pagination until the real new snapshot is rendered');
  const expectedSnapshotUrls = (await (await fetch(api + '/items?page_size=200', { headers })).json()).items.map(item => item.url);
  const paginationBackup = await (await fetch(api + '/data/backup', { headers })).json();
  const snapshotStatus = await page.getByTestId('records-status').textContent();
  const arrivalUrl = 'https://example.invalid/new-during-pagination';
  const arrivalResponse = await fetch(api + '/collect', { method: 'POST', headers: { Authorization: 'Bearer ' + collector, 'Content-Type': 'application/json' },
    body: JSON.stringify({ url: arrivalUrl, title: 'New arrival during pagination', text: 'Synthetic arrival', ts: Date.now(), source: 'plugin' }) });
  assert.equal(arrivalResponse.status, 200); assert.equal((await arrivalResponse.json()).inserted, 1);
  await page.getByTestId('load-more').click();
  await until(async () => await page.getByTestId('record-row').count() === 53, 'second drawer page');
  assert.equal(await savedTotal.textContent(), '53', 'New arrivals must not alter the current snapshot total');
  assert.equal(await page.getByTestId('records-status').textContent(), snapshotStatus, 'All pages retain the first snapshot timestamp');
  assert.equal(await page.getByTestId('load-more').count(), 0);
  const displayedUrls = () => page.getByTestId('record-row').locator('a').evaluateAll(links => links.map(link => link.href));
  const recordUrls = await displayedUrls();
  assert.equal(recordUrls.length, 53); assert.equal(new Set(recordUrls).size, 53);
  assert.deepEqual(recordUrls, expectedSnapshotUrls);
  assert.equal(recordUrls.includes(arrivalUrl), false);
  pass('concurrent collection leaves every original snapshot record visible exactly once with a consistent total and end-of-pages state');
  await page.getByTestId('drawer-close').click(); await drawer.waitFor({ state: 'hidden' });
  await page.getByTestId('records-toggle').click();
  await until(async () => await savedTotal.textContent() === '54' && await page.getByTestId('record-row').count() === 50, 'new arrival after reopening');
  assert.equal((await displayedUrls())[0], arrivalUrl);
  pass('reopening the drawer starts a fresh snapshot and shows arrivals excluded from the previous one');

  // Hold an analysis response while another client changes the records. The
  // snapshot recovery must cancel and invalidate this older analysis request.
  await page.getByTestId('drawer-close').click(); await drawer.waitFor({ state: 'hidden' });
  let releaseOldAnalysis;
  await page.route(api + '/dashboard/visualization?**', async route => {
    await new Promise(resolve => { releaseOldAnalysis = resolve; });
    await route.fulfill({ json: visualizationFixture }).catch(() => {});
  });
  await page.getByTestId('run-analysis').click();
  await until(() => Boolean(releaseOldAnalysis), 'held analysis request');
  // Keyboard navigation remains available during the visual loading overlay;
  // the record drawer sits above it and can still request its next page.
  await page.keyboard.press('Tab');
  assert.equal(await page.getByTestId('records-toggle').evaluate(element => document.activeElement === element), true);
  await page.keyboard.press('Enter');
  await drawer.waitFor({ state: 'visible' });
  await until(async () => await page.getByTestId('record-row').count() === 50 && !(await page.getByTestId('load-more').isDisabled()), 'drawer available during pending analysis');

  // A second management client deletes an already-loaded row. The pending
  // cursor must expire, and the browser must replace rather than append pages.
  const arrival = (await (await fetch(api + '/items?page_size=200', { headers })).json()).items.find(item => item.url === arrivalUrl);
  const externalDelete = await fetch(api + '/items/' + arrival.id, { method: 'DELETE', headers: { ...headers, 'X-IDM-Confirm': 'delete-record' } });
  assert.equal(externalDelete.status, 200);
  await page.getByTestId('load-more').click();
  await until(async () => await savedTotal.textContent() === '53' && (await page.getByTestId('records-status').textContent()).includes('记录已发生变化'), 'expired cursor automatically resets after deletion');
  assert.equal(await page.getByTestId('record-row').count(), 50);
  assert.equal((await displayedUrls()).includes(arrivalUrl), false);
  releaseOldAnalysis(); await delay(500);
  assert.ok((await page.getByTestId('analysis-status').textContent()).includes('尚未运行实验分析'));
  assert.equal(await page.locator('.global-loading').count(), 0);
  await page.unroute(api + '/dashboard/visualization?**');
  pass('snapshot expiration cancels an older delayed analysis so its result cannot repopulate cleared charts');
  await page.getByTestId('load-more').click();
  await until(async () => await page.getByTestId('record-row').count() === 53, 'complete refreshed snapshot after deletion');
  assert.deepEqual(await displayedUrls(), expectedSnapshotUrls);
  assert.equal(await page.getByTestId('load-more').count(), 0);
  pass('external deletion expires the cursor, replaces loaded rows once, and preserves every surviving record without duplicates');

  // Restore also expires cursors even when all restored URLs and totals match.
  await page.getByTestId('drawer-close').click(); await drawer.waitFor({ state: 'hidden' });
  await page.getByTestId('records-toggle').click();
  await until(async () => await page.getByTestId('record-row').count() === 50 && !(await page.getByTestId('load-more').isDisabled()), 'first page before external restore');
  const externalRestore = await fetch(api + '/data/restore', { method: 'POST', headers: { ...headers, 'Content-Type': 'application/json', 'X-IDM-Confirm': 'replace-records' }, body: JSON.stringify(paginationBackup) });
  assert.equal(externalRestore.status, 200);
  await page.getByTestId('load-more').click();
  await until(async () => (await page.getByTestId('records-status').textContent()).includes('记录已发生变化') && !(await page.getByTestId('load-more').isDisabled()), 'expired cursor automatically resets after restore');
  assert.equal(await page.getByTestId('record-row').count(), 50);
  await page.getByTestId('load-more').click();
  await until(async () => await page.getByTestId('record-row').count() === 53, 'complete restored snapshot');
  const restoredUrls = (await (await fetch(api + '/items?page_size=200', { headers })).json()).items.map(item => item.url);
  assert.deepEqual(await displayedUrls(), restoredUrls);
  assert.equal(new Set(await displayedUrls()).size, 53);
  assert.equal(await page.getByTestId('load-more').count(), 0);
  pass('external restore invalidates the old snapshot even at the same total and reloads replacement IDs without mixing records');

  // Fault injection checks the retry bound only. All insert/delete/restore
  // scenarios above exercised real backend responses and synthetic SQLite data.
  await page.getByTestId('drawer-close').click(); await drawer.waitFor({ state: 'hidden' });
  await page.getByTestId('records-toggle').click();
  await until(async () => await page.getByTestId('record-row').count() === 50 && !(await page.getByTestId('load-more').isDisabled()), 'first page before repeated conflict fixture');
  let conflictRequests = 0;
  await page.route(api + '/items?**', route => {
    conflictRequests++;
    return route.fulfill({ status: 409, json: { detail: { code: 'items_snapshot_expired' } } });
  });
  await page.getByTestId('load-more').click();
  await drawer.getByRole('alert').waitFor(); await delay(250);
  assert.equal(conflictRequests, 2, 'Only one fresh first-page request is permitted after cursor expiration');
  assert.equal(await page.getByTestId('record-row').count(), 0, 'Expired private records must clear even when recovery fails');
  assert.equal(await savedTotal.textContent(), '—');
  assert.equal(await page.getByTestId('load-more').count(), 0);
  await page.unroute(api + '/items?**');
  await page.getByTestId('drawer-close').click(); await drawer.waitFor({ state: 'hidden' });
  await page.getByTestId('records-toggle').click();
  await until(async () => await savedTotal.textContent() === '53' && !(await page.getByTestId('load-more').isDisabled()), 'manual retry after conflict fixture');
  await page.getByTestId('load-more').click();
  await until(async () => await page.getByTestId('record-row').count() === 53, 'manual retry completes snapshot');
  assert.deepEqual(await displayedUrls(), restoredUrls);
  pass('repeated conflict fixture stops after one recovery request, clears expired data and supports a later manual retry');
  await page.screenshot({ path: path.join(artifacts, 'records-drawer.png'), fullPage: true });
  await page.getByTestId('drawer-close').click(); await drawer.waitFor({ state: 'hidden' });
  await showSettings();
  pass('right-side drawer loads both real API pages, displays all 53 records once and removes the completed pagination action');

  const olderRecord = (await (await fetch(api + '/items?page=2&page_size=50', { headers })).json()).items.at(-1);
  assert.ok(olderRecord, 'The deletion target must come from the second real API page');
  await page.getByTestId('delete-record-id').selectOption(String(olderRecord.id));
  await delay(31000); // Cross a real 30-second polling interval without mocking the browser clock.
  assert.equal(await page.getByTestId('delete-record-id').inputValue(), String(olderRecord.id));
  assert.equal(await page.getByTestId('delete-record-id').locator('option').count(), 54);
  page.once('dialog', dialog => dialog.accept()); await page.getByTestId('delete-record').click();
  await until(async () => await savedTotal.textContent() === '52', 'second-page record deletion');
  const afterOlderDeletion = await (await fetch(api + '/items?page=2&page_size=50', { headers })).json();
  assert.equal(afterOlderDeletion.total, 52);
  assert.equal(afterOlderDeletion.items.some(item => item.id === olderRecord.id), false);
  pass('second-page selection survives a real background polling interval and can be deleted from settings');

  page.once('dialog', dialog => dialog.dismiss()); await page.getByTestId('delete-all').click();
  assert.equal((await (await fetch(api + '/items', { headers })).json()).total, 52);
  page.once('dialog', dialog => dialog.accept()); await page.getByTestId('delete-all').click();
  await until(async () => await savedTotal.textContent() === '0', 'clear all');
  pass('cancelled deletion preserves records; confirmed deletion empties the database');

  // Hold an earlier response across logout: it must not restore private data in the page.
  await page.route(api + '/items?**', async route => { await delay(500); await route.fulfill({ json: { page: 1, page_size: 50, total: 999, items: [] } }).catch(() => {}); });
  await page.getByTestId('refresh-records').click();
  await page.getByTestId('disconnect').click(); await delay(700);
  assert.equal(await savedTotal.textContent(), '—');
  assert.equal(await page.getByTestId('backup').count(), 0);
  await page.reload(); await showSettings(); assert.equal(await page.getByTestId('admin-key').inputValue(), '');
  pass('disconnect clears view and key; a stale response and reload cannot restore the session');
  await page.unroute(api + '/items?**');
  await page.getByTestId('admin-key').fill(admin); await page.getByTestId('connect').click();
  await until(async () => await savedTotal.textContent() === '0', 'reconnection');
  await page.route(api + '/items?**', route => route.fulfill({ status: 401, json: { detail: 'Local access key required' } }));
  await page.getByTestId('refresh-records').click();
  await page.getByTestId('admin-key').waitFor();
  assert.equal(await savedTotal.textContent(), '—');
  pass('server authentication rejection removes the local management session and visible data');
  await page.screenshot({ path: path.join(artifacts, 'final.png'), fullPage: true });

  // Exercise Vue's actual unmount hook without navigating or closing the tab,
  // which would cancel fetches independently and hide missing app cleanup.
  await page.unroute(api + '/items?**');
  await page.getByTestId('admin-key').fill(admin); await page.getByTestId('connect').click();
  await until(async () => await savedTotal.textContent() === '0' && !(await page.getByTestId('refresh-records').isDisabled()), 'reconnection before unmount');
  let releaseUnmountResponse, unmountResponseFinished = false;
  await page.route(recordsFirstPage, async route => {
    const response = await route.fetch();
    await new Promise(resolve => { releaseUnmountResponse = resolve; });
    await route.fulfill({ response }).catch(() => {});
    unmountResponseFinished = true;
  }, { times: 1 });
  await page.getByTestId('refresh-records').click();
  await until(() => Boolean(releaseUnmountResponse), 'real records response held before unmount');
  const unmountedRequest = page.waitForEvent('requestfailed', {
    predicate: request => request.method() === 'GET' && recordsFirstPage(new URL(request.url())), timeout: 5000,
  });
  await page.evaluate(() => document.querySelector('#app').__vue_app__.unmount());
  assert.match((await unmountedRequest).failure().errorText, /abort/i);
  releaseUnmountResponse();
  await until(() => unmountResponseFinished, 'late response released after unmount');
  assert.equal(await page.locator('#app').textContent(), '');
  assert.equal(await page.getByTestId('saved-total').count(), 0);
  pass('actual component unmount aborts a pending real records request and late delivery cannot restore private UI');
  assert.deepEqual(errors, []);
  fs.writeFileSync(path.join(artifacts, 'verification.json'), JSON.stringify({ browser: browser.version(), checks, temporaryData: true }, null, 2));
  console.log(JSON.stringify({ passed: checks.length, artifacts }, null, 2));
}
run().catch(error => { console.error(error); console.error(backendLog.slice(-2500)); process.exitCode = 1; }).finally(async () => {
  if (browser) await browser.close();
  if (backend && backend.exitCode === null) backend.kill();
  await close(ui);
});

import test from 'node:test'
import assert from 'node:assert/strict'
import { createLocalApi, localApiBase } from '../src/local-api.js'

const key = 'a'.repeat(43)
const ok = data => Response.json(data)

test('API addresses cannot send local keys to remote, credential-bearing or ambiguous URLs', () => {
  for (const value of ['https://remote.test', 'http://127.0.0.1@remote.test', 'http://localhost:8000/path', 'file:///private', 'http://localhost?token=x']) assert.throws(() => localApiBase(value))
  assert.equal(localApiBase('http://127.0.0.1:8000/'), 'http://127.0.0.1:8000')
})
test('unpaired calls never hit network; collector role cannot pair as manager', async () => {
  let calls = 0
  const api = createLocalApi('http://localhost:8000', async () => { calls++; return ok({ role: 'collector' }) })
  await assert.rejects(api.get('/items'), /NOT_PAIRED/); assert.equal(calls, 0)
  await assert.rejects(api.pair(key), /ADMIN_KEY_REQUIRED/)
  await assert.rejects(api.get('/data/backup'), /NOT_PAIRED/)
})
test('paired requests use Bearer only, no cookies or redirect following; disconnect forgets key', async () => {
  const requests = []
  const api = createLocalApi('http://localhost:8000', async (url, options) => { requests.push({ url, options }); return ok(url.endsWith('/session') ? { role: 'admin' } : { items: [] }) })
  await api.pair(key); await api.get('/items', { params: { page: 1 } })
  assert.equal(requests[1].options.headers.Authorization, 'Bearer ' + key)
  assert.equal(requests[1].url.includes(key), false)
  assert.equal(requests[1].options.credentials, 'omit'); assert.equal(requests[1].options.redirect, 'error')
  assert.equal(requests[1].options.cache, 'no-store')
  await assert.rejects(api.get('//remote.test/items'), /INVALID_API_PATH/)
  api.disconnect(); await assert.rejects(api.get('/items'), /NOT_PAIRED/)
})
test('disconnect invalidates an already received but delayed response', async () => {
  let release
  const api = createLocalApi('http://localhost:8000', async url => url.endsWith('/session') ? ok({ role: 'admin' }) : new Promise(resolve => { release = resolve }))
  await api.pair(key)
  const pending = api.get('/items'); api.disconnect(); release(ok({ private: 'old session data' }))
  await assert.rejects(pending, { name: 'AbortError' })
})
test('failed authentication never establishes a session', async () => {
  const api = createLocalApi('http://localhost:8000', async () => new Response('{}', { status: 401 }))
  await assert.rejects(api.pair(key), /HTTP_401/)
  await assert.rejects(api.get('/items'), /NOT_PAIRED/)
})
test('revoked server key clears the established session', async () => {
  const api = createLocalApi('http://localhost:8000', async url => url.endsWith('/session') ? ok({ role: 'admin' }) : new Response('{}', { status: 401 }))
  await api.pair(key)
  await assert.rejects(api.get('/items'), /HTTP_401/)
  await assert.rejects(api.get('/items'), /NOT_PAIRED/)
})

test('only the known snapshot conflict exposes a recovery code; arbitrary server text stays private', async () => {
  let detail = { code: 'items_snapshot_expired', message: 'private server diagnostic' }
  const api = createLocalApi('http://localhost:8000', async url => url.endsWith('/session')
    ? ok({ role: 'admin' }) : Response.json({ detail }, { status: 409 }))
  await api.pair(key)
  await assert.rejects(api.get('/items'), error => error.message === 'HTTP_409'
    && error.code === 'items_snapshot_expired' && !JSON.stringify(error).includes('private'))
  detail = { code: 'unknown_error', message: key }
  await assert.rejects(api.get('/items'), error => error.message === 'HTTP_409'
    && error.code === undefined && !JSON.stringify(error).includes(key))
})

test('a non-JSON conflict remains a bounded HTTP error', async () => {
  const api = createLocalApi('http://localhost:8000', async url => url.endsWith('/session')
    ? ok({ role: 'admin' }) : new Response('conflict', { status: 409 }))
  await api.pair(key)
  await assert.rejects(api.get('/items'), { message: 'HTTP_409' })
})

// Keys stay in this page's memory; they are never placed in URLs or browser storage.
export function localApiBase(value) {
  const url = new URL(value)
  if (!['http:', 'https:'].includes(url.protocol) || !['localhost', '127.0.0.1', '[::1]'].includes(url.hostname)
    || url.username || url.password || url.search || url.hash || url.pathname !== '/') throw new Error('INVALID_LOCAL_API')
  return url.origin
}

export function createLocalApi(base, fetchImpl = fetch) {
  const baseURL = localApiBase(base)
  let token = '', generation = 0
  const controllers = new Set()
  function disconnect() {
    token = ''; generation++
    for (const controller of controllers) controller.abort()
    controllers.clear()
  }
  async function request(method, path, data, options = {}, candidate = token) {
    if (!candidate) throw new Error('NOT_PAIRED')
    if (!path.startsWith('/') || path.startsWith('//')) throw new Error('INVALID_API_PATH')
    const url = new URL(path, baseURL)
    if (url.origin !== baseURL) throw new Error('INVALID_API_PATH')
    for (const [key, value] of Object.entries(options.params || {})) url.searchParams.set(key, String(value))
    const epoch = generation, controller = new AbortController()
    controllers.add(controller)
    const signals = [controller.signal, AbortSignal.timeout(30000)]
    if (options.signal) signals.push(options.signal)
    try {
      const body = data instanceof Blob ? data : data === undefined ? undefined : JSON.stringify(data)
      const response = await fetchImpl(url.href, {
        method, body, signal: AbortSignal.any(signals), redirect: 'error', credentials: 'omit', cache: 'no-store',
        headers: { ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
          ...options.headers, Authorization: 'Bearer ' + candidate },
      })
      if (!response.ok) {
        if (response.status === 401 && epoch === generation) disconnect()
        const error = new Error('HTTP_' + response.status)
        // Only expose this public recovery signal, never arbitrary server text.
        if (response.status === 409) {
          const problem = await response.json().catch(() => null)
          if (problem?.detail?.code === 'items_snapshot_expired') error.code = 'items_snapshot_expired'
        }
        throw error
      }
      const result = options.responseType === 'blob' ? await response.blob() : await response.json()
      if (epoch !== generation) throw new DOMException('Session changed', 'AbortError')
      return { data: result }
    } finally { controllers.delete(controller) }
  }
  return {
    disconnect,
    async pair(candidate) {
      disconnect()
      if (!/^[A-Za-z0-9_-]{43,128}$/.test(candidate)) throw new Error('INVALID_KEY')
      const response = await request('GET', '/session', undefined, {}, candidate)
      if (response.data?.role !== 'admin') throw new Error('ADMIN_KEY_REQUIRED')
      token = candidate
    },
    get: (path, options) => request('GET', path, undefined, options),
    post: (path, data, options) => request('POST', path, data, options),
    delete: (path, options) => request('DELETE', path, undefined, options),
  }
}

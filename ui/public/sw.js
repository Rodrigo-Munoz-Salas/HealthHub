const CACHE = 'hh-cache-v4';
const OFFLINE_URLS = ['/', '/ai.html']; // keep short in dev

self.addEventListener('install', (event) => {
  event.waitUntil(caches.open(CACHE).then(c => c.addAll(OFFLINE_URLS)));
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil((async () => {
    const keys = await caches.keys();
    await Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)));
    await self.clients.claim();
  })());
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  const url = new URL(req.url);

  // Only handle GET + http/https + same-origin
  if (req.method !== 'GET') return;
  if (url.protocol !== 'http:' && url.protocol !== 'https:') return;  // <-- fixes chrome-extension
  if (url.origin !== self.location.origin) return;

  // Never intercept API calls; UI handles queue/sync
  if (url.pathname.startsWith('/api/')) return;

  event.respondWith(
    caches.match(req).then(cached => {
      if (cached) return cached;
      return fetch(req).then(resp => {
        // cache only successful, basic (same-origin) responses
        if (resp.ok && resp.type === 'basic') {
          const copy = resp.clone();
          caches.open(CACHE).then(c => c.put(req, copy)).catch(() => {});
        }
        return resp;
      }).catch(() => {
        // Fallback: if it's a navigation, return shell; else empty 504
        if (req.mode === 'navigate') return caches.match('/');
        return new Response('', { status: 504, statusText: 'Offline' });
      });
    })
  );
});

self.addEventListener('sync', (event) => {
  if (event.tag === 'hh-sync') {
    event.waitUntil(
      self.clients.matchAll({ includeUncontrolled: true, type: 'window' })
        .then(clients => clients.forEach(c => c.postMessage({ type: 'TRIGGER_SYNC' })))
    );
  }
});

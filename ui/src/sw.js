// Simple service worker scaffold with optional Background Sync message to page.
self.addEventListener('install', () => self.skipWaiting())
self.addEventListener('activate', (e) => e.waitUntil(self.clients.claim()))

self.addEventListener('sync', async (event) => {
  if (event.tag === 'sync-queue') {
    event.waitUntil((async () => {
      const clients = await self.clients.matchAll({ includeUncontrolled: true })
      for (const c of clients) c.postMessage({ type: 'FLUSH_QUEUE' })
    })())
  }
})

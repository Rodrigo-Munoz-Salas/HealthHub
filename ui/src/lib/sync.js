// src/lib/sync.js
import { Storage } from './storage.js'
import { peekQueue, clearQueueItems } from './queue.js'

const SYNC_ENDPOINT = '/api/sync'  // your Go backend endpoint

// Try to sync queued items to server
export async function trySync() {
  const q = peekQueue()
  if (!q.length) return { synced: 0 }

  // batch first 50 items
  const batch = q.slice(0, 50)
  try {
    const res = await fetch(SYNC_ENDPOINT, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ items: batch }),
    })
    if (!res.ok) throw new Error('sync failed: ' + res.status)

    // clear successful items
    clearQueueItems(batch.map((i) => i.id))
    Storage.setLastSync(new Date().toISOString())

    return { synced: batch.length }
  } catch (e) {
    // don’t clear the queue if it fails
    return { error: e.message }
  }
}

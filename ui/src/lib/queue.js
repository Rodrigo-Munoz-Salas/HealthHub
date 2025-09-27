import { put, getAll, deleteByIds } from './idb'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080'

// Enqueue a job for later sync (works offline).
export async function enqueue(job) {
  await put('syncQueue', { ...job, createdAt: Date.now() })
}

// Attempt to flush queued jobs to server (call on load + 'online').
export async function flushQueue() {
  const items = await getAll('syncQueue')
  if (!items.length) return

  // 1) Users (batch upsert)
  const userJobs = items.filter(i => i.type === 'createUser')
  if (userJobs.length) {
    const payload = userJobs.map(j => j.payload)
    try {
      const res = await fetch(`${API_URL}/sync/users`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ users: payload })
      })
      if (!res.ok) throw new Error('sync/users failed')
      const toDelete = userJobs.map(j => j.id).filter(Boolean)
      if (toDelete.length) await deleteByIds('syncQueue', toDelete)
      console.info('[Queue] Flushed users:', toDelete.length)
    } catch (e) {
      console.warn('[Queue] Flush failed:', e.message)
    }
  }

  // Add other job types later (e.g., queued chats).
}

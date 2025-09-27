import { Storage } from './storage.js'

function genId() {
  return (crypto?.randomUUID?.() ?? Math.random().toString(36).slice(2) + Date.now())
}

export function enqueue(type, payload) {
  const item = { id: genId(), type, payload, ts: new Date().toISOString() }
  const q = Storage.getQueue()
  q.push(item)
  Storage.setQueue(q)
  return item
}

export function peekQueue() {
  return Storage.getQueue()
}

export function clearQueueItems(ids) {
  const set = new Set(ids)
  const remaining = Storage.getQueue().filter(i => !set.has(i.id))
  Storage.setQueue(remaining)
}

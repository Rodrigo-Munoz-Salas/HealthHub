import React, { useEffect, useState } from 'react'
import { put, get } from './lib/idb'
import { enqueue, flushQueue } from './lib/queue'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080'

export default function App() {
  const [form, setForm] = useState({ name:'', age:'', weight:'', height:'', medical_conditions:'' })
  const [createdId, setCreatedId] = useState('')

  useEffect(() => {
    flushQueue()
    const onOnline = () => flushQueue()
    window.addEventListener('online', onOnline)
    navigator.serviceWorker?.addEventListener?.('message', (evt) => {
      if (evt.data?.type === 'FLUSH_QUEUE') flushQueue()
    })
    return () => window.removeEventListener('online', onOnline)
  }, [])

  const handleChange = (e) => setForm(f => ({ ...f, [e.target.name]: e.target.value }))

  const createUser = async (e) => {
    e.preventDefault()
    const uuid = crypto.randomUUID()
    const user = {
      uuid,
      name: form.name,
      age: Number(form.age) || 0,
      weight: Number(form.weight) || 0,
      height: Number(form.height) || 0,
      medical_conditions: form.medical_conditions.split(',').map(s => s.trim()).filter(Boolean)
    }

    // 1) Save locally for offline use
    await put('users', { ...user, id: user.uuid })

    // 2) Enqueue for server sync (works both offline/online)
    await enqueue({ type: 'createUser', payload: user })

    // 3) Try flush if online
    if (navigator.onLine) await flushQueue()

    alert(`User saved locally! Local UUID: ${uuid}`)
    setCreatedId(uuid)
  }

  const openAIWindow = () => window.open('/ai.html', '_blank')

  return (
    <div style={{ fontFamily:'Inter, system-ui, Arial', padding:24, maxWidth:900, margin:'0 auto' }}>
      <h1>HealthHub Dashboard (Offline-first)</h1>

      <section style={{ marginTop:24, padding:16, border:'1px solid #ddd', borderRadius:12 }}>
        <h2>Create User (Local → Queue → Sync)</h2>
        <form onSubmit={createUser} style={{ display:'grid', gap:12, maxWidth:480 }}>
          <input name="name" placeholder="Name" value={form.name} onChange={handleChange} required />
          <input name="age" placeholder="Age" value={form.age} onChange={handleChange} />
          <input name="weight" placeholder="Weight (kg)" value={form.weight} onChange={handleChange} />
          <input name="height" placeholder="Height (cm)" value={form.height} onChange={handleChange} />
          <input name="medical_conditions" placeholder="Medical conditions (comma-separated)" value={form.medical_conditions} onChange={handleChange} />
          <button type="submit">Create User</button>
        </form>
        {createdId && <p style={{ marginTop:8 }}>Last created (local) UUID: <code>{createdId}</code></p>}
        <p style={{ marginTop:8 }}>Status: {navigator.onLine ? 'Online' : 'Offline'}</p>
      </section>

      <section style={{ marginTop:24, padding:16, border:'1px solid #ddd', borderRadius:12 }}>
        <h2>Use AI Agent</h2>
        <p>Opens a new window to verify UUID and chat. (Patient info is sent from local storage.)</p>
        <button onClick={openAIWindow}>Use AI</button>
      </section>
    </div>
  )
}

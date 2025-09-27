import React, { useState } from 'react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080'
const AGENT_URL = import.meta.env.VITE_AGENT_URL || 'http://localhost:8001'

export default function App() {
  const [form, setForm] = useState({
    name: '', age: '', weight: '', height: '', medical_conditions: ''
  })
  const [createdId, setCreatedId] = useState('')

  const handleChange = (e) => {
    const { name, value } = e.target
    setForm(prev => ({ ...prev, [name]: value }))
  }

  const createUser = async (e) => {
    e.preventDefault()
    try {
      const payload = {
        name: form.name,
        age: Number(form.age) || 0,
        weight: Number(form.weight) || 0,
        height: Number(form.height) || 0,
        medical_conditions: form.medical_conditions
          .split(',')
          .map(s => s.trim())
          .filter(Boolean)
      }
      const res = await fetch(`${API_URL}/users`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      })
      const data = await res.json()
      if (!res.ok) throw new Error(data.error || 'Request failed')
      alert(`User created! UUID: ${data.uuid}`)
      setCreatedId(data.uuid)
    } catch (err) {
      alert(`Create failed: ${err.message}`)
    }
  }

  const openAIWindow = () => {
    const url = `/ai.html`
    window.open(url, '_blank')
  }

  return (
    <div style={{ fontFamily: 'Inter, system-ui, Arial', padding: 24, maxWidth: 900, margin: '0 auto' }}>
      <h1>HealthHub Dashboard</h1>

      <section style={{ marginTop: 24, padding: 16, border: '1px solid #ddd', borderRadius: 12 }}>
        <h2>Create User (Go API)</h2>
        <form onSubmit={createUser} style={{ display: 'grid', gap: 12, maxWidth: 480 }}>
          <input name="name" placeholder="Name" value={form.name} onChange={handleChange} required />
          <input name="age" placeholder="Age" value={form.age} onChange={handleChange} />
          <input name="weight" placeholder="Weight (kg)" value={form.weight} onChange={handleChange} />
          <input name="height" placeholder="Height (cm)" value={form.height} onChange={handleChange} />
          <input name="medical_conditions" placeholder="Medical conditions (comma-separated)" value={form.medical_conditions} onChange={handleChange} />
          <button type="submit">Create User</button>
        </form>
        {createdId && <p style={{ marginTop: 8 }}>Last created UUID: <code>{createdId}</code></p>}
      </section>

      <section style={{ marginTop: 24, padding: 16, border: '1px solid #ddd', borderRadius: 12 }}>
        <h2>Use AI Agent</h2>
        <p>Opens a new window to verify UUID and chat with the agent.</p>
        <button onClick={openAIWindow}>Use AI</button>
      </section>
    </div>
  )
}

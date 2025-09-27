// src/App.jsx
import React, { useEffect, useState } from 'react'
import { put } from './lib/idb.js'
import { enqueue } from './lib/queue.js'
import { useSync } from './context/SyncContext.jsx'
import { Storage } from './lib/storage.js'
import { sendPatientToAgent } from './lib/agentBridge.js'

export default function App() {
  const { online, syncing, lastSync, forceSync } = useSync()

  const [user, setUser] = useState(() => Storage.getUser() || null)
  const [open, setOpen] = useState(false)
  const [isEdit, setIsEdit] = useState(false)

  const [form, setForm] = useState({
    name: '', age: '', weight: '', height: '', medical_conditions: '',
  })

  useEffect(() => {
    if (!user) return
    setForm({
      name: user.name || '',
      age: user.age || '',
      weight: user.weight || '',
      height: user.height || '',
      medical_conditions: (user.medical_conditions || []).join(', '),
    })
  }, []) // mount only

  const handleChange = (e) =>
    setForm((f) => ({ ...f, [e.target.name]: e.target.value }))

  const openCreate = () => {
    setIsEdit(false)
    setForm({ name:'', age:'', weight:'', height:'', medical_conditions:'' })
    setOpen(true)
  }

  const openEdit = () => {
    if (!user) return
    setIsEdit(true)
    setForm({
      name: user.name || '',
      age: user.age || '',
      weight: user.weight || '',
      height: user.height || '',
      medical_conditions: (user.medical_conditions || []).join(', '),
    })
    setOpen(true)
  }

  const saveUser = async (e) => {
    e.preventDefault()
    const uuid = isEdit && user?.uuid ? user.uuid : crypto.randomUUID()
    const record = {
      uuid,
      name: form.name.trim(),
      age: Number(form.age) || 0,
      weight: Number(form.weight) || 0,
      height: Number(form.height) || 0,
      medical_conditions: form.medical_conditions.split(',').map(s=>s.trim()).filter(Boolean),
    }

    await put('users', { ...record, id: record.uuid })
    Storage.setUser(record)
    await enqueue(isEdit ? 'updateUser' : 'createUser', record)
    if (navigator.onLine) await forceSync()

    setUser(record)
    setOpen(false)
  }

  // Auto-send context to agent on mount + any localStorage change
  useEffect(() => {
    sendPatientToAgent()
    const onStorage = (e) => {
      if (e.key === Storage.K.user || e.key === Storage.K.patient) {
        if (e.key === Storage.K.user) setUser(Storage.getUser() || null)
        sendPatientToAgent()
      }
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [])

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900 dark:bg-zinc-950 dark:text-zinc-100">
      {/* Header with inline Status card + Create User */}
      <header className="sticky top-0 z-20 border-b border-zinc-200/60 bg-white/80 backdrop-blur dark:border-zinc-800 dark:bg-zinc-950/70">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-5 py-3">
          <div className="flex items-center gap-3">
            <Logo />
            <h1 className="text-lg font-semibold tracking-tight">HealthHub</h1>
          </div>
          <div className="flex items-center gap-3">
            <StatusInline online={online} syncing={syncing} lastSync={lastSync} />
            <button
              onClick={openCreate}
              className="rounded-xl bg-blue-600 px-3 py-1.5 text-sm text-white hover:bg-blue-700"
            >
              Create User
            </button>
          </div>
        </div>
      </header>

      {/* Main content: User card then Console (both full width) */}
      <main className="mx-auto max-w-6xl px-5 py-6 space-y-6">
        <Card>
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-sm font-semibold">User Details</h3>
            <button
              onClick={openEdit}
              className="rounded-lg border border-zinc-300 px-2.5 py-1 text-xs hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-900"
            >
              Update
            </button>
          </div>
          <UserDetailsCard user={user} />
        </Card>

        <AIConsole />
      </main>

      {/* Create/Update Modal */}
      {open && (
        <Modal onClose={() => setOpen(false)} title={isEdit ? 'Update User' : 'Create User'}>
          <form onSubmit={saveUser} className="mt-3 grid grid-cols-1 gap-4 md:grid-cols-2">
            <TextField label="Full name" name="name" value={form.name} onChange={handleChange} required />
            <TextField label="Age" name="age" value={form.age} onChange={handleChange} inputMode="numeric" />
            <TextField label="Weight (kg)" name="weight" value={form.weight} onChange={handleChange} inputMode="numeric" />
            <TextField label="Height (cm)" name="height" value={form.height} onChange={handleChange} inputMode="numeric" />
            <div className="md:col-span-2">
              <TextField
                label="Medical conditions (comma-separated)"
                name="medical_conditions"
                value={form.medical_conditions}
                onChange={handleChange}
                placeholder="Asthma, Hypertension"
              />
            </div>
            <div className="mt-2 flex items-center gap-3 md:col-span-2">
              <button
                type="submit"
                className="inline-flex items-center justify-center rounded-xl bg-blue-600 px-4 py-2 font-medium text-white shadow-sm ring-1 ring-inset ring-blue-600/30 transition hover:bg-blue-700"
              >
                {isEdit ? 'Save changes' : 'Save locally & queue'}
              </button>
              <button
                type="button"
                onClick={() => setOpen(false)}
                className="rounded-xl border border-zinc-300 px-4 py-2 text-sm hover:bg-zinc-100 dark:border-zinc-700 dark:hover:bg-zinc-900"
              >
                Cancel
              </button>
            </div>
          </form>
        </Modal>
      )}
    </div>
  )
}

/* ---------- Inline Status Card ---------- */

function StatusInline({ online, syncing, lastSync }) {
  const label = online ? 'Online' : 'Offline'
  const msg = online
    ? (lastSync ? `Last synced: ${new Date(lastSync).toLocaleString()}` : (syncing ? 'Syncing…' : 'Waiting to sync…'))
    : 'Not syncing (offline)'

  return (
    <div className="flex items-center gap-3 rounded-2xl border border-zinc-200 bg-zinc-100/40 px-3 py-1.5 dark:border-zinc-800 dark:bg-zinc-900/60">
      <StatusPill online={online} syncing={syncing} />
      <span className="hidden sm:block text-sm text-zinc-700 dark:text-zinc-300">{msg}</span>
    </div>
  )
}

/* ---------- Console ---------- */

function AIConsole() {
  const [input, setInput] = React.useState('')
  const [messages, setMessages] = React.useState([])

  async function send() {
    const text = input.trim()
    if (!text) return
    setMessages((m) => [...m, { role: 'user', content: text, ts: Date.now() }])
    setInput('')
    try {
      // TODO connect to python chat endpoint
      const data = { reply: 'Agent reply placeholder.' }
      setTimeout(() => {
        setMessages((m) => [...m, { role: 'assistant', content: data.reply, ts: Date.now() }])
      }, 120)
    } catch {
      setMessages((m) => [...m, { role: 'assistant', content: 'Failed to reach agent.', ts: Date.now() }])
    }
  }

  return (
    <Card>
      <h2 className="mb-4 text-base font-semibold">AI Console</h2>

      <div className="rounded-2xl border border-zinc-200 p-4 dark:border-zinc-800">
        {/* Chat list - user right, assistant left */}
        <div className="mb-3 max-h-[48vh] space-y-3 overflow-auto pr-1">
          {messages.length === 0 && (
            <div className="text-sm text-zinc-500 dark:text-zinc-400">No messages yet.</div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={[
                  'max-w-[75%] rounded-2xl px-3 py-2 text-sm shadow-sm',
                  m.role === 'user'
                    ? 'bg-blue-600 text-white'
                    : 'bg-zinc-100 text-zinc-900 dark:bg-zinc-800 dark:text-zinc-100',
                ].join(' ')}
              >
                {m.content}
              </div>
            </div>
          ))}
        </div>

        {/* Composer */}
        <div className="grid gap-3 md:grid-cols-[1fr_auto]">
          <input
            className="w-full rounded-xl border border-zinc-300 bg-white px-3 py-2 text-sm outline-none hover:border-zinc-400 focus:border-blue-500 dark:border-zinc-700 dark:bg-zinc-950"
            placeholder="How are you feeling?"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), send())}
          />
          <button
            onClick={send}
            className="rounded-xl bg-purple-600 px-3 py-2 text-sm font-medium text-white shadow-sm ring-1 ring-inset ring-purple-600/30 hover:bg-purple-700"
          >
            Send
          </button>
        </div>

        <p className="mt-3 text-xs text-zinc-500 dark:text-zinc-400">
          Patient context is sent automatically from local storage.
        </p>
      </div>
    </Card>
  )
}

/* ---------- UI Primitives ---------- */

function Card({ children, className = '' }) {
  return (
    <div className={`rounded-2xl border border-zinc-200 bg-white p-5 shadow-sm dark:border-zinc-800 dark:bg-zinc-900 ${className}`}>
      {children}
    </div>
  )
}

function TextField({ label, ...props }) {
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-zinc-700 dark:text-zinc-300">{label}</span>
      <input
        {...props}
        className="w-full rounded-xl border border-zinc-300 bg-white px-3 py-2 text-sm outline-none hover:border-zinc-400 focus:border-blue-500 dark:border-zinc-700 dark:bg-zinc-950"
      />
    </label>
  )
}

function StatusPill({ online, syncing }) {
  return (
    <div
      className={[
        'inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-medium',
        syncing
          ? 'bg-blue-600/10 text-blue-700 dark:text-blue-400'
          : online
          ? 'bg-emerald-600/10 text-emerald-700 dark:text-emerald-400'
          : 'bg-amber-600/10 text-amber-700 dark:text-amber-400',
      ].join(' ')}
      title={syncing ? 'Syncing queued items…' : online ? 'You are online' : 'Offline mode'}
    >
      <span className="relative flex h-2 w-2">
        <span className={['absolute inline-flex h-full w-full rounded-full opacity-75', syncing ? 'animate-ping bg-blue-500' : online ? 'bg-emerald-500' : 'bg-amber-500'].join(' ')} />
        <span className={['relative inline-flex h-2 w-2 rounded-full', syncing ? 'bg-blue-600' : online ? 'bg-emerald-600' : 'bg-amber-600'].join(' ')} />
      </span>
      {syncing ? 'Syncing…' : online ? 'Online' : 'Offline'}
    </div>
  )
}

function Logo() {
  return (
    <svg width="22" height="22" viewBox="0 0 24 24" className="text-blue-600 dark:text-blue-400">
      <path d="M12 2c5.523 0 10 4.477 10 10a9.99 9.99 0 0 1-5.74 9.09c-.3.14-.65.02-.8-.28l-1.62-3.23a1 1 0 0 0-.9-.55H11a1 1 0 0 1-1-1v-2H8a1 1 0 0 1-1-1v-2H5a1 1 0 0 1-.9-1.45l3.1-6.2A10 10 0 0 1 12 2Z" fill="currentColor"/>
    </svg>
  )
}

function Modal({ title, children, onClose }) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" onClick={onClose} />
      <div className="relative w-full max-w-2xl rounded-2xl border border-zinc-200 bg-white p-6 shadow-xl dark:border-zinc-800 dark:bg-zinc-900">
        <div className="mb-3 flex items-center justify-between">
          <h3 className="text-base font-semibold">{title}</h3>
          <button onClick={onClose} className="rounded-lg px-2 py-1 text-sm hover:bg-zinc-100 dark:hover:bg-zinc-800">✕</button>
        </div>
        {children}
      </div>
    </div>
  )
}

function UserDetailsCard({ user }) {
  if (!user) {
    return (
      <div className="rounded-xl border border-dashed border-zinc-300 p-4 text-sm text-zinc-500 dark:border-zinc-700">
        No user saved yet. Use <span className="font-medium">Create User</span> to add one.
      </div>
    )
  }
  return (
    <div className="grid grid-cols-1 gap-3">
      <Detail label="Name" value={user.name || '—'} />
      <div className="grid grid-cols-2 gap-3">
        <Detail label="Age" value={user.age || '—'} />
        <Detail label="Height (cm)" value={user.height || '—'} />
      </div>
      <Detail label="Weight (kg)" value={user.weight || '—'} />
      <Detail label="Medical conditions" value={(user.medical_conditions || []).join(', ') || '—'} />
    </div>
  )
}

function Detail({ label, value }) {
  return (
    <div className="rounded-xl border border-zinc-200 bg-white px-3 py-2 text-sm dark:border-zinc-800 dark:bg-zinc-950">
      <div className="text-xs text-zinc-500 dark:text-zinc-400">{label}</div>
      <div className="mt-0.5 font-medium break-words">{value}</div>
    </div>
  )
}

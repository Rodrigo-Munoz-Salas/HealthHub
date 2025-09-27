import { useState } from 'react';
import { Storage } from '../lib/storage';
import { enqueue } from '../lib/queue';

export default function AccountForm() {
  const existing = Storage.getUser();
  const [form, setForm] = useState(existing ?? { mode: 'guest', name: '', email: '' });

  function save(mode) {
    const user = { ...form, mode, updatedAt: new Date().toISOString() };
    Storage.setUser(user);

    // Enqueue for server sync regardless of online/offline
    const type = existing ? 'updateUser' : 'createUser';
    enqueue(type, { name: user.name, email: user.email, mode: user.mode });

    alert('Saved locally. Will sync when online.');
  }

  return (
    <div className="max-w-md space-y-4">
      <div className="space-y-2">
        <label className="block text-sm">Name</label>
        <input className="w-full rounded-lg border px-3 py-2" value={form.name}
               onChange={e=>setForm(f=>({...f, name:e.target.value}))}/>
      </div>
      <div className="space-y-2">
        <label className="block text-sm">Email (optional)</label>
        <input className="w-full rounded-lg border px-3 py-2" value={form.email}
               onChange={e=>setForm(f=>({...f, email:e.target.value}))}/>
      </div>

      <div className="flex gap-2">
        <button onClick={()=>save('guest')} className="rounded-lg bg-zinc-800 px-4 py-2 text-white">Continue as guest</button>
        <button onClick={()=>save('registered')} className="rounded-lg bg-blue-600 px-4 py-2 text-white">Create account</button>
      </div>
    </div>
  );
}

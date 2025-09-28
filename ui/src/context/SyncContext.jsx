// src/context/SyncContext.jsx
import { createContext, useContext, useEffect, useState } from 'react';
import { useOnlineStatus } from '../hooks/useOnlineStatus';
import { trySync } from '../lib/sync';
import { Storage } from '../lib/storage';

const SyncCtx = createContext({
  online: false,
  syncing: false,
  lastSync: null,
  forceSync: async () => {},
});

export function SyncProvider({ children }) {
  const online = useOnlineStatus();
  const [syncing, setSyncing] = useState(false);
  // ✅ Read the parsed value via Storage helper (not raw localStorage)
  const [lastSync, setLastSync] = useState(Storage.getLastSync());

  async function forceSync() {
    if (!online || syncing) return;
    setSyncing(true);
    const res = await trySync();
    setSyncing(false);

    // ✅ Re-read via Storage helper to avoid Invalid Date
    const ls = Storage.getLastSync();
    if (ls) setLastSync(ls);
    return res;
  }

  // 🔔 Listen for background sync trigger from the SW (optional)
  useEffect(() => {
    const h = () => forceSync();
    window.addEventListener('hh-sync', h);
    return () => window.removeEventListener('hh-sync', h);
  }, []);

  // 🔄 Auto-sync when coming online
  useEffect(() => {
    if (!online) return;
    forceSync();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [online]);

  return (
    <SyncCtx.Provider value={{ online, syncing, lastSync, forceSync }}>
      {children}
    </SyncCtx.Provider>
  );
}

export const useSync = () => useContext(SyncCtx);

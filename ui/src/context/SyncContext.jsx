import { createContext, useContext, useEffect, useState } from 'react';
import { useOnlineStatus } from '../hooks/useOnlineStatus';
import { trySync } from '../lib/sync';

const SyncCtx = createContext({
  online: false,
  syncing: false,
  lastSync: null,
  forceSync: async () => {},
});

export function SyncProvider({ children }) {
  const online = useOnlineStatus();
  const [syncing, setSyncing] = useState(false);
  const [lastSync, setLastSync] = useState(localStorage.getItem('hh:lastSync'));

  async function forceSync() {
    if (!online || syncing) return;
    setSyncing(true);
    const res = await trySync();
    setSyncing(false);
    const ls = localStorage.getItem('hh:lastSync');
    if (ls) setLastSync(ls);
    return res;
  }

  // 🔔 Listen for service worker background sync trigger
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

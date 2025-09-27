import { useSync } from '../context/SyncContext';

export default function SyncBanner() {
  const { online, syncing, lastSync, forceSync } = useSync();
  return (
    <div className={`w-full px-4 py-2 text-sm ${online ? 'bg-emerald-600' : 'bg-zinc-700'} text-white flex items-center justify-between`}>
      <span>
        {online ? (syncing ? 'Syncing…' : 'Online') : 'Offline mode'}
        {lastSync ? ` • Last sync: ${new Date(lastSync).toLocaleString()}` : ''}
      </span>
      <button onClick={forceSync} className="rounded-md bg-white/10 px-3 py-1 hover:bg-white/20">
        Sync now
      </button>
    </div>
  );
}

// Simple typed wrappers around localStorage
const K = {
  user: 'hh:user',
  patient: 'hh:patient',
  queue: 'hh:queue',
  lastSync: 'hh:lastSync',
};

function read(key, fallback = null) {
  try { return JSON.parse(localStorage.getItem(key)) ?? fallback; }
  catch { return fallback; }
}

function write(key, value) {
  localStorage.setItem(key, JSON.stringify(value));
}

export const Storage = {
  K,
  getUser: () => read(K.user),
  setUser: (u) => write(K.user, u),

  getPatient: () => read(K.patient),
  setPatient: (p) => write(K.patient, p),

  getQueue: () => read(K.queue, []),
  setQueue: (q) => write(K.queue, q),

  getLastSync: () => read(K.lastSync),
  setLastSync: (ts) => write(K.lastSync, ts),
};

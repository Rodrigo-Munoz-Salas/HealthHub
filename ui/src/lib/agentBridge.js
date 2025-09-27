// src/lib/agentBridge.js
import { Storage } from './storage.js';

export function getPatientForAgent() {
  const patient = Storage.getPatient() ?? {};
  const user = Storage.getUser() ?? {};
  return { user: { name: user.name, mode: user.mode, email: user.email, uuid: user.uuid }, patient };
}

export async function sendPatientToAgent() {
  const payload = getPatientForAgent();
  try {
    await fetch('/py/agent/patient-intake', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    return { ok: true };
  } catch (e) {
    return { ok: false, error: e.message };
  }
}

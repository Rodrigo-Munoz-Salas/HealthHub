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
      body: JSON.stringify({ user: payload.user, patient: payload.patient }),
    });
    return { ok: true };
  } catch (e) {
    return { ok: false, error: e.message };
  }
}

export async function sendChatToAgent(message) {
  const user = Storage.getUser() ?? {};
  const patient = Storage.getPatient() ?? {};
  const res = await fetch('/py/agent/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: user.uuid, message, patient_info: patient }),
  });
  if (!res.ok) throw new Error(`Agent error ${res.status}`);
  return res.json();
}

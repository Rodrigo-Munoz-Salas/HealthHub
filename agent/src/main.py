# agent/src/main.py
from typing import Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

app = FastAPI(title="HealthHub Agent")

# Allow Vite dev + preview
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------

class AgentRequest(BaseModel):
    user_id: str
    message: str
    patient_info: Optional[Dict[str, Any]] = None  # sent by UI from IndexedDB

# Payload shape used by UI's agentBridge.sendPatientToAgent()
class PatientIntake(BaseModel):
    user: Optional[Dict[str, Any]] = None  # { name, mode, email, uuid }
    patient: Optional[Dict[str, Any]] = None

# ---------- Health ----------

@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Patient context intake (used automatically by UI) ----------

@app.post("/agent/patient-intake")
async def agent_patient_intake(payload: PatientIntake):
    uid = (payload.user or {}).get("uuid")
    logging.info(
        "[Intake] received user=%s, patient_keys=%s",
        uid,
        list((payload.patient or {}).keys()),
    )
    # TODO: stash to your in-memory/session store if you plan to make agents stateful
    return {"ok": True, "received": payload.model_dump()}

# ---------- Agents (simple echo placeholders) ----------

@app.post("/agent/profile")
def agent_profile(req: AgentRequest):
    logging.info(
        "[Profile] user=%s, has_patient_info=%s",
        req.user_id, bool(req.patient_info),
    )
    # TODO: integrate Google GenAI (profile-aware prompt)
    return {
        "agent": "profile",
        "reply": f"Echo: {req.message}",
        "has_patient_info": bool(req.patient_info),
    }

@app.post("/agent/triage")
def agent_triage(req: AgentRequest):
    logging.info(
        "[Triage] user=%s, has_patient_info=%s",
        req.user_id, bool(req.patient_info),
    )
    return {
        "agent": "triage",
        "reply": f"Echo: {req.message}",
        "has_patient_info": bool(req.patient_info),
    }

@app.post("/agent/qa")
def agent_qa(req: AgentRequest):
    logging.info(
        "[QA] user=%s, has_patient_info=%s",
        req.user_id, bool(req.patient_info),
    )
    return {
        "agent": "qa",
        "reply": f"Echo: {req.message}",
        "has_patient_info": bool(req.patient_info),
    }

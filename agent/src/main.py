from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
app = FastAPI(title="HealthHub Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class AgentRequest(BaseModel):
    user_id: str
    message: str
    patient_info: dict | None = None  # sent by UI from IndexedDB

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/agent/profile")
def agent_profile(req: AgentRequest):
    logging.info("[Profile] input received. user=%s, has_patient_info=%s",
                 req.user_id, bool(req.patient_info))
    # TODO: integrate Google GenAI (profile-aware prompt)
    return {"agent": "profile", "reply": f"Echo: {req.message}", "has_patient_info": bool(req.patient_info)}

@app.post("/agent/triage")
def agent_triage(req: AgentRequest):
    logging.info("[Triage] input received. user=%s, has_patient_info=%s",
                 req.user_id, bool(req.patient_info))
    return {"agent": "triage", "reply": f"Echo: {req.message}", "has_patient_info": bool(req.patient_info)}

@app.post("/agent/qa")
def agent_qa(req: AgentRequest):
    logging.info("[QA] input received. user=%s, has_patient_info=%s",
                 req.user_id, bool(req.patient_info))
    return {"agent": "qa", "reply": f"Echo: {req.message}", "has_patient_info": bool(req.patient_info)}

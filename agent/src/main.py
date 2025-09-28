# agent/src/main.py
import logging
from typing import Optional, Dict, Any
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from .rag_backend import init_rag, RAG_STATUS, rag_answer

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("agent")

app = FastAPI(title="HealthHub Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ---------------- Models ----------------

class ChatRequest(BaseModel):
    user_id: Optional[str] = None
    message: str
    patient_info: Optional[Dict[str, Any]] = None  # sent by UI

class IntakePayload(BaseModel):
    user: Dict[str, Any] = {}
    patient: Dict[str, Any] = {}

# ---------------- Startup ----------------

@app.on_event("startup")
def _startup():
    ok = init_rag()
    if ok:
        log.info("[Startup] Local RAG ready ✓")
    else:
        log.warning("[Startup] Local RAG NOT ready; will use rule-based fallback. Status=%s", RAG_STATUS)

# ---------------- Health ----------------

@app.get("/health")
def health():
    return {"status": "ok", "rag": RAG_STATUS}

# ---------------- Intake (unchanged) ----------------

@app.post("/agent/patient-intake")
def patient_intake(payload: IntakePayload):
    user = payload.user or {}
    patient = payload.patient or {}
    log.info("[Intake] user=%s, patient_keys=%s", user.get("uuid") or user.get("name"), list(patient.keys()))
    return {"ok": True}

# ---------------- Chat ----------------

def rule_fallback(text: str, patient: Dict[str, Any]) -> str:
    t = (text or "").lower()
    name = (patient or {}).get("name") or "there"

    red_flags = [
        ("chest pain", "Chest pain + trouble breathing is a medical emergency. Call 911 now."),
        ("shortness of breath", "Difficulty breathing can be serious. If severe or worsening, call 911."),
        ("can’t breathe", "Trouble breathing is an emergency. Call 911."),
        ("can't breathe", "Trouble breathing is an emergency. Call 911."),
        ("faint", "Fainting with injury or repeated episodes needs urgent evaluation."),
        ("stroke", "Stroke signs require 911 immediately."),
        ("choking", "Choking is life-threatening — call 911 and perform Heimlich if trained.")
    ]
    for key, msg in red_flags:
        if key in t:
            return msg

    return f"I've logged your symptoms, {name}. If symptoms worsen or new red flags appear (severe pain, trouble breathing, confusion), seek urgent care or call 911."

@app.post("/agent/chat")
def chat(req: ChatRequest):
    """
    If RAG is ready -> use it.
    Else -> rule fallback.
    """
    text = (req.message or "").strip()
    patient = (req.patient_info or {}).get("patient") or (req.patient_info or {})
    user = (req.patient_info or {}).get("user") or {}

    if not text:
        return {"reply": "Please describe how you're feeling.", "source": "validation"}

    try:
        if RAG_STATUS.get("system_ready"):
            res = rag_answer(text)

            if "error" not in res:
                reply = (
                    res.get("natural_response")
                    or res.get("response")
                    or "I've analyzed your message."
                )
                # 🔴 Log clearly which path was used
                log.info("[CHAT] USING RAG | et=%s | c911=%s | conf=%s",
                         res.get("emergency_type"),
                         res.get("call_911"),
                         res.get("confidence"))
                return {
                    "reply": reply,
                    "source": "rag",
                    "meta": {
                        "emergency_type": res.get("emergency_type"),
                        "call_911": res.get("call_911"),
                        "confidence": res.get("confidence"),
                        "processing_time": res.get("processing_time"),
                    },
                }
            else:
                log.warning("[CHAT] RAG error -> falling back to rules: %s", res["error"])

        # Rule fallback
        reply = rule_fallback(text, patient or user)
        log.info("[CHAT] USING RULES")
        return {"reply": reply, "source": "rules"}

    except Exception as e:
        log.exception("[CHAT] Unexpected error")
        return {
            "reply": "I hit an unexpected error. Please try again.",
            "source": "server_error",
        }


# agent/src/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import re
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
log = logging.getLogger("agent")

app = FastAPI(title="HealthHub Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- In-memory "context" ----------------

LATEST_CONTEXT: Dict[str, Any] = {
    "user": None,
    "patient": None,
    "updated_at": None,
}

class IntakePayload(BaseModel):
    user: Optional[Dict[str, Any]] = None
    patient: Optional[Dict[str, Any]] = None

class ChatPayload(BaseModel):
    user_id: Optional[str] = None
    message: str

# ---------------- Utilities ----------------

EMERGENCY_PATTERNS: Dict[str, List[str]] = {
    "chest_pain": [r"\bchest pain\b", r"\bpressure in (my|the) chest\b", r"\btight(?:ness)? in (my|the) chest\b"],
    "shortness_breath": [r"\bshort(ness)? of breath\b", r"\b(can'?t|cannot) breathe\b", r"\bbreath(ing)? difficulty\b", r"\bdyspnea\b"],
    "bleeding": [r"\b(cough(ing)? (up )?blood|bleed(ing)?)\b", r"\bspitting blood\b", r"\bblood in (my|the) (cough|sputum)\b"],
    "fainting": [r"\bfaint(ed|ing)?\b", r"\bpassed out\b", r"\bunconscious\b", r"\bcollapse(d)?\b"],
    "stroke": [r"\bfacial droop\b", r"\bslurred speech\b", r"\bweakness (on|in) (one|the) side\b", r"\bparalysis\b", r"\bFAST\b"],
    "choking": [r"\bchoking\b", r"\bairway (block|blocked)\b", r"\bcan('t|not) speak\b", r"\bfood stuck\b"],
    "severe_pain": [r"\bsevere pain\b", r"\b10\/10 pain\b"],
}

def match_any(text: str, patterns: List[str]) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in patterns)

def format_patient_blurb(ctx: Dict[str, Any]) -> str:
    user = ctx.get("user") or {}
    name = user.get("name") or "Patient"
    age = user.get("age")
    weight = user.get("weight")
    height = user.get("height")
    med = ", ".join(user.get("medical_conditions", []) or [])
    parts = [f"{name}"]
    if age: parts.append(f"age {age}")
    if height: parts.append(f"height {height} cm")
    if weight: parts.append(f"weight {weight} kg")
    if med: parts.append(f"conditions: {med}")
    return " | ".join(parts)

def rule_based_triage(message: str, ctx: Dict[str, Any]) -> str:
    """
    Short, actionable guidance. We keep it brief so it looks good in the chat bubble.
    """
    # Priority emergencies
    if match_any(message, EMERGENCY_PATTERNS["chest_pain"]) and match_any(message, EMERGENCY_PATTERNS["shortness_breath"]):
        return "Chest pain + trouble breathing is a medical emergency. Call 911 now and avoid exertion. Sit upright and stay calm."

    if match_any(message, EMERGENCY_PATTERNS["choking"]):
        return "Possible choking. If the person can’t speak or breathe: call 911 and perform the Heimlich maneuver if trained."

    if match_any(message, EMERGENCY_PATTERNS["stroke"]):
        return "Possible stroke. Call 911 immediately. Note the time symptoms began; faster treatment improves outcomes."

    if match_any(message, EMERGENCY_PATTERNS["fainting"]):
        return "Fainting with potential head injury or ongoing symptoms: call 911. If conscious, lie flat and elevate legs slightly."

    # Serious but not always 911
    if match_any(message, EMERGENCY_PATTERNS["bleeding"]):
        return "Coughing up blood requires urgent evaluation. If bleeding is heavy or breathing worsens, call 911; otherwise go to ER/urgent care now."

    if match_any(message, EMERGENCY_PATTERNS["shortness_breath"]):
        return "Shortness of breath needs prompt care. If severe or worsening, call 911. If mild, seek urgent care or ER today."

    if match_any(message, EMERGENCY_PATTERNS["severe_pain"]):
        return "Severe pain needs assessment. Use rest and avoid food if abdominal. Seek urgent care/ER if pain persists or worsens."

    # Default supportive reply
    blurb = format_patient_blurb(ctx)
    return f"I've logged your symptoms. {blurb}. If symptoms worsen or new red flags appear (severe pain, trouble breathing, confusion), seek urgent care or call 911."

# ---------------- Endpoints ----------------

@app.get("/health")
def health():
    return {"status": "ok", "backend": "rule-based", "updated_at": LATEST_CONTEXT.get("updated_at")}

@app.post("/agent/patient-intake")
def patient_intake(payload: IntakePayload):
    LATEST_CONTEXT["user"] = payload.user or LATEST_CONTEXT.get("user")
    LATEST_CONTEXT["patient"] = payload.patient or LATEST_CONTEXT.get("patient")
    LATEST_CONTEXT["updated_at"] = datetime.utcnow().isoformat()
    log.info("[Intake] user=%s, patient_keys=%s",
             (payload.user or {}).get("uuid"),
             list((payload.patient or {}).keys()))
    return {"ok": True}

@app.post("/agent/chat")
def agent_chat(req: ChatPayload):
    msg = (req.message or "").strip()
    if not msg:
        return {"reply": "Please describe what you're experiencing."}

    # Always produce a rule-based reply (no 'backend not available' anymore)
    reply = rule_based_triage(msg, LATEST_CONTEXT)
    return {"reply": reply, "backend": "rule-based"}

# prediction.py
# FastAPI + in-process LocalHealthRAG with rich, actionable responses.
# Frontend sends JSON: { session_id, envelope: { user, chat[] } }
# Not medical advice. Prototype only.

import os, json, re
from typing import Dict, List, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator  # Pydantic v2

# Keep a tiny chat history structure (no full LangChain chain needed)
from langchain_community.chat_message_histories import ChatMessageHistory

# ─────────────────────────────────────────
# In-process RAG pipeline (your code)
# ─────────────────────────────────────────
from src.local_rag_system import LocalHealthRAG

rag_system = LocalHealthRAG()
status = rag_system.get_system_status()
if not status.get("system_ready", False):
    raise RuntimeError("LocalHealthRAG system is not ready. Check your setup.")

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
CHAT_WINDOW_TURNS = int(os.getenv("CHAT_WINDOW_TURNS", "12"))
YELLOW_THRESHOLD  = float(os.getenv("YELLOW_THRESHOLD", "0.6"))  # promote to yellow if confidence > threshold
INCLUDE_SOURCES   = os.getenv("INCLUDE_SOURCES", "1") == "1"     # toggle source list in replies
DEBUG_RESPONSES   = os.getenv("DEBUG_RESPONSES", "0") == "1"     # log raw pipeline outputs

# ─────────────────────────────────────────
# Safety: red-flag detector
# ─────────────────────────────────────────
RED_PATTERNS = [
    r"not\s+breathing", r"blue\s+lips",
    r"(?:one|left|right)[-\s]?sided\s+weakness",
    r"face\s+droop|drooping\s+face|half\s+droop",
    r"speech\s+difficulty|slurred\s+speech",
    r"severe\s+chest\s+pain",
    r"radiating\s+(?:left\s+)?arm",
    r"faint(?:ing)?|blacking\s+out|passed\s+out",
    r"anaphylaxis",
    r"unconscious|cannot\s+wake",
    r"seizure\s*(?:lasting|>\s*5)|grand\s+mal",
]
def has_red_flags(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in RED_PATTERNS)

# ─────────────────────────────────────────
# Triage + formatting helpers
# ─────────────────────────────────────────
def _map_localrag_to_schema(result: Dict, user_text: str) -> Dict:
    """
    Map LocalHealthRAG's structured output into a strict prediction schema we can also
    reuse in the UI.
    """
    call_911   = bool(result.get("call_911"))
    confidence = float(result.get("confidence", 0.0))
    emer_raw   = (result.get("emergency_type") or "Uncertain").replace("_", " ").title()

    if call_911 or has_red_flags(user_text):
        triage = "red"
    else:
        triage = "yellow" if confidence > YELLOW_THRESHOLD else "green"

    # citations from vector_results (optional)
    vr = result.get("vector_results", []) or []
    citations: List[str] = []
    for h in vr[:5]:
        src = h.get("title") or h.get("source_id") or "localrag"
        if src not in citations:
            citations.append(src)

    return {
        "triage_level": triage,  # red | yellow | green
        "top_conditions": [{
            "code": emer_raw.upper().replace(" ", "_"),
            "name": emer_raw,
            "confidence": confidence
        }],
        "rationale": (result.get("natural_response") or "").strip()[:500] or "Limited information.",
        "citations": citations,
        "missing_critical_info": result.get("missing_info", ["onset_time","duration","severity_scale","associated_symptoms"]),
        "disclaimer": "Prototype only. Not medical advice."
    }

def _format_rich_message(prediction: Dict, raw: Dict) -> str:
    """
    Build a helpful, multi-section message:
      • Triage + Likely condition + Confidence
      • Plain-English rationale
      • Immediate actions (top 3)
      • Warning signs (top 5)
      • When to seek care
      • Sources (if enabled)
      • Closing disclaimer
    """
    triage = prediction["triage_level"].upper()
    top    = (prediction["top_conditions"] or [{}])[0]
    cond   = top.get("name", "Uncertain")
    conf   = float(top.get("confidence", 0.0))

    parts: List[str] = []
    # Header
    parts.append(f"Triage: {triage}  •  Most likely: {cond}  •  Confidence: {conf:.0%}")

    # Rationale
    rationale = prediction.get("rationale") or ""
    if rationale:
        parts.append(rationale.strip())

    # Immediate actions
    ia = raw.get("immediate_actions") or []
    if ia:
        parts.append("Immediate actions:")
        parts += [f"• {a}" for a in ia[:3]]

    # Warning signs
    ws = raw.get("warning_signs") or []
    if ws:
        parts.append("Warning signs to watch for:")
        parts += [f"• {w}" for w in ws[:5]]

    # Care guidance
    if prediction["triage_level"] == "red" or raw.get("call_911"):
        parts.append("When to seek care: If symptoms are severe or worsening, call 911 or go to the nearest emergency department now.")
    elif prediction["triage_level"] == "yellow":
        parts.append("When to seek care: Arrange urgent evaluation (same-day clinic or urgent care), especially if symptoms persist or escalate.")
    else:
        parts.append("When to seek care: Monitor and follow up with a primary care clinician if symptoms persist, recur, or worsen.")

    # Sources (optional)
    if INCLUDE_SOURCES:
        cites = prediction.get("citations") or []
        if cites:
            parts.append("Sources:")
            parts += [f"• {c}" for c in cites[:5]]

    # Close with disclaimer
    parts.append("Not medical advice.")
    return "\n".join(parts).strip()

# ─────────────────────────────────────────
# Pydantic request models
# ─────────────────────────────────────────
Role = Literal["user", "assistant", "system"]

class ChatTurn(BaseModel):
    role: Role
    content: str = Field(..., min_length=1)

class FrontendUser(BaseModel):
    name: str = Field(..., min_length=1)
    age: int = Field(..., ge=0, le=120)
    weight: float = Field(..., ge=0)
    height: float = Field(..., ge=0)
    conditions: List[str]

class ChatEnvelope(BaseModel):
    user: FrontendUser
    chat: List[ChatTurn]

    @field_validator("chat")
    @classmethod
    def chat_must_have_user_message(cls, v: List[ChatTurn]):
        if not v:
            raise ValueError("chat must contain at least one message")
        if not any(t.role == "user" for t in v):
            raise ValueError("chat must include at least one 'user' message")
        return v

class ChatReq(BaseModel):
    session_id: str = "default"
    envelope: ChatEnvelope

# ─────────────────────────────────────────
# Minimal in-memory chat histories
# ─────────────────────────────────────────
_histories: Dict[str, ChatMessageHistory] = {}
def get_history(session_id: str) -> ChatMessageHistory:
    return _histories.setdefault(session_id, ChatMessageHistory())

def _cap_window(hist: ChatMessageHistory, max_turns: int = CHAT_WINDOW_TURNS):
    msgs = list(hist.messages)
    if len(msgs) > max_turns:
        hist.messages = type(hist.messages)(msgs[-max_turns:])

# ─────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────
app = FastAPI()

@app.post("/chat")
async def chat(req: ChatReq):
    session_id = req.session_id or "default"
    # derive a simple user_id (slug) from name
    user_id = re.sub(r"[^a-z0-9_-]+", "-", req.envelope.user.name.strip().lower())

    # build profile dict (kept for parity/analytics; not used by LocalHealthRAG directly)
    profile_json = {
        "name": req.envelope.user.name,
        "age": req.envelope.user.age,
        "weight": req.envelope.user.weight,
        "height": req.envelope.user.height,
        "conditions": req.envelope.user.conditions,
    }

    # latest user message
    last_user_msg = next((t.content for t in reversed(req.envelope.chat) if t.role == "user"), None)
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No user message found in 'chat'.")

    # Keep history manually and cap
    hist = get_history(session_id)
    hist.add_user_message(last_user_msg)
    _cap_window(hist, CHAT_WINDOW_TURNS)

    # ==== Single in-process inference (no network) ====
    result = rag_system.query_health_emergency(last_user_msg)

    if DEBUG_RESPONSES:
        import sys
        print("[LocalHealthRAG result]", json.dumps(result, indent=2), file=sys.stderr)

    # Map to strict predictor schema + build an informative message
    prediction = _map_localrag_to_schema(result, last_user_msg)
    message    = _format_rich_message(prediction, result)

    # Ensure emergency post-script is present if truly red
    if prediction["triage_level"] == "red" and "emergency" not in message.lower():
        message += ("\n\n⚠️ If you’re experiencing severe or worsening symptoms "
                    "(e.g., severe chest pain, trouble breathing, stroke-like symptoms), "
                    "please seek emergency care now. Not medical advice.")

    # Append assistant message and cap again
    hist.add_ai_message(message)
    _cap_window(hist, CHAT_WINDOW_TURNS)

    # Return both text and structured data so the UI can render it nicely
    return {
        "output": message,
        "data": {
            "prediction": prediction,
            "input_bundle": {"profile": profile_json, "current_input": last_user_msg},
            "model_info": {"provider": "localrag", "model_name": "LocalHealthRAG", "prompt_version": "v2-rich"},
            # Raw fields that UIs can present directly:
            "emergency_type": result.get("emergency_type"),
            "call_911": bool(result.get("call_911")),
            "confidence": float(result.get("confidence", 0.0)),
            "immediate_actions": result.get("immediate_actions", []),
            "warning_signs": result.get("warning_signs", []),
            "vector_results": (result.get("vector_results") or [])[:5],
            "user_id": user_id,
            "session_id": session_id,
        }
    }

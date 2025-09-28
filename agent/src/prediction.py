# server.py
# FastAPI + LangChain chat with in-memory history + ADK-style predictor tool grounded via RAG-Anything.
# No database. Frontend sends strict JSON: { session_id, envelope: { user, chat[] } }
# Not medical advice. Prototype only.

import os, json, re, asyncio
from typing import Dict, List, Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator  # Pydantic v2
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.memory import ChatMessageHistory
from langchain.tools import StructuredTool

# ─────────────────────────────────────────
# Config
# ─────────────────────────────────────────
CHAT_WINDOW_TURNS = int(os.getenv("CHAT_WINDOW_TURNS", "12"))

# ─────────────────────────────────────────
# Safety: simple red-flag detector
# ─────────────────────────────────────────
RED_PATTERNS = [
    r"not\s+breathing", r"blue\s+lips", r"one[-\s]?sided\s+weakness",
    r"face\s+droop", r"speech\s+difficulty", r"severe\s+chest\s+pain",
    r"radiating\s+(left\s+)?arm", r"faint(ing)?", r"anaphylaxis",
    r"unconscious", r"cannot\s+wake", r"seizure\s*(lasting|>\s*5)"
]

def has_red_flags(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in RED_PATTERNS)

# ─────────────────────────────────────────
# Local LLM client (plug your model here)
# Expect .generate(system, user, temperature, max_tokens) -> str
# ─────────────────────────────────────────
class LocalLLMClient:
    def __init__(self, model):
        self.model = model

    async def complete(self, system: str, user: str, temperature: float = 0.2, max_tokens: int = 600) -> str:
        def _run():
            return self.model.generate(system=system, user=user, temperature=temperature, max_tokens=max_tokens)
        return await asyncio.to_thread(_run)

# Demo stub model — replace with your real local model
class MyChatModel:
    def generate(self, system, user, temperature, max_tokens):
        return "Thanks for sharing. When did this start, how severe is it (0–10), and what makes it better or worse?\n\nNot medical advice."

llm_client = LocalLLMClient(MyChatModel())

def make_local_llm_runnable(client: LocalLLMClient):
    async def _call(inputs: dict):
        system = inputs["system"]
        user = inputs["user"]
        text = await client.complete(system=system, user=user, temperature=0.2, max_tokens=600)
        return {"output": text}
    return RunnableLambda(_call)

lc_llm = make_local_llm_runnable(llm_client)

# ─────────────────────────────────────────
# Predictor JSON contract (strict)
# ─────────────────────────────────────────
REQUIRED_KEYS = {"triage_level","top_conditions","rationale","citations","missing_critical_info","disclaimer"}
VALID_TRIAGE = {"red","yellow","green"}

def validate_prediction_json(raw: str) -> Dict:
    obj = json.loads(raw)
    if not REQUIRED_KEYS.issubset(obj.keys()):
        raise ValueError(f"Missing required keys. Got: {list(obj.keys())}")
    if obj["triage_level"] not in VALID_TRIAGE:
        raise ValueError("Invalid triage_level")
    tcs = obj["top_conditions"]
    if not isinstance(tcs, list) or not tcs:
        raise ValueError("top_conditions must be a non-empty list")
    for c in tcs:
        if not {"code","name","confidence"}.issubset(c.keys()):
            raise ValueError("Each condition must include code, name, confidence")
        cf = float(c["confidence"])
        if not (0.0 <= cf <= 1.0):
            raise ValueError("confidence must be in [0,1]")
    return obj

# ─────────────────────────────────────────
# Pydantic models (strict, no Optional)
# Frontend payload:
# {
#   "session_id": "s1",
#   "envelope": {
#     "user": { "name": "Malik", "age": 21, "weight": 185, "height": 180, "conditions": ["Asthma"] },
#     "chat": [{ "role": "user", "content": "..." }]
#   }
# }
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
    conditions: List[str]  # can be empty list, but must be present

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
# RAG-Anything client (import + instance)
# ─────────────────────────────────────────
from rag_client import RAGAnythingClient
rag = RAGAnythingClient(base_url=os.getenv("RAG_ANYTHING_URL", "http://localhost:9999"))

def make_evidence_block(chunks: List[Dict], max_chars: int = 3500):
    lines, citations, used = [], [], 0
    for ch in chunks:
        tag = ch.get("source_id") or "unknown"
        block = f"[{tag}]\n{ch['content'].strip()}\n"
        if used + len(block) > max_chars:
            break
        lines.append(block); citations.append(tag); used += len(block)
    return "\n".join(lines), citations

# ─────────────────────────────────────────
# Predictor prompting (ADK tool) with RAG grounding
# ─────────────────────────────────────────
PREDICT_SYSTEM = (
    "You are a cautious clinical reasoning assistant for prototyping only. "
    "You DO NOT give medical advice or diagnoses. "
    "Ground your reasoning ONLY in the patient data and the EVIDENCE block. "
    "If evidence is insufficient, be conservative.\n\n"
    "Output STRICT JSON with keys: triage_level, top_conditions, rationale, citations, "
    "missing_critical_info, disclaimer. triage_level ∈ {red,yellow,green}. "
    "top_conditions: array of {code, name, confidence}. Rationale ≤2 sentences. "
    "List citation IDs exactly as shown in EVIDENCE."
)

def build_predict_user(profile: Dict, current_input: str, evidence_block: str) -> str:
    return (
        f"PROFILE_JSON:\n{json.dumps(profile, ensure_ascii=False)}\n\n"
        f"CURRENT_INPUT:\n{current_input}\n\n"
        f"EVIDENCE (each block prefixed with [id]):\n{evidence_block or '(no retrieved evidence)'}\n\n"
        "Return ONLY the JSON object described above."
    )

class PredictArgs(BaseModel):
    user_id: str
    current_input: str
    profile: Dict  # strict: must be present

async def predictor_tool_fn(user_id: str, current_input: str, profile: Dict) -> str:
    # Red-flag prefilter
    red_prefilter = has_red_flags(current_input)

    # Retrieve grounding evidence from RAG-Anything
    try:
        raw_chunks = rag.retrieve(current_input, k=5)
    except Exception:
        raw_chunks = []
    evidence_block, rag_citations = make_evidence_block(raw_chunks)

    # Build prompt with evidence
    user_prompt = build_predict_user(profile, current_input, evidence_block)

    # Call local LLM; expect strict JSON string
    raw = await llm_client.complete(system=PREDICT_SYSTEM, user=user_prompt, temperature=0.1, max_tokens=700)

    # Validate or fallback
    try:
        obj = validate_prediction_json(raw)
    except Exception as e:
        obj = {
            "triage_level": "red" if red_prefilter else "yellow",
            "top_conditions": [{"code":"UNSURE","name":"Uncertain","confidence":0.0}],
            "rationale": f"Model output invalid ({str(e)}). Fallback engaged.",
            "citations": rag_citations,  # still surface evidence IDs on fallback
            "missing_critical_info": ["onset_time","duration","severity_scale","associated_symptoms"],
            "disclaimer": "Prototype only. Not medical advice."
        }

    if red_prefilter and obj.get("triage_level") != "red":
        obj["triage_level"] = "red"

    # Ensure citations include RAG IDs
    obj.setdefault("citations", [])
    existing = set(obj["citations"])
    for cid in rag_citations:
        if cid not in existing:
            obj["citations"].append(cid); existing.add(cid)

    return json.dumps({
        "prediction": obj,
        "input_bundle": {"profile": profile, "current_input": current_input},
        "model_info": {"provider":"local+rag", "model_name":"my-small-llm", "prompt_version":"v1-rag"}
    })

predict_tool = StructuredTool.from_function(
    coroutine=predictor_tool_fn,
    name="condition_predictor",
    description="Estimate likely conditions and triage from profile + current input (RAG-grounded). Returns strict JSON.",
    args_schema=PredictArgs,
)

# ─────────────────────────────────────────
# Chat chain with in-memory history (short window)
# ─────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a careful, supportive health chat assistant for prototyping only.\n"
    "CRITICAL RULES:\n"
    "• You are NOT a doctor and do NOT provide medical advice or diagnoses.\n"
    "• Be empathetic and concise. Ask focused follow-ups when needed.\n"
    "• If RED-FLAG symptoms appear (severe chest pain, stroke signs, trouble breathing), advise immediate emergency care.\n"
    "• End your response with: 'Not medical advice.'"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder("history"),
    ("human", "{input}")
])

_histories: Dict[str, ChatMessageHistory] = {}

def get_history(session_id: str) -> ChatMessageHistory:
    return _histories.setdefault(session_id, ChatMessageHistory())

def _cap_window(hist: ChatMessageHistory, max_turns: int = CHAT_WINDOW_TURNS):
    msgs = list(hist.messages)
    if len(msgs) > max_turns:
        hist.messages = type(hist.messages)(msgs[-max_turns:])

async def _chat_invoke_with_cap(inputs, config):
    out = await (prompt | lc_llm).ainvoke(inputs, config=config)
    session_id = (config.get("configurable") or {}).get("session_id", "default")
    _cap_window(get_history(session_id))
    return out

chat_with_memory = RunnableWithMessageHistory(
    RunnableLambda(_chat_invoke_with_cap),
    get_session_history=get_history,
    input_messages_key="input",
    history_messages_key="history"
)

# ─────────────────────────────────────────
# Router: chat vs predictor tool
# ─────────────────────────────────────────
DIAG_RE = re.compile(r"\b(diagnose|what\s+is\s+this|condition|triage|predict|assessment)\b", re.I)

async def router_fn(inputs: dict):
    text = inputs["input"]
    user_id = inputs["user_id"]
    session_id = inputs.get("session_id", "default")
    profile_json = inputs["profile"]  # required

    # Emergency check
    if has_red_flags(text):
        res = await chat_with_memory.ainvoke({"input": text}, config={"configurable":{"session_id": session_id}})
        out = res["output"]
        if "emergency" not in out.lower():
            out += ("\n\n⚠️ If you’re experiencing severe or worsening symptoms (e.g., severe chest pain, "
                    "trouble breathing, stroke-like symptoms), please seek emergency care now. Not medical advice.")
        return {"output": out}

    # Ask for assessment → call ADK tool
    if DIAG_RE.search(text):
        payload = await predict_tool.ainvoke({
            "user_id": user_id,
            "current_input": text,
            "profile": profile_json
        })
        data = json.loads(payload)
        top = (data["prediction"]["top_conditions"] or [{}])[0]
        triage = data["prediction"]["triage_level"]
        conf = float(top.get("confidence", 0.0))
        summary = (f"Triage: {triage.upper()}\n"
                   f"Most likely: {top.get('name','?')} ({top.get('code','?')}) p≈{conf:.2f}\n"
                   f"Not medical advice.")
        await chat_with_memory.ainvoke({"input": f"[System note] Tool result shared: {summary}"},
                                       config={"configurable":{"session_id": session_id}})
        return {"output": summary, "data": data}

    # Normal chat
    return await chat_with_memory.ainvoke({"input": text}, config={"configurable":{"session_id": session_id}})

router = RunnableLambda(router_fn)

# ─────────────────────────────────────────
# FastAPI
# ─────────────────────────────────────────
app = FastAPI()

@app.post("/chat")
async def chat(req: ChatReq):
    session_id = req.session_id or "default"
    # derive a simple user_id (slug) from name; replace with real ID if you have one
    user_id = re.sub(r"[^a-z0-9_-]+", "-", req.envelope.user.name.strip().lower())

    # build profile dict (strict, no units parsing)
    profile_json = {
        "name": req.envelope.user.name,
        "age": req.envelope.user.age,
        "weight": req.envelope.user.weight,
        "height": req.envelope.user.height,
        "conditions": req.envelope.user.conditions,
    }

    # use the latest user message
    last_user_msg = next((t.content for t in reversed(req.envelope.chat) if t.role == "user"), None)
    if not last_user_msg:
        raise HTTPException(status_code=400, detail="No user message found in 'chat'.")

    out = await router.ainvoke({
        "input": last_user_msg,
        "user_id": user_id,
        "session_id": session_id,
        "profile": profile_json
    })
    return out
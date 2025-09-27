# condition_predictor_adk.py
import os, json, sqlite3, asyncio, re
from typing import Any, Dict, List, Optional

# ───────────────────────────────
#  Config
# ───────────────────────────────
DEFAULT_DB_PATH = os.getenv("MEDICAL_DB_PATH", "medical.db")
MAX_EVIDENCE_SYMPTOMS = int(os.getenv("MAX_SYMPTOMS", "5"))  # recent symptoms to include

# ───────────────────────────────
#  Safety: basic red-flag scan
# ───────────────────────────────
RED_PATTERNS = [
    r"not\s+breathing", r"blue\s+lips", r"one[-\s]?sided\s+weakness",
    r"face\s+droop", r"speech\s+difficulty", r"severe\s+chest\s+pain",
    r"radiating\s+(left\s+)?arm", r"faint(ing)?", r"anaphylaxis",
    r"unconscious", r"cannot\s+wake", r"seizure\s*(lasting|>\s*5)"
]
def has_red_flags(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in RED_PATTERNS)

# ───────────────────────────────
#  JSON schema validation
# ───────────────────────────────
REQUIRED_KEYS = {"triage_level","top_conditions","rationale","citations","missing_critical_info","disclaimer"}
VALID_TRIAGE = {"red","yellow","green"}

def validate_prediction_json(raw: str) -> Dict[str, Any]:
    obj = json.loads(raw)
    if not REQUIRED_KEYS.issubset(obj.keys()):
        raise ValueError(f"Missing required keys. Got: {list(obj.keys())}")
    if obj["triage_level"] not in VALID_TRIAGE:
        raise ValueError("Invalid triage_level")
    if not isinstance(obj["top_conditions"], list) or not obj["top_conditions"]:
        raise ValueError("top_conditions must be a non-empty list")
    for c in obj["top_conditions"]:
        if not {"code","name","confidence"}.issubset(c.keys()):
            raise ValueError("Each condition must include code, name, confidence")
        cf = float(c["confidence"])
        if not (0.0 <= cf <= 1.0):
            raise ValueError("confidence must be in [0,1]")
    return obj

# ───────────────────────────────
#  Local DB helpers
# ───────────────────────────────
class LocalDB:
    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row

    def get_profile(self, user_id: str) -> Dict[str, Any]:
        row = self.conn.execute(
            """SELECT user_id, age, sex, weight_kg, height_cm, allergies, chronic_conditions, meds
               FROM users WHERE user_id = ?""",
            (user_id,)
        ).fetchone()
        return dict(row) if row else {}

    def get_recent_symptoms(self, user_id: str, limit: int = MAX_EVIDENCE_SYMPTOMS) -> List[Dict[str, Any]]:
        rows = self.conn.execute(
            """SELECT ts, text, coded
               FROM symptoms
               WHERE user_id = ?
               ORDER BY ts DESC
               LIMIT ?""",
            (user_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]

# ───────────────────────────────
#  Your LLM client (choose one)
# ───────────────────────────────
class AbstractLLMClient:
    """Implement .complete(system: str, user: str, temperature: float, max_tokens: int) -> str"""
    async def complete(self, system: str, user: str, temperature: float = 0.1, max_tokens: int = 800) -> str:
        raise NotImplementedError

class LocalHTTPClient(AbstractLLMClient):
    """
    Example: call your own inference server (FastAPI/Flask/etc.)
    Expected JSON response: {"text": "<model_output_string>"}
    """
    def __init__(self, base_url: str = "http://localhost:8000/complete"):
        self.base_url = base_url

    async def complete(self, system: str, user: str, temperature: float = 0.1, max_tokens: int = 800) -> str:
        import requests  # sync; wrap in thread for async
        def _run():
            payload = {
                "system": system,
                "user": user,
                "temperature": temperature,
                "max_tokens": max_tokens
            }
            r = requests.post(self.base_url, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data.get("text", "")
        return await asyncio.to_thread(_run)

class InProcessClient(AbstractLLMClient):
    """
    Example: call your model directly (e.g., llama.cpp, vLLM, or a local generate() function).
    Replace the pass with your own generation call.
    """
    def __init__(self, model):
        self.model = model  # your object with .generate(system, user, temperature, max_tokens)

    async def complete(self, system: str, user: str, temperature: float = 0.1, max_tokens: int = 800) -> str:
        def _run():
            # Must return a string that is STRICT JSON per the schema
            return self.model.generate(system=system, user=user, temperature=temperature, max_tokens=max_tokens)
        return await asyncio.to_thread(_run)

# ───────────────────────────────
#  The ADK Tool
# ───────────────────────────────
class ConditionPredictorADK:
    """
    ADK tool:
      - Reads user profile + recent symptoms from local DB
      - Prompts YOUR LLM to estimate likely conditions (strict JSON)
      - Writes to state via output_key
    """
    name = "condition_predictor"
    description = "Predicts likely medical conditions from local medical records + current symptoms via local LLM."
    output_key = "last_condition_estimate"

    def __init__(self, llm_client: AbstractLLMClient, db_path: str = DEFAULT_DB_PATH, model_name: str = "local-llm"):
        self.llm = llm_client
        self.db = LocalDB(db_path=db_path)
        self.model_name = model_name

    def _build_prompt(self, profile: Dict[str, Any], recent_symptoms: List[Dict[str, Any]], live_query: Optional[str]) -> Dict[str, str]:
        system = (
            "You are a cautious clinical reasoning assistant for prototyping only. "
            "You DO NOT give medical advice or diagnoses. "
            "Estimate LIKELY conditions based on the patient's profile and symptoms. "
            "Output STRICT JSON with keys: "
            "triage_level, top_conditions, rationale, citations, missing_critical_info, disclaimer. "
            "triage_level ∈ {red,yellow,green}. "
            "top_conditions is an array of objects with fields {code, name, confidence}. "
            "If red-flag symptoms are present, triage_level MUST be 'red'. "
            "Keep rationale ≤2 sentences. Citations can be simple placeholders like ['local_record']."
        )

        sym_lines = []
        for r in recent_symptoms:
            line = f"- {r.get('ts','')} : {r.get('text','')}"
            if r.get('coded'):
                line += f" (coded: {r['coded']})"
            sym_lines.append(line)
        sym_block = "\n".join(sym_lines) if sym_lines else "(none on record)"

        user_block = (
            f"PROFILE:\n{json.dumps(profile, ensure_ascii=False)}\n\n"
            f"RECENT_SYMPTOMS:\n{sym_block}\n\n"
            f"CURRENT_INPUT:\n{live_query or ''}\n\n"
            "Return ONLY a JSON object with the exact keys described."
        )
        return {"system": system, "user": user_block}

    def _render_text(self, obj: Dict[str, Any]) -> str:
        lines = [f"Triage: **{obj['triage_level'].upper()}**", "Most likely:"]
        for i, c in enumerate(obj["top_conditions"][:5], 1):
            try:
                conf = float(c["confidence"])
            except Exception:
                conf = 0.0
            lines.append(f"{i}. {c['name']} ({c['code']}) — p≈{conf:.2f}")
        if obj.get("citations"):
            lines.append(f"Citations: {', '.join(obj['citations'])}")
        lines.append(obj.get("disclaimer","Prototype only. Not medical advice."))
        return "\n".join(lines)

    async def __call__(self, *, query: str, user_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        # 1) Pull profile + recent symptoms from local DB
        profile = self.db.get_profile(user_id) or {}
        recent_symptoms = self.db.get_recent_symptoms(user_id, limit=MAX_EVIDENCE_SYMPTOMS)

        # 2) Pre-scan for red flags across stored + live text
        combined_text = " ".join([query] + [s.get("text","") for s in recent_symptoms])
        red_prefilter = has_red_flags(combined_text)

        # 3) Prompt your LLM
        prompt = self._build_prompt(profile, recent_symptoms, query)
        raw = await self.llm.complete(system=prompt["system"], user=prompt["user"], temperature=0.1, max_tokens=700)

        # 4) Validate JSON (fallback safe if invalid)
        try:
            obj = validate_prediction_json(raw)
        except Exception as e:
            obj = {
                "triage_level": "red" if red_prefilter else "yellow",
                "top_conditions": [{"code":"UNSURE","name":"Uncertain","confidence":0.0}],
                "rationale": f"Model output invalid ({str(e)}). Fallback engaged.",
                "citations": ["local_record"] if recent_symptoms else [],
                "missing_critical_info": ["onset_time","duration","severity_scale","associated_symptoms"],
                "disclaimer": "Prototype only. Not medical advice."
            }

        # 5) Never down-triage if we pre-detected red flags
        if red_prefilter and obj.get("triage_level") != "red":
            obj["triage_level"] = "red"

        # 6) Compose response payload
        text = self._render_text(obj)
        result_payload = {
            "prediction": obj,
            "input_bundle": {  # JSON extracted from your DB (easy for frontend + next ADK)
                "profile": profile,
                "recent_symptoms": recent_symptoms,
                "current_input": query
            },
            "model_info": {
                "provider": "local",
                "model_name": self.model_name,
                "prompt_version": "v1"
            }
        }

        # 7) Persist to session for chaining
        writes = {
            self.output_key: result_payload,
            "triage_level": obj["triage_level"],
            "last_condition_audit": {
                "used_profile_keys": list(profile.keys()),
                "symptom_count_used": len(recent_symptoms),
                "prefilter_red": red_prefilter
            }
        }

        return {
            "text": text,          # optional human-friendly rendering
            "data": result_payload, # machine-friendly JSON for frontend
            "write_state": writes   # makes it easy for the next ADK to reuse
        }

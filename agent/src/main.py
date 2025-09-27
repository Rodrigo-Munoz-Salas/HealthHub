# agent/src/main.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import logging

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
app = FastAPI(title="HealthHub Agent")

# Allow your UI (Vite dev) to call the agent
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # or ["*"] during dev
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: str
    message: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/agent/chat")
def chat(req: ChatRequest):
    logging.info("User input received! AI agent is working. Payload: %s", req.model_dump())
    return {"reply": f"Echo: {req.message}", "user_id": req.user_id}

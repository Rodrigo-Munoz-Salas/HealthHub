# agent/src/rag_backend.py
import logging

logger = logging.getLogger(__name__)

RAG = None
RAG_STATUS = {"system_ready": False, "error": "RAG not initialized"}

def init_rag() -> bool:
    """
    Import and initialize the local RAG system once at startup.
    Returns True if ready, False otherwise.
    """
    global RAG, RAG_STATUS
    try:
        # ⬇⬇⬇ relative import because we run uvicorn with `src.main:app`
        from .windows_rag_system import WindowsRAGSystem
    except Exception as e:
        logger.error("RAG import failed: %s", e)
        RAG_STATUS = {"system_ready": False, "error": f"import failed: {e}"}
        return False

    try:
        RAG = WindowsRAGSystem()  # uses your friend's defaults for paths/models
        RAG_STATUS = RAG.get_system_status()
        logger.info("RAG initialized: %s", RAG_STATUS)
        return True
    except Exception:
        logger.exception("RAG initialization error")
        RAG = None
        RAG_STATUS = {"system_ready": False, "error": "init failed (see logs)"}
        return False


def rag_answer(text: str) -> dict:
    """
    Run a query through the RAG system.
    Returns the full dict produced by query_health_emergency, or a minimal error payload.
    """
    if RAG is None:
        return {"error": "rag_not_ready"}
    try:
        return RAG.query_health_emergency(text)
    except Exception as e:
        logger.exception("RAG query error")
        return {"error": f"rag_query_failed: {e}"}

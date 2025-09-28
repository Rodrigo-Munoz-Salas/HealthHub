# rag_client.py
# In-process retriever shim backed by LocalHealthRAG (no HTTP).

from src.local_rag_system import LocalHealthRAG

class RAGAnythingClient:
    def __init__(self, *args, **kwargs):
        self._rag = LocalHealthRAG()
        status = self._rag.get_system_status()
        if not status.get("system_ready", False):
            raise RuntimeError("LocalHealthRAG system is not ready. Check your setup.")

    def retrieve(self, query: str, k: int = 5):
        """
        Use LocalHealthRAG to fetch vector results. Normalize to:
          [{ "content": "...", "source_id": "<title or localrag>" }, ...]
        """
        res = self._rag.query_health_emergency(query)
        hits = (res.get("vector_results") or [])[:k]
        out = []
        for h in hits:
            out.append({
                "content": h.get("content", ""),
                "source_id": h.get("title") or h.get("source_id") or "localrag"
            })
        return out

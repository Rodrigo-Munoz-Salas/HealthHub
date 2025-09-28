# rag_client.py
# Tiny HTTP client for your RAG-Anything retriever.

import requests

class RAGAnythingClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def retrieve(self, query: str, k: int = 5):
        """
        Call your RAG-Anything server. Adjust the path/body to match your deployment.
        Expected return: list of { "content": "...", "source_id": "who-asthma-12ab3c" }
        """
        resp = requests.post(f"{self.base_url}/retrieve", json={"query": query, "k": k}, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        # Normalize defensively
        out = []
        for item in data:
            out.append({
                "content": item.get("content", ""),
                "source_id": item.get("source_id", "unknown")
            })
        return out

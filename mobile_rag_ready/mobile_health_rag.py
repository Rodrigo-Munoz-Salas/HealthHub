"""
Mobile Health Emergency RAG System
Pre-built for instant deployment on mobile devices
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import time

class MobileHealthRAG:
    """Pre-built mobile RAG system for health emergencies"""
    
    def __init__(self, models_dir: str = "mobile_models", data_dir: str = "mobile_rag_ready"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        # Load pre-built data
        self.guidelines = self._load_guidelines()
        self.vector_db = self._load_vector_database()
        self.emergency_protocols = self._load_emergency_protocols()
        
        print("🏥 Mobile Health Emergency RAG System Loaded!")
        print(f"📚 Guidelines: {len(self.guidelines)}")
        print(f"🔍 Vector Database: {len(self.vector_db)} guidelines")
        print(f"🚨 Emergency Protocols: {len(self.emergency_protocols)}")
    
    def _load_guidelines(self) -> Dict:
        """Load pre-processed guidelines"""
        guidelines_path = self.data_dir / "processed_guidelines.json"
        if guidelines_path.exists():
            with open(guidelines_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _load_vector_database(self) -> Dict:
        """Load pre-built vector database"""
        vector_db_path = self.data_dir / "vector_database.pkl"
        if vector_db_path.exists():
            with open(vector_db_path, 'rb') as f:
                return pickle.load(f)
        return {}
    
    def _load_emergency_protocols(self) -> Dict:
        """Load emergency protocols"""
        protocols_path = self.data_dir / "emergency_protocols.json"
        if protocols_path.exists():
            with open(protocols_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def query_emergency(self, query: str) -> Dict[str, Any]:
        """Query the mobile RAG system for health emergencies"""
        start_time = time.time()
        
        # Pre-process query
        query_lower = query.lower()
        
        # Check for emergency keywords
        emergency_keywords = {
            "chest_pain": ["chest pain", "heart attack", "cardiac", "heart"],
            "fainting": ["fainted", "fainting", "unconscious", "passed out"],
            "burn": ["burn", "burned", "fire", "hot"],
            "choking": ["choking", "can't breathe", "blocked airway"],
            "stroke": ["stroke", "facial droop", "slurred speech", "weakness"]
        }
        
        # Find matching emergency type
        emergency_type = None
        for etype, keywords in emergency_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                emergency_type = etype
                break
        
        # Get relevant information
        if emergency_type and emergency_type in self.emergency_protocols:
            protocol = self.emergency_protocols[emergency_type]
            response = {
                "emergency_type": emergency_type,
                "protocol": protocol,
                "immediate_actions": protocol["immediate_actions"][:3],  # First 3 steps
                "warning_signs": protocol["warning_signs"],
                "call_911": protocol.get("call_911", True),
                "confidence": 0.9,
                "source": "emergency_protocols"
            }
        else:
            # Use vector search for general health queries
            response = self._vector_search(query)
        
        response["query_time"] = time.time() - start_time
        response["timestamp"] = time.time()
        
        return response
    
    def _vector_search(self, query: str) -> Dict[str, Any]:
        """Perform vector search on pre-built database"""
        # Simple keyword matching for now
        query_lower = query.lower()
        
        # Find best matching guideline
        best_match = None
        best_score = 0
        
        for guideline_id, guideline in self.guidelines.items():
            score = 0
            for keyword in guideline.get("keywords", []):
                if keyword.lower() in query_lower:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = guideline
        
        if best_match:
            return {
                "emergency_type": "general_health",
                "response": best_match["content"][:500] + "...",
                "confidence": min(best_score / 5, 1.0),
                "call_911": best_match.get("emergency_level") == "critical",
                "source": "guidelines"
            }
        else:
            return {
                "emergency_type": "general_health",
                "response": "Please provide more specific details about the health emergency.",
                "confidence": 0.3,
                "call_911": False,
                "source": "fallback"
            }
    
    def get_emergency_protocol(self, emergency_type: str) -> Dict[str, Any]:
        """Get specific emergency protocol"""
        return self.emergency_protocols.get(emergency_type, {})
    
    def get_all_protocols(self) -> Dict[str, Any]:
        """Get all available emergency protocols"""
        return self.emergency_protocols
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "guidelines_loaded": len(self.guidelines),
            "vector_db_loaded": len(self.vector_db),
            "emergency_protocols": len(self.emergency_protocols),
            "system_ready": True
        }

# Usage example
if __name__ == "__main__":
    # Initialize mobile RAG system
    mobile_rag = MobileHealthRAG()
    
    # Test emergency queries
    test_queries = [
        "Someone is having chest pain and shortness of breath",
        "A person fainted and is not breathing",
        "There's a severe burn on someone's hand"
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        response = mobile_rag.query_emergency(query)
        print(f"Emergency Type: {response['emergency_type']}")
        print(f"Immediate Actions: {response.get('immediate_actions', [])}")
        print(f"Call 911: {response.get('call_911', False)}")
        print(f"Response Time: {response.get('query_time', 0):.2f}s")

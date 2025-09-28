"""
Mobile RAG System for Google ADK Integration
Mobile-optimized RAG system for health emergencies
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import time
import logging

logger = logging.getLogger(__name__)

class MobileHealthRAG:
    """Mobile-optimized RAG system for health emergencies with ADK integration"""
    
    def __init__(self, models_dir: str = "mobile_models", data_dir: str = "mobile_rag_ready"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        
        # Load pre-built data
        self.guidelines = self._load_guidelines()
        self.vector_db = self._load_vector_database()
        self.emergency_protocols = self._load_emergency_protocols()
        
        # Initialize mobile models (if available)
        self.llm_model = None
        self.llm_tokenizer = None
        self.embedding_model = None
        
        logger.info("🏥 Mobile Health Emergency RAG System Loaded!")
        logger.info(f"📚 Guidelines: {len(self.guidelines)}")
        logger.info(f"🔍 Vector Database: {len(self.vector_db)} guidelines")
        logger.info(f"🚨 Emergency Protocols: {len(self.emergency_protocols)}")
    
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
        
        try:
            # Pre-process query
            query_lower = query.lower()
            
            # Check for emergency keywords
            emergency_keywords = {
                "chest_pain": ["chest pain", "heart attack", "cardiac", "heart", "chest"],
                "fainting": ["fainted", "fainting", "unconscious", "passed out", "collapsed"],
                "burn": ["burn", "burned", "fire", "hot", "scald", "thermal"],
                "choking": ["choking", "can't breathe", "blocked airway", "suffocating"],
                "stroke": ["stroke", "facial droop", "slurred speech", "weakness", "paralysis"]
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
            
        except Exception as e:
            logger.error(f"❌ Error in query_emergency: {e}")
            return {
                "emergency_type": "error",
                "response": "Unable to process health emergency query",
                "call_911": True,
                "confidence": 0.0,
                "query_time": time.time() - start_time,
                "error": str(e)
            }
    
    def _vector_search(self, query: str) -> Dict[str, Any]:
        """Perform vector search on pre-built database"""
        try:
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
                
        except Exception as e:
            logger.error(f"❌ Error in vector search: {e}")
            return {
                "emergency_type": "error",
                "response": "Unable to search health guidelines",
                "confidence": 0.0,
                "call_911": True,
                "error": str(e)
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
            "system_ready": True,
            "models_dir": str(self.models_dir),
            "data_dir": str(self.data_dir)
        }
    
    def search_guidelines(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Search guidelines for specific information"""
        try:
            query_lower = query.lower()
            results = []
            
            for guideline_id, guideline in self.guidelines.items():
                score = 0
                for keyword in guideline.get("keywords", []):
                    if keyword.lower() in query_lower:
                        score += 1
                
                if score > 0:
                    results.append({
                        "guideline_id": guideline_id,
                        "title": guideline["title"],
                        "score": score,
                        "emergency_level": guideline.get("emergency_level", "medium"),
                        "content_preview": guideline["content"][:200] + "..."
                    })
            
            # Sort by score and return top results
            results.sort(key=lambda x: x["score"], reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"❌ Error searching guidelines: {e}")
            return []
    
    def get_emergency_summary(self) -> Dict[str, Any]:
        """Get summary of available emergency protocols"""
        try:
            summary = {
                "total_protocols": len(self.emergency_protocols),
                "protocols": {}
            }
            
            for protocol_id, protocol in self.emergency_protocols.items():
                summary["protocols"][protocol_id] = {
                    "title": protocol["title"],
                    "emergency_level": protocol.get("emergency_level", "medium"),
                    "call_911": protocol.get("call_911", False),
                    "immediate_actions_count": len(protocol.get("immediate_actions", [])),
                    "warning_signs_count": len(protocol.get("warning_signs", []))
                }
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error getting emergency summary: {e}")
            return {"error": str(e)}

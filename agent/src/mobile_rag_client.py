"""
Mobile RAG Client for Testing Health Emergency Queries
Integrates with existing prediction.py and rag_client.py for mobile device simulation
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys
import os

# Add the current directory to the path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import existing modules
from rag_client import RAGAnythingClient
from mobile_rag_system import MobileHealthRAG

logger = logging.getLogger(__name__)

class MobileRAGClient:
    """Mobile RAG client that integrates prediction and RAG capabilities"""
    
    def __init__(self, 
                 mobile_rag_dir: Optional[str] = None,
                 rag_server_url: str = "http://localhost:9999"):
        """
        Initialize mobile RAG client
        
        Args:
            mobile_rag_dir: Directory containing mobile RAG data (auto-detected if None)
            rag_server_url: URL of the RAG-Anything server
        """
        # Auto-detect mobile_rag_ready directory if not provided
        if mobile_rag_dir is None:
            mobile_rag_dir = self._find_mobile_rag_directory()
        
        self.mobile_rag_dir = Path(mobile_rag_dir)
        self.rag_server_url = rag_server_url
        
        # Initialize mobile RAG system
        self.mobile_rag = MobileHealthRAG(
            models_dir="mobile_models",
            data_dir=str(self.mobile_rag_dir)
        )
        
        # Initialize RAG-Anything client
        self.rag_client = RAGAnythingClient(base_url=rag_server_url)
        
        logger.info("📱 Mobile RAG Client initialized")
        logger.info(f"📁 Mobile RAG Directory: {self.mobile_rag_dir}")
        logger.info(f"🔗 RAG Server URL: {self.rag_server_url}")
    
    def _find_mobile_rag_directory(self) -> str:
        """Find the mobile_rag_ready directory in multiple possible locations"""
        possible_paths = [
            Path("mobile_rag_ready"),  # Current directory
            Path("../mobile_rag_ready"),  # Parent directory
            Path("../../mobile_rag_ready"),  # Two levels up
            Path("../../agent/mobile_rag_ready"),  # Agent subdirectory
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"✅ Found mobile_rag_ready at: {path.absolute()}")
                return str(path)
        
        # If not found, default to current directory (will raise error later)
        logger.warning("⚠️ mobile_rag_ready directory not found, using default")
        return "mobile_rag_ready"
    
    def test_emergency_query(self, query: str, user_profile: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Test an emergency query using mobile RAG system
        
        Args:
            query: Health emergency query
            user_profile: Optional user profile information
            
        Returns:
            Dict containing response and metadata
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Testing emergency query: {query[:50]}...")
            
            # Use mobile RAG system for emergency detection
            mobile_response = self.mobile_rag.query_emergency(query)
            
            # Try to get additional context from RAG-Anything server
            rag_context = []
            try:
                rag_chunks = self.rag_client.retrieve(query, k=3)
                rag_context = rag_chunks
            except Exception as e:
                logger.warning(f"⚠️ RAG-Anything server unavailable: {e}")
            
            # Combine responses
            response = {
                "query": query,
                "mobile_rag_response": mobile_response,
                "rag_context": rag_context,
                "user_profile": user_profile,
                "processing_time": time.time() - start_time,
                "timestamp": time.time(),
                "success": True
            }
            
            logger.info(f"✅ Query processed in {response['processing_time']:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error processing emergency query: {e}")
            return {
                "query": query,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "timestamp": time.time(),
                "success": False
            }
    
    def test_vector_search(self, query: str, k: int = 5) -> Dict[str, Any]:
        """
        Test vector search capabilities
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            Dict containing search results
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Testing vector search: {query[:50]}...")
            
            # Test RAG-Anything server
            rag_results = []
            try:
                rag_results = self.rag_client.retrieve(query, k=k)
            except Exception as e:
                logger.warning(f"⚠️ RAG-Anything server unavailable: {e}")
            
            # Test mobile RAG guidelines search
            mobile_results = self.mobile_rag.search_guidelines(query, limit=k)
            
            response = {
                "query": query,
                "rag_server_results": rag_results,
                "mobile_rag_results": mobile_results,
                "processing_time": time.time() - start_time,
                "timestamp": time.time(),
                "success": True
            }
            
            logger.info(f"✅ Vector search completed in {response['processing_time']:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in vector search: {e}")
            return {
                "query": query,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "timestamp": time.time(),
                "success": False
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            mobile_status = self.mobile_rag.get_system_status()
            
            # Test RAG-Anything server connectivity
            rag_server_status = "unknown"
            try:
                # Simple connectivity test
                test_results = self.rag_client.retrieve("test", k=1)
                rag_server_status = "connected"
            except Exception as e:
                rag_server_status = f"disconnected: {str(e)}"
            
            return {
                "mobile_rag_status": mobile_status,
                "rag_server_status": rag_server_status,
                "mobile_rag_dir": str(self.mobile_rag_dir),
                "rag_server_url": self.rag_server_url,
                "system_ready": mobile_status.get("system_ready", False),
                "timestamp": time.time()
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting system status: {e}")
            return {
                "error": str(e),
                "timestamp": time.time(),
                "system_ready": False
            }
    
    def test_emergency_protocols(self) -> Dict[str, Any]:
        """Test emergency protocols functionality"""
        try:
            protocols = self.mobile_rag.get_all_protocols()
            emergency_summary = self.mobile_rag.get_emergency_summary()
            
            return {
                "protocols": protocols,
                "emergency_summary": emergency_summary,
                "protocol_count": len(protocols),
                "timestamp": time.time(),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"❌ Error testing emergency protocols: {e}")
            return {
                "error": str(e),
                "timestamp": time.time(),
                "success": False
            }
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive test of all capabilities"""
        test_results = {
            "system_status": self.get_system_status(),
            "emergency_protocols": self.test_emergency_protocols(),
            "test_queries": []
        }
        
        # Test emergency queries
        test_queries = [
            "Someone is having severe chest pain and can't breathe",
            "A person fainted and is unconscious",
            "There's a severe burn on someone's arm",
            "Someone is choking and can't speak",
            "Person showing signs of stroke with facial droop"
        ]
        
        for query in test_queries:
            result = self.test_emergency_query(query)
            test_results["test_queries"].append(result)
        
        # Test vector search
        test_results["vector_search"] = self.test_vector_search("chest pain emergency", k=3)
        
        return test_results

# Usage example and testing
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("🏥 Mobile RAG Client Test")
    print("=" * 50)
    
    # Initialize client
    client = MobileRAGClient()
    
    # Get system status
    print("\n📊 System Status:")
    status = client.get_system_status()
    print(json.dumps(status, indent=2))
    
    # Test emergency query
    print("\n🚨 Testing Emergency Query:")
    test_query = "Someone is having severe chest pain and shortness of breath"
    result = client.test_emergency_query(test_query)
    print(f"Query: {test_query}")
    print(f"Response: {json.dumps(result, indent=2)}")
    
    # Test vector search
    print("\n🔍 Testing Vector Search:")
    search_result = client.test_vector_search("heart attack symptoms", k=2)
    print(f"Search Results: {json.dumps(search_result, indent=2)}")

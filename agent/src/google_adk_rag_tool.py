"""
Google ADK RAG Tool for Health Emergency Assistance
Integrates with RAG-Anything instance for health emergency queries
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

class GoogleADKRAGTool:
    """
    Google ADK Tool for Health Emergency RAG System
    Provides structured tool interface for Google Assistant Development Kit
    """
    
    def __init__(self, 
                 rag_server_url: str = "http://localhost:9999",
                 mobile_rag_dir: Optional[str] = None):
        """
        Initialize Google ADK RAG Tool
        
        Args:
            rag_server_url: URL of the RAG-Anything server
            mobile_rag_dir: Directory containing mobile RAG data (auto-detected if None)
        """
        self.rag_server_url = rag_server_url
        
        # Initialize RAG-Anything client
        self.rag_client = RAGAnythingClient(base_url=rag_server_url)
        
        # Initialize mobile RAG system
        if mobile_rag_dir is None:
            mobile_rag_dir = self._find_mobile_rag_directory()
        
        self.mobile_rag = MobileHealthRAG(
            models_dir="mobile_models",
            data_dir=mobile_rag_dir
        )
        
        logger.info("🔧 Google ADK RAG Tool initialized")
        logger.info(f"🔗 RAG Server: {rag_server_url}")
        logger.info(f"📱 Mobile RAG: {mobile_rag_dir}")
    
    def _find_mobile_rag_directory(self) -> str:
        """Find the mobile_rag_ready directory"""
        possible_paths = [
            Path("mobile_rag_ready"),
            Path("../mobile_rag_ready"),
            Path("../../mobile_rag_ready"),
            Path("../../agent/mobile_rag_ready"),
        ]
        
        for path in possible_paths:
            if path.exists():
                logger.info(f"✅ Found mobile_rag_ready at: {path.absolute()}")
                return str(path)
        
        logger.warning("⚠️ mobile_rag_ready directory not found, using default")
        return "mobile_rag_ready"
    
    def health_emergency_assistant(self, 
                                 query: str, 
                                 user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Google ADK Tool: Health Emergency Assistant
        
        This tool provides immediate health emergency assistance by:
        1. Analyzing the emergency query
        2. Retrieving relevant medical protocols
        3. Providing step-by-step emergency response guidance
        4. Determining if 911 should be called
        
        Args:
            query: Health emergency description
            user_context: Optional user profile (age, conditions, location)
            
        Returns:
            Dict containing emergency response guidance
        """
        start_time = time.time()
        
        try:
            logger.info(f"🚨 Health Emergency Query: {query[:100]}...")
            
            # Step 1: Use mobile RAG for immediate emergency detection
            mobile_response = self.mobile_rag.query_emergency(query)
            
            # Step 2: Get additional context from RAG-Anything server
            rag_context = []
            try:
                rag_chunks = self.rag_client.retrieve(query, k=5)
                rag_context = rag_chunks
                logger.info(f"📚 Retrieved {len(rag_chunks)} relevant documents")
            except Exception as e:
                logger.warning(f"⚠️ RAG-Anything server unavailable: {e}")
            
            # Step 3: Combine and structure response for Google ADK
            response = {
                "tool_name": "health_emergency_assistant",
                "query": query,
                "emergency_detected": mobile_response.get("emergency_type") != "general_health",
                "emergency_type": mobile_response.get("emergency_type", "unknown"),
                "immediate_actions": mobile_response.get("immediate_actions", []),
                "call_911": mobile_response.get("call_911", False),
                "confidence": mobile_response.get("confidence", 0.0),
                "warning_signs": mobile_response.get("warning_signs", []),
                "additional_context": rag_context,
                "user_context": user_context,
                "processing_time": time.time() - start_time,
                "timestamp": time.time(),
                "success": True
            }
            
            # Add Google ADK specific formatting
            response["adk_response"] = self._format_for_google_adk(response)
            
            logger.info(f"✅ Emergency response generated in {response['processing_time']:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in health emergency assistant: {e}")
            return {
                "tool_name": "health_emergency_assistant",
                "query": query,
                "error": str(e),
                "call_911": True,  # Default to calling 911 on error
                "processing_time": time.time() - start_time,
                "timestamp": time.time(),
                "success": False,
                "adk_response": "I'm unable to process this health emergency. Please call 911 immediately."
            }
    
    def health_information_search(self, 
                                query: str, 
                                search_type: str = "general") -> Dict[str, Any]:
        """
        Google ADK Tool: Health Information Search
        
        Searches health information and guidelines for non-emergency queries
        
        Args:
            query: Health information query
            search_type: Type of search (general, symptoms, treatment, prevention)
            
        Returns:
            Dict containing health information
        """
        start_time = time.time()
        
        try:
            logger.info(f"🔍 Health Information Search: {query[:100]}...")
            
            # Search mobile RAG guidelines
            mobile_results = self.mobile_rag.search_guidelines(query, limit=5)
            
            # Search RAG-Anything server
            rag_results = []
            try:
                rag_results = self.rag_client.retrieve(query, k=3)
            except Exception as e:
                logger.warning(f"⚠️ RAG-Anything server unavailable: {e}")
            
            response = {
                "tool_name": "health_information_search",
                "query": query,
                "search_type": search_type,
                "mobile_results": mobile_results,
                "rag_results": rag_results,
                "total_results": len(mobile_results) + len(rag_results),
                "processing_time": time.time() - start_time,
                "timestamp": time.time(),
                "success": True
            }
            
            # Add Google ADK specific formatting
            response["adk_response"] = self._format_search_for_google_adk(response)
            
            logger.info(f"✅ Health search completed in {response['processing_time']:.2f}s")
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in health information search: {e}")
            return {
                "tool_name": "health_information_search",
                "query": query,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "timestamp": time.time(),
                "success": False,
                "adk_response": "I'm unable to search health information at this time."
            }
    
    def get_emergency_protocols(self) -> Dict[str, Any]:
        """
        Google ADK Tool: Get Available Emergency Protocols
        
        Returns all available emergency protocols for reference
        
        Returns:
            Dict containing all emergency protocols
        """
        try:
            protocols = self.mobile_rag.get_all_protocols()
            emergency_summary = self.mobile_rag.get_emergency_summary()
            
            return {
                "tool_name": "get_emergency_protocols",
                "protocols": protocols,
                "emergency_summary": emergency_summary,
                "protocol_count": len(protocols),
                "timestamp": time.time(),
                "success": True,
                "adk_response": f"Available {len(protocols)} emergency protocols ready for health emergencies."
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting emergency protocols: {e}")
            return {
                "tool_name": "get_emergency_protocols",
                "error": str(e),
                "timestamp": time.time(),
                "success": False,
                "adk_response": "Unable to retrieve emergency protocols."
            }
    
    def _format_for_google_adk(self, response: Dict[str, Any]) -> str:
        """Format response for Google ADK display with natural language explanations"""
        if not response.get("success"):
            return "I'm unable to process this health emergency. Please call 911 immediately."
        
        emergency_type = response.get('emergency_type', 'general_health')
        call_911 = response.get("call_911", False)
        immediate_actions = response.get("immediate_actions", [])
        warning_signs = response.get("warning_signs", [])
        
        # Use model-generated response if available, otherwise fallback to static responses
        generated_response = response.get("generated_response")
        if generated_response and generated_response != "Model not available for text generation.":
            return generated_response
        
        # Fallback to static responses based on emergency type
        if emergency_type == "chest_pain":
            if call_911:
                return f"Based on your symptoms, this appears to be a potential heart attack or cardiac emergency. " \
                       f"Chest pain with shortness of breath is a serious medical emergency that requires immediate attention. " \
                       f"You should call 911 right away and try to stay calm while waiting for help. " \
                       f"While waiting, sit down, loosen any tight clothing, and avoid any physical exertion."
            else:
                return f"Your chest pain symptoms could indicate several conditions ranging from heartburn to anxiety. " \
                       f"However, any chest pain should be taken seriously and evaluated by a healthcare provider. " \
                       f"Monitor your symptoms closely and seek medical attention if they worsen or persist."
        
        elif emergency_type == "fainting":
            if call_911:
                return f"Fainting with potential head injury is a serious medical emergency that requires immediate attention. " \
                       f"Loss of consciousness can indicate various serious conditions including head trauma, cardiac issues, or neurological problems. " \
                       f"Call 911 immediately and while waiting, check if the person is breathing and position them on their side if possible."
            else:
                return f"Fainting episodes can have various causes including dehydration, low blood pressure, or stress. " \
                       f"However, any loss of consciousness should be evaluated by a healthcare provider to rule out serious conditions. " \
                       f"Monitor the person closely and seek medical attention if symptoms persist or worsen."
        
        elif emergency_type == "choking":
            return f"Choking is a life-threatening emergency that requires immediate action. " \
                   f"When someone cannot speak or breathe due to a blocked airway, every second counts. " \
                   f"Call 911 immediately and perform the Heimlich maneuver if you're trained to do so, or encourage the person to cough forcefully."
        
        elif emergency_type == "stroke":
            return f"Facial drooping is a classic sign of stroke, which is a medical emergency that requires immediate treatment. " \
                   f"Time is critical with strokes - the sooner treatment begins, the better the outcome. " \
                   f"Call 911 immediately and note the time when symptoms started, as this information is crucial for treatment decisions."
        
        elif emergency_type == "general_health":
            # Use model-generated response if available, otherwise fallback to static responses
            generated_response = response.get("generated_response")
            if generated_response and generated_response != "Model not available for text generation.":
                return generated_response
            
            # Fallback to static responses for specific symptoms
            if "shortness of breath" in response.get("query", "").lower() and "pale" in response.get("query", "").lower():
                return f"Your symptoms of shortness of breath with pale, cold skin could indicate several serious conditions including shock, heart problems, or severe respiratory issues. " \
                       f"These symptoms suggest your body may not be getting enough oxygen, which is a medical emergency. " \
                       f"You should call 911 immediately and try to stay calm while waiting for help."
            
            elif "allergic reaction" in response.get("query", "").lower() and "throat" in response.get("query", "").lower():
                return f"Throat swelling during an allergic reaction is a medical emergency that can quickly become life-threatening. " \
                       f"Anaphylaxis can cause the airway to close completely, making it impossible to breathe. " \
                       f"Call 911 immediately and if you have an epinephrine auto-injector, use it right away."
            
            else:
                return f"Your symptoms could indicate various health conditions that require medical evaluation. " \
                       f"While I cannot provide a specific diagnosis, it's important to monitor your symptoms and seek medical attention if they worsen. " \
                       f"If you're experiencing severe symptoms or are concerned, don't hesitate to call 911 or visit an emergency room."
        
        else:
            return f"Your symptoms require medical attention and should be evaluated by a healthcare provider. " \
                   f"While I cannot provide a specific diagnosis, it's important to take your symptoms seriously. " \
                   f"If you're experiencing severe or worsening symptoms, call 911 or seek immediate medical care."
    
    def _format_search_for_google_adk(self, response: Dict[str, Any]) -> str:
        """Format search results for Google ADK display"""
        if not response.get("success"):
            return "Unable to search health information at this time."
        
        results = response.get("mobile_results", [])
        if results:
            return f"Found {len(results)} health information results:\n\n" + \
                   "\n".join([f"• {result.get('title', 'Unknown')}" for result in results[:3]])
        else:
            return "No specific health information found. Please provide more details."
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status for monitoring"""
        try:
            mobile_status = self.mobile_rag.get_system_status()
            
            # Test RAG-Anything server
            rag_status = "unknown"
            try:
                test_results = self.rag_client.retrieve("test", k=1)
                rag_status = "connected"
            except Exception as e:
                rag_status = f"disconnected: {str(e)}"
            
            return {
                "tool_name": "google_adk_rag_tool",
                "mobile_rag_status": mobile_status,
                "rag_server_status": rag_status,
                "system_ready": mobile_status.get("system_ready", False),
                "timestamp": time.time(),
                "success": True
            }
            
        except Exception as e:
            return {
                "tool_name": "google_adk_rag_tool",
                "error": str(e),
                "timestamp": time.time(),
                "success": False
            }


def test_rag_pipeline():
    """
    Test function to run the RAG pipeline with various prompts
    """
    print("🧪 Testing RAG Pipeline with Google ADK Tool")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Initialize the Google ADK RAG Tool
        print("🔧 Initializing Google ADK RAG Tool...")
        adk_tool = GoogleADKRAGTool()
        
        # Test system status
        print("\n📊 System Status:")
        status = adk_tool.get_system_status()
        print(f"   System Ready: {status.get('system_ready', False)}")
        print(f"   Mobile RAG: {status.get('mobile_rag_status', {}).get('system_ready', False)}")
        print(f"   RAG Server: {status.get('rag_server_status', 'unknown')}")
        
        # Test emergency queries
        emergency_queries = [
            "Someone is having severe chest pain and can't breathe",
            "A person fainted and is unconscious",
            "There's a severe burn on someone's arm",
            "Someone is choking and can't speak",
            "Person showing signs of stroke with facial droop"
        ]
        
        print("\n🚨 Testing Emergency Queries:")
        for i, query in enumerate(emergency_queries, 1):
            print(f"\n{i}. Query: {query}")
            result = adk_tool.health_emergency_assistant(query)
            
            if result.get("success"):
                print(f"   ✅ Success: {result.get('emergency_type', 'unknown')}")
                print(f"   🚨 Call 911: {result.get('call_911', False)}")
                print(f"   ⏱️  Processing Time: {result.get('processing_time', 0.0):.2f}s")
                print(f"   📝 ADK Response: {result.get('adk_response', '')[:100]}...")
            else:
                print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
        
        # Test health information search
        print("\n🔍 Testing Health Information Search:")
        search_queries = [
            "heart attack symptoms",
            "first aid for burns",
            "stroke warning signs",
            "choking prevention"
        ]
        
        for i, query in enumerate(search_queries, 1):
            print(f"\n{i}. Search: {query}")
            result = adk_tool.health_information_search(query)
            
            if result.get("success"):
                print(f"   ✅ Results: {result.get('total_results', 0)}")
                print(f"   ⏱️  Processing Time: {result.get('processing_time', 0.0):.2f}s")
                print(f"   📝 ADK Response: {result.get('adk_response', '')[:100]}...")
            else:
                print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
        
        # Test emergency protocols
        print("\n📋 Testing Emergency Protocols:")
        protocols_result = adk_tool.get_emergency_protocols()
        if protocols_result.get("success"):
            print(f"   ✅ Protocols Available: {protocols_result.get('protocol_count', 0)}")
            print(f"   📝 ADK Response: {protocols_result.get('adk_response', '')}")
        else:
            print(f"   ❌ Error: {protocols_result.get('error', 'Unknown error')}")
        
        print("\n✅ RAG Pipeline Test Completed!")
        print("\n💡 Google ADK Tool is ready for integration!")
        
    except Exception as e:
        print(f"❌ Error testing RAG pipeline: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_rag_pipeline()

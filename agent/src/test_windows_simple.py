#!/usr/bin/env python3
"""
Simple Windows RAG System Test
Quick test to ensure the Windows RAG system works properly
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_windows_rag():
    """Test the Windows RAG system with a simple query"""
    try:
        print("🧪 Testing Windows RAG System...")
        print("=" * 50)
        
        # Import the Windows RAG system
        from windows_rag_system import WindowsRAGSystem
        
        # Initialize the system
        print("🔧 Initializing Windows RAG System...")
        rag_system = WindowsRAGSystem()
        
        # Check system status
        status = rag_system.get_system_status()
        print(f"\n📊 System Status:")
        print(f"   Guidelines: {status['guidelines_loaded']}")
        print(f"   Emergency Protocols: {status['emergency_protocols']}")
        print(f"   LLM Model: {'✅' if status['llm_model_loaded'] else '❌'}")
        print(f"   Embedding Model: {'✅' if status['embedding_model_loaded'] else '❌'}")
        print(f"   Vector Index: {'✅' if status['vector_index_built'] else '❌'}")
        print(f"   Device: {status['device']}")
        
        # Test with a simple query
        print(f"\n🔍 Testing with sample query...")
        test_query = "I have chest pain and shortness of breath"
        print(f"Query: {test_query}")
        
        result = rag_system.query_health_emergency(test_query)
        
        print(f"\n🤖 Response:")
        print(f"   {result.get('natural_response', 'No response available')}")
        
        print(f"\n📋 Analysis:")
        print(f"   Emergency Type: {result.get('emergency_type', 'Unknown').replace('_', ' ').title()}")
        print(f"   Call 911: {'YES' if result.get('call_911') else 'NO'}")
        print(f"   Confidence: {result.get('confidence', 0.0):.1%}")
        print(f"   Processing Time: {result.get('processing_time', 0.0):.2f}s")
        
        if result.get('immediate_actions'):
            print(f"\n⚡ Immediate Actions:")
            for i, action in enumerate(result['immediate_actions'], 1):
                print(f"   {i}. {action}")
        
        print(f"\n✅ Windows RAG System test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing Windows RAG system: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_windows_rag()
    sys.exit(0 if success else 1)

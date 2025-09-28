#!/usr/bin/env python3
"""
Simple Workflow Test for Health RAG System
Tests the basic workflow of the health emergency RAG system
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_simple_workflow():
    """Test the simple workflow of the health RAG system"""
    try:
        print("🧪 Testing Simple Health RAG Workflow...")
        print("=" * 50)
        
        # Import the RAG system
        from windows_rag_system import WindowsRAGSystem
        
        # Initialize the system
        print("🔧 Initializing RAG System...")
        rag_system = WindowsRAGSystem()
        
        # Check system status
        status = rag_system.get_system_status()
        print(f"\n📊 System Status:")
        print(f"   Guidelines: {status['guidelines_loaded']}")
        print(f"   Emergency Protocols: {status['emergency_protocols']}")
        print(f"   LLM Model: {'✅' if status['llm_model_loaded'] else '❌'}")
        print(f"   Embedding Model: {'✅' if status['embedding_model_loaded'] else '❌'}")
        print(f"   Vector Index: {'✅' if status['vector_index_built'] else '❌'}")
        
        # Test scenarios
        test_scenarios = [
            {
                "query": "I have chest pain",
                "expected_911": True,
                "description": "Chest pain emergency"
            },
            {
                "query": "I have a headache",
                "expected_911": False,
                "description": "Non-emergency headache"
            },
            {
                "query": "Someone is choking",
                "expected_911": True,
                "description": "Choking emergency"
            }
        ]
        
        print(f"\n🔍 Testing {len(test_scenarios)} scenarios...")
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n--- Scenario {i}: {scenario['description']} ---")
            print(f"Query: {scenario['query']}")
            
            # Process the query
            result = rag_system.query_health_emergency(scenario['query'])
            
            # Display results
            print(f"Response: {result.get('natural_response', 'No response')[:100]}...")
            print(f"Emergency Type: {result.get('emergency_type', 'Unknown')}")
            print(f"Call 911: {'YES' if result.get('call_911') else 'NO'}")
            print(f"Confidence: {result.get('confidence', 0.0):.1%}")
            
            # Check if 911 recommendation matches expectation
            actual_911 = result.get('call_911', False)
            expected_911 = scenario['expected_911']
            match = "✅" if actual_911 == expected_911 else "❌"
            print(f"Expected 911: {expected_911} | Actual: {actual_911} {match}")
            
            if result.get('immediate_actions'):
                print("Immediate Actions:")
                for j, action in enumerate(result['immediate_actions'], 1):
                    print(f"  {j}. {action}")
        
        print(f"\n✅ Simple workflow test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error in simple workflow test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_simple_workflow()
    sys.exit(0 if success else 1)

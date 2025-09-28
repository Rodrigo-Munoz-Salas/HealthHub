#!/usr/bin/env python3
"""
Batch Test Interface for Windows RAG System
Test multiple health queries with Qwen2.5-0.5B-Instruct
"""

import sys
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_batch_queries():
    """Batch test multiple health queries"""
    try:
        print("🏥 Batch Testing Windows RAG System")
        print("=" * 60)
        print("Using Qwen2.5-0.5B-Instruct model for health queries")
        print("=" * 60)
        
        # Import the Windows RAG system
        from windows_rag_system import WindowsRAGSystem
        
        # Initialize the system
        print("🔧 Initializing Windows RAG System...")
        start_time = time.time()
        rag_system = WindowsRAGSystem()
        init_time = time.time() - start_time
        
        # Check system status
        status = rag_system.get_system_status()
        print(f"\n📊 System Status:")
        print(f"   Guidelines: {status['guidelines_loaded']}")
        print(f"   Emergency Protocols: {status['emergency_protocols']}")
        print(f"   LLM Model: {'✅' if status['llm_model_loaded'] else '❌'}")
        print(f"   Embedding Model: {'✅' if status['embedding_model_loaded'] else '❌'}")
        print(f"   Vector Index: {'✅' if status['vector_index_built'] else '❌'}")
        print(f"   Device: {status['device']}")
        print(f"   Initialization Time: {init_time:.2f}s")
        
        if not status['llm_model_loaded']:
            print("\n❌ LLM Model not loaded! Cannot proceed with testing.")
            return False
        
        print(f"\n✅ System ready! Model loaded in {init_time:.2f}s")
        
        # Test queries
        test_queries = [
            "I have chest pain",
            "I fell down the stairs", 
            "I can't breathe properly",
            "I have a severe headache",
            "I feel dizzy and nauseous",
            "I have a fever and body aches",
            "I cut my finger and it's bleeding",
            "I have stomach pain",
            "I have a sore throat",
            "I feel very tired"
        ]
        
        print(f"\n🧪 Testing {len(test_queries)} health queries...")
        print("=" * 60)
        
        total_time = 0
        successful_queries = 0
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: '{query}'")
            print("-" * 40)
            
            try:
                start_time = time.time()
                result = rag_system.query_health_emergency(query)
                processing_time = time.time() - start_time
                total_time += processing_time
                
                print(f"⏱️ Processing Time: {processing_time:.2f}s")
                print(f"🚨 Emergency Type: {result.get('emergency_type', 'Unknown')}")
                print(f"📞 Call 911: {'YES' if result.get('call_911') else 'NO'}")
                print(f"🎯 Confidence: {result.get('confidence', 0.0):.1%}")
                
                # Show AI response
                ai_response = result.get('ai_response')
                if ai_response and ai_response != "Model not available for text generation.":
                    print(f"🤖 AI Response: {ai_response}")
                    successful_queries += 1
                else:
                    print(f"⚠️ AI Response: Not available (using rule-based response)")
                
                # Show natural response
                natural_response = result.get('natural_response', '')
                if natural_response:
                    print(f"📋 Full Response: {natural_response[:100]}...")
                
            except Exception as e:
                print(f"❌ Error processing query: {e}")
        
        # Summary
        print(f"\n📊 Batch Test Summary:")
        print(f"   Total Queries: {len(test_queries)}")
        print(f"   Successful AI Responses: {successful_queries}")
        print(f"   Average Processing Time: {total_time/len(test_queries):.2f}s")
        print(f"   Total Time: {total_time:.2f}s")
        print(f"   Success Rate: {successful_queries/len(test_queries):.1%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in batch testing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_batch_queries()
    sys.exit(0 if success else 1)


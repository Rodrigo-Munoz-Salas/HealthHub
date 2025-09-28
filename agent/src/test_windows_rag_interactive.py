#!/usr/bin/env python3
"""
Interactive Test Interface for Windows RAG System
Test different health queries with Qwen2.5-0.5B-Instruct
"""

import sys
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_interactive_rag():
    """Interactive test interface for Windows RAG system"""
    try:
        print("🏥 Interactive Windows RAG System Test")
        print("=" * 60)
        print("Using Qwen2.5-0.5B-Instruct model for health queries")
        print("Type 'quit' to exit, 'help' for examples")
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
        print("\n" + "=" * 60)
        
        # Interactive testing loop
        while True:
            try:
                # Get user input
                user_input = input("\n🏥 Enter health query (or 'quit'/'help'): ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Goodbye!")
                    break
                
                if user_input.lower() == 'help':
                    print("\n📋 Example queries to try:")
                    print("   • I have chest pain")
                    print("   • I fell down the stairs")
                    print("   • I can't breathe properly")
                    print("   • I have a severe headache")
                    print("   • I feel dizzy and nauseous")
                    print("   • I have a fever and body aches")
                    print("   • I cut my finger and it's bleeding")
                    print("   • I have stomach pain")
                    continue
                
                if not user_input:
                    print("⚠️ Please enter a health query or 'quit'/'help'")
                    continue
                
                # Process the query
                print(f"\n🔍 Processing: '{user_input}'")
                print("-" * 40)
                
                start_time = time.time()
                result = rag_system.query_health_emergency(user_input)
                processing_time = time.time() - start_time
                
                # Display results
                print(f"⏱️ Processing Time: {processing_time:.2f}s")
                print(f"🚨 Emergency Type: {result.get('emergency_type', 'Unknown')}")
                print(f"📞 Call 911: {'YES' if result.get('call_911') else 'NO'}")
                print(f"🎯 Confidence: {result.get('confidence', 0.0):.1%}")
                
                # Show AI response
                ai_response = result.get('ai_response')
                if ai_response and ai_response != "Model not available for text generation.":
                    print(f"\n🤖 AI Response:")
                    print(f"   {ai_response}")
                else:
                    print(f"\n⚠️ AI Response: Not available (using rule-based response)")
                
                # Show natural response
                natural_response = result.get('natural_response', '')
                if natural_response:
                    print(f"\n📋 Full Response:")
                    print(f"   {natural_response}")
                
                # Show vector search results if available
                vector_results = result.get('vector_results', [])
                if vector_results:
                    print(f"\n📚 Relevant Health Information:")
                    for i, result_item in enumerate(vector_results[:2], 1):
                        title = result_item.get('title', 'Health Info')
                        content = result_item.get('content', '')[:100]
                        print(f"   {i}. {title}: {content}...")
                
                print("\n" + "=" * 60)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error processing query: {e}")
                print("Please try again or type 'quit' to exit.")
        
        return True
        
    except Exception as e:
        print(f"❌ Error initializing Windows RAG system: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_interactive_rag()
    sys.exit(0 if success else 1)

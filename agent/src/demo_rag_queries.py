"""
Demo RAG Pipeline with Example Queries
Shows natural language responses from the health emergency RAG system
"""

import json
import time
import logging
from google_adk_rag_tool import GoogleADKRAGTool

def demo_rag_queries():
    """
    Demo the RAG pipeline with example health emergency queries
    """
    print("🏥 Health Emergency RAG Pipeline Demo")
    print("=" * 60)
    print("Testing various health emergency scenarios...")
    print()
    
    # Configure logging to reduce noise
    logging.basicConfig(level=logging.WARNING)
    
    try:
        # Initialize the Google ADK RAG Tool
        print("🔧 Initializing RAG Pipeline...")
        adk_tool = GoogleADKRAGTool()
        
        # Get system status
        status = adk_tool.get_system_status()
        if not status.get('system_ready'):
            print("❌ System not ready. Please check your setup.")
            return
        
        print("✅ RAG Pipeline Ready!")
        print(f"📚 Guidelines Loaded: {status.get('mobile_rag_status', {}).get('guidelines_loaded', 0)}")
        print(f"🚨 Emergency Protocols: {status.get('mobile_rag_status', {}).get('emergency_protocols', 0)}")
        print()
        
        # Example health emergency queries
        demo_queries = [
            {
                "query": "My friend is having severe chest pain and can't breathe properly",
                "description": "Heart attack symptoms"
            },
            {
                "query": "Someone just fainted and hit their head on the floor",
                "description": "Unconsciousness with head injury"
            },
            {
                "query": "A person is choking on food at the restaurant and can't speak",
                "description": "Choking emergency"
            },
            {
                "query": "My neighbor is showing signs of a stroke - one side of their face is drooping",
                "description": "Stroke symptoms"
            },
            {
                "query": "A child has a severe burn from hot water on their arm",
                "description": "Burn injury"
            }
        ]
        
        for i, demo in enumerate(demo_queries, 1):
            print(f"🚨 SCENARIO {i}: {demo['description']}")
            print(f"Query: {demo['query']}")
            print("-" * 50)
            
            # Process the query
            start_time = time.time()
            result = adk_tool.health_emergency_assistant(demo['query'])
            processing_time = time.time() - start_time
            
            # Show the natural language assistant response first (most important)
            adk_response = result.get('adk_response', '')
            if adk_response:
                print(f"\n🤖 HEALTH ASSISTANT RESPONSE:")
                print(adk_response)
            
            # Show additional technical details
            print(f"\n📋 TECHNICAL ANALYSIS:")
            print(f"Emergency Type: {result.get('emergency_type', 'Unknown').replace('_', ' ').title()}")
            print(f"Call 911: {'YES - IMMEDIATELY' if result.get('call_911') else 'NO - Monitor situation'}")
            print(f"Confidence Level: {result.get('confidence', 0.0):.1%}")
            print(f"Processing Time: {processing_time:.2f}s")
            
            # Show immediate actions
            immediate_actions = result.get('immediate_actions', [])
            if immediate_actions:
                print(f"\n⚡ IMMEDIATE ACTIONS:")
                for j, action in enumerate(immediate_actions, 1):
                    print(f"   {j}. {action}")
            
            # Show warning signs
            warning_signs = result.get('warning_signs', [])
            if warning_signs:
                print(f"\n⚠️ WARNING SIGNS TO WATCH FOR:")
                for sign in warning_signs:
                    print(f"   • {sign}")
            
            # Show information from vector database
            additional_context = result.get('additional_context', [])
            if additional_context:
                print(f"\n📚 INFORMATION FROM HEALTH DATABASE:")
                for j, context in enumerate(additional_context[:2], 1):
                    content = context.get('content', '')
                    if len(content) > 150:
                        content = content[:150] + "..."
                    source = context.get('source_id', 'Health Guidelines')
                    print(f"   {j}. {content}")
                    print(f"      Source: {source}")
            
            # Show any errors
            if not result.get('success'):
                print(f"\n❌ ERROR: {result.get('error', 'Unknown error')}")
            
            print("\n" + "=" * 60)
            print()
            
        print("✅ Demo completed! The RAG pipeline is working with natural language responses.")
        print("\n💡 To test your own query, modify the 'demo_queries' list in this file.")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_your_query(custom_query: str):
    """
    Test a specific custom query
    """
    print(f"🏥 Testing Custom Query: {custom_query}")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING)
    
    try:
        # Initialize the Google ADK RAG Tool
        adk_tool = GoogleADKRAGTool()
        
        # Process the custom query
        start_time = time.time()
        result = adk_tool.health_emergency_assistant(custom_query)
        processing_time = time.time() - start_time
        
        # Show the natural language assistant response first (most important)
        adk_response = result.get('adk_response', '')
        if adk_response:
            print(f"\n🤖 HEALTH ASSISTANT RESPONSE:")
            print(adk_response)
        
        # Show model-generated response if available
        generated_response = result.get('mobile_rag_response', {}).get('generated_response')
        if generated_response and generated_response != "Model not available for text generation.":
            print(f"\n🧠 MODEL-GENERATED RESPONSE:")
            print(generated_response)
        
        # Show additional technical details
        print(f"\n📋 TECHNICAL ANALYSIS:")
        print(f"Emergency Type: {result.get('emergency_type', 'Unknown').replace('_', ' ').title()}")
        print(f"Call 911: {'YES - IMMEDIATELY' if result.get('call_911') else 'NO - Monitor situation'}")
        print(f"Confidence Level: {result.get('confidence', 0.0):.1%}")
        print(f"Processing Time: {processing_time:.2f}s")
        
        # Show immediate actions
        immediate_actions = result.get('immediate_actions', [])
        if immediate_actions:
            print(f"\n⚡ IMMEDIATE ACTIONS:")
            for i, action in enumerate(immediate_actions, 1):
                print(f"   {i}. {action}")
        
        # Show warning signs
        warning_signs = result.get('warning_signs', [])
        if warning_signs:
            print(f"\n⚠️ WARNING SIGNS TO WATCH FOR:")
            for sign in warning_signs:
                print(f"   • {sign}")
        
        # Show information from vector database
        additional_context = result.get('additional_context', [])
        if additional_context:
            print(f"\n📚 INFORMATION FROM HEALTH DATABASE:")
            for i, context in enumerate(additional_context[:3], 1):
                content = context.get('content', '')
                if len(content) > 200:
                    content = content[:200] + "..."
                source = context.get('source_id', 'Health Guidelines')
                print(f"   {i}. {content}")
                print(f"      Source: {source}")
        
        # Show any errors
        if not result.get('success'):
            print(f"\n❌ ERROR: {result.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Test specific query from command line
        custom_query = " ".join(sys.argv[1:])
        test_your_query(custom_query)
    else:
        # Run demo with example queries
        demo_rag_queries()

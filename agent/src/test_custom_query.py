"""
Test Custom Query with RAG Pipeline
Interactive test to input your own health emergency queries
"""

import json
import time
import logging
from google_adk_rag_tool import GoogleADKRAGTool

def test_custom_query():
    """
    Interactive test function for custom health emergency queries
    """
    print("🏥 Health Emergency RAG Pipeline - Custom Query Test")
    print("=" * 60)
    print("Enter your health emergency query to see the RAG pipeline response")
    print("Type 'quit' to exit")
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
        
        while True:
            # Get user input
            query = input("🏥 Enter your health emergency query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not query:
                print("⚠️ Please enter a query or type 'quit' to exit")
                continue
            
            print(f"\n🔍 Processing: {query}")
            print("-" * 50)
            
            # Process the query
            start_time = time.time()
            result = adk_tool.health_emergency_assistant(query)
            processing_time = time.time() - start_time
            
            # Display results in natural language
            print(f"\n📋 EMERGENCY RESPONSE:")
            print(f"Emergency Type: {result.get('emergency_type', 'Unknown').replace('_', ' ').title()}")
            print(f"Call 911: {'YES' if result.get('call_911') else 'NO'}")
            print(f"Confidence: {result.get('confidence', 0.0):.1%}")
            print(f"Processing Time: {processing_time:.2f}s")
            
            # Show immediate actions
            immediate_actions = result.get('immediate_actions', [])
            if immediate_actions:
                print(f"\n🚨 IMMEDIATE ACTIONS:")
                for i, action in enumerate(immediate_actions, 1):
                    print(f"   {i}. {action}")
            
            # Show warning signs if available
            warning_signs = result.get('warning_signs', [])
            if warning_signs:
                print(f"\n⚠️ WARNING SIGNS TO WATCH FOR:")
                for sign in warning_signs:
                    print(f"   • {sign}")
            
            # Show additional context from vector database
            additional_context = result.get('additional_context', [])
            if additional_context:
                print(f"\n📚 ADDITIONAL INFORMATION FROM DATABASE:")
                for i, context in enumerate(additional_context[:2], 1):  # Show first 2 results
                    content = context.get('content', '')[:200] + "..." if len(context.get('content', '')) > 200 else context.get('content', '')
                    source = context.get('source_id', 'Unknown source')
                    print(f"   {i}. {content}")
                    print(f"      Source: {source}")
            
            # Show the formatted ADK response
            adk_response = result.get('adk_response', '')
            if adk_response:
                print(f"\n🤖 ASSISTANT RESPONSE:")
                print(adk_response)
            
            # Show any errors
            if not result.get('success'):
                print(f"\n❌ ERROR: {result.get('error', 'Unknown error')}")
            
            print("\n" + "=" * 60)
            print()
            
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def test_specific_queries():
    """
    Test with specific predefined queries to demonstrate the system
    """
    print("🧪 Testing Specific Health Emergency Queries")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(level=logging.WARNING)
    
    try:
        # Initialize the Google ADK RAG Tool
        adk_tool = GoogleADKRAGTool()
        
        # Test queries
        test_queries = [
            "My friend is having severe chest pain and can't breathe properly",
            "Someone just fainted and hit their head",
            "There's a person choking on food at the restaurant",
            "My neighbor is showing signs of a stroke - one side of their face is drooping",
            "A child has a severe burn from hot water",
            "Someone is having an allergic reaction and their throat is swelling",
            "A person is having a seizure and shaking uncontrollably",
            "Someone fell and can't move their legs - possible spinal injury"
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. QUERY: {query}")
            print("-" * 50)
            
            result = adk_tool.health_emergency_assistant(query)
            
            # Natural language response
            print(f"🚨 EMERGENCY DETECTED: {result.get('emergency_type', 'Unknown').replace('_', ' ').title()}")
            print(f"📞 CALL 911: {'YES - IMMEDIATELY' if result.get('call_911') else 'NO - Monitor situation'}")
            
            if result.get('immediate_actions'):
                print(f"\n⚡ IMMEDIATE ACTIONS:")
                for action in result.get('immediate_actions', []):
                    print(f"   • {action}")
            
            if result.get('warning_signs'):
                print(f"\n⚠️ WATCH FOR:")
                for sign in result.get('warning_signs', []):
                    print(f"   • {sign}")
            
            print(f"\n🤖 ASSISTANT: {result.get('adk_response', '')}")
            print("=" * 60)
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        test_specific_queries()
    else:
        test_custom_query()

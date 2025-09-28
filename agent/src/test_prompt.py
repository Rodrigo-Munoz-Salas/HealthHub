"""
Simple script to test health emergency prompts
Usage: python test_prompt.py "Your health emergency query here"
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from local_rag_system import LocalHealthRAG
import logging

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(
        level=logging.WARNING,  # Reduce noise
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def print_response(result, query):
    """Print formatted response"""
    print("🤖 HEALTH ASSISTANT RESPONSE:")
    print(f"   {result.get('natural_response', 'No response available')}")
    print()
    
    print("📋 TECHNICAL ANALYSIS:")
    print(f"   Emergency Type: {result.get('emergency_type', 'Unknown').replace('_', ' ').title()}")
    print(f"   Call 911: {'🚨 YES - IMMEDIATELY' if result.get('call_911') else '✅ NO - Monitor situation'}")
    print(f"   Confidence: {result.get('confidence', 0.0):.1%}")
    print(f"   Processing Time: {result.get('processing_time', 0.0):.2f}s")
    print()
    
    # Show immediate actions if available
    immediate_actions = result.get('immediate_actions', [])
    if immediate_actions:
        print("⚡ IMMEDIATE ACTIONS:")
        for i, action in enumerate(immediate_actions, 1):
            print(f"   {i}. {action}")
        print()
    
    # Show warning signs if available
    warning_signs = result.get('warning_signs', [])
    if warning_signs:
        print("⚠️ WARNING SIGNS TO WATCH FOR:")
        for sign in warning_signs:
            print(f"   • {sign}")
        print()
    
    # Show vector search results if available
    vector_results = result.get('vector_results', [])
    if vector_results:
        print("📚 RELEVANT HEALTH INFORMATION:")
        for i, result_item in enumerate(vector_results[:2], 1):
            content = result_item.get('content', '')
            if len(content) > 200:
                content = content[:200] + "..."
            print(f"   {i}. {content}")
            print(f"      Source: {result_item.get('title', 'Health Guidelines')}")
        print()

def main():
    """Main function"""
    # Setup logging
    setup_logging()
    
    # Get query from command line arguments
    if len(sys.argv) < 2:
        print("Usage: python test_prompt.py \"Your health emergency query here\"")
        print("\nExample queries:")
        print("  python test_prompt.py \"I have severe chest pain and can't breathe\"")
        print("  python test_prompt.py \"Someone just fainted and hit their head\"")
        print("  python test_prompt.py \"A person is choking on food and can't speak\"")
        print("  python test_prompt.py \"My neighbor is showing signs of stroke\"")
        print("  python test_prompt.py \"I have shortness of breath, pale skin, and cold skin\"")
        return
    
    query = " ".join(sys.argv[1:])
    
    try:
        # Initialize RAG system
        print("🔧 Initializing Local RAG System...")
        rag_system = LocalHealthRAG()
        
        # Check system status
        status = rag_system.get_system_status()
        if not status['system_ready']:
            print("❌ System not ready. Please check your setup.")
            return
        
        print(f"✅ System ready! Processing query...")
        print()
        
        # Process query
        print("=" * 60)
        print(f"🚨 HEALTH EMERGENCY QUERY: {query}")
        print("=" * 60)
        
        result = rag_system.query_health_emergency(query)
        
        # Print response
        print_response(result, query)
        
        print("=" * 60)
        print("✅ Query processed successfully!")
        
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

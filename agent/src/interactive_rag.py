"""
Interactive RAG System Interface
Simple interface for testing health emergency queries
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

def print_banner():
    """Print welcome banner"""
    print("🏥" + "=" * 58 + "🏥")
    print("🏥" + " " * 20 + "HEALTH EMERGENCY RAG" + " " * 20 + "🏥")
    print("🏥" + " " * 15 + "Local AI Assistant" + " " * 15 + "🏥")
    print("🏥" + "=" * 58 + "🏥")
    print()

def print_system_status(rag_system):
    """Print system status"""
    status = rag_system.get_system_status()
    
    print("📊 SYSTEM STATUS:")
    print(f"   📚 Health Guidelines: {status['guidelines_loaded']}")
    print(f"   🚨 Emergency Protocols: {status['emergency_protocols']}")
    print(f"   🤖 AI Model: {'✅ Ready' if status['llm_model_loaded'] else '❌ Not Available'}")
    print(f"   🔍 Vector Search: {'✅ Ready' if status['vector_index_built'] else '❌ Not Available'}")
    print(f"   📊 Embeddings: {'✅ Ready' if status['embedding_model_loaded'] else '❌ Not Available'}")
    print()

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

def interactive_mode(rag_system):
    """Run interactive mode"""
    print("💬 INTERACTIVE MODE")
    print("Type your health emergency query and press Enter.")
    print("Type 'quit', 'exit', or 'q' to stop.")
    print("Type 'status' to see system status.")
    print("Type 'help' for example queries.")
    print()
    
    while True:
        try:
            # Get user input
            query = input("🏥 Health Query: ").strip()
            
            # Handle special commands
            if query.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye! Stay healthy!")
                break
            elif query.lower() == 'status':
                print_system_status(rag_system)
                continue
            elif query.lower() == 'help':
                print_help()
                continue
            elif not query:
                continue
            
            print("\n" + "=" * 60)
            print(f"🚨 Processing: {query}")
            print("=" * 60)
            
            # Process query
            result = rag_system.query_health_emergency(query)
            
            # Print response
            print_response(result, query)
            
            print("=" * 60)
            print()
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye! Stay healthy!")
            break
        except Exception as e:
            print(f"❌ Error processing query: {e}")
            print()

def print_help():
    """Print help information"""
    print("\n💡 EXAMPLE HEALTH QUERIES:")
    print("   • I have severe chest pain and can't breathe")
    print("   • Someone just fainted and hit their head")
    print("   • A person is choking on food and can't speak")
    print("   • My neighbor is showing signs of stroke")
    print("   • I have shortness of breath, pale skin, and cold skin")
    print("   • There's a severe burn on someone's arm")
    print("   • Someone is having an allergic reaction with throat swelling")
    print("   • A child has a high fever and is very lethargic")
    print()

def demo_mode(rag_system):
    """Run demo mode with predefined queries"""
    print("🎬 DEMO MODE")
    print("Running predefined health emergency scenarios...")
    print()
    
    demo_queries = [
        "I have severe chest pain and can't breathe properly",
        "Someone just fainted and hit their head on the floor",
        "A person is choking on food and can't speak",
        "My neighbor is showing signs of stroke with facial drooping",
        "I have shortness of breath, pale skin, and cold skin",
        "There's a severe burn on someone's arm from hot water",
        "Someone is having an allergic reaction with throat swelling"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"🚨 SCENARIO {i}: {query}")
        print("-" * 50)
        
        result = rag_system.query_health_emergency(query)
        print_response(result, query)
        
        print("=" * 60)
        print()
        
        # Pause between scenarios
        if i < len(demo_queries):
            input("Press Enter to continue to next scenario...")
            print()

def main():
    """Main function"""
    print_banner()
    
    # Setup logging
    setup_logging()
    
    try:
        # Initialize RAG system
        print("🔧 Initializing Local RAG System...")
        rag_system = LocalHealthRAG()
        
        # Print system status
        print_system_status(rag_system)
        
        # Check if system is ready
        status = rag_system.get_system_status()
        if not status['system_ready']:
            print("❌ System not ready. Please check your setup.")
            return
        
        # Choose mode
        print("🎯 CHOOSE MODE:")
        print("   1. Interactive Mode (type your own queries)")
        print("   2. Demo Mode (predefined scenarios)")
        print("   3. Single Query Mode")
        print()
        
        while True:
            choice = input("Enter your choice (1-3): ").strip()
            
            if choice == '1':
                interactive_mode(rag_system)
                break
            elif choice == '2':
                demo_mode(rag_system)
                break
            elif choice == '3':
                query = input("Enter your health emergency query: ").strip()
                if query:
                    print("\n" + "=" * 60)
                    print(f"🚨 Processing: {query}")
                    print("=" * 60)
                    result = rag_system.query_health_emergency(query)
                    print_response(result, query)
                    print("=" * 60)
                break
            else:
                print("Please enter 1, 2, or 3.")
        
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

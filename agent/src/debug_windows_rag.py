#!/usr/bin/env python3
"""
Debug Windows RAG System - Check why AI responses aren't working
"""

import sys
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_symptom_context(query: str) -> str:
    """Extract key symptoms from query for global context"""
    symptoms = []
    query_lower = query.lower()
    
    # Common emergency symptoms
    if any(word in query_lower for word in ['chest pain', 'chest discomfort']):
        symptoms.append("chest pain")
    if any(word in query_lower for word in ['shortness of breath', 'breathing', 'can\'t breathe']):
        symptoms.append("breathing difficulty")
    if any(word in query_lower for word in ['severe headache', 'head pain']):
        symptoms.append("severe headache")
    if any(word in query_lower for word in ['dizzy', 'dizziness', 'faint']):
        symptoms.append("dizziness")
    if any(word in query_lower for word in ['fever', 'high temperature']):
        symptoms.append("fever")
    if any(word in query_lower for word in ['nausea', 'vomiting', 'sick']):
        symptoms.append("nausea/vomiting")
    if any(word in query_lower for word in ['head injury', 'hit head', 'fell']):
        symptoms.append("head trauma")
    
    return ", ".join(symptoms) if symptoms else "general symptoms"

def debug_windows_rag():
    """Debug the Windows RAG system to see what's happening with AI responses"""
    try:
        print("🔍 Debugging Windows RAG System...")
        print("=" * 60)
        
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
        
        # Test AI model directly
        print(f"\n🤖 Testing AI Model Directly:")
        if rag_system.llm_model is not None:
            print("   ✅ LLM Model is loaded")
            
            # Test simple generation with globally applicable prompt
            test_prompt = "Emergency: \"I have chest pain\"\n\n- Chest Pain Emergency\n\nEmergency: Yes/No. Action: [seek hospital/see doctor]. Include key symptoms if specific. 25 words max."
            print(f"   Testing prompt: {test_prompt}...")
            
            try:
                ai_response = rag_system._generate_response(test_prompt, max_length=25)
                print(f"   AI Response: {ai_response}")
                print(f"   Response Length: {len(ai_response)}")
                print(f"   Is Empty: {ai_response == ''}")
                print(f"   Is 'Model not available': {ai_response == 'Model not available for text generation.'}")
            except Exception as e:
                print(f"   ❌ Error generating AI response: {e}")
        else:
            print("   ❌ LLM Model is NOT loaded")
            print("   🔍 Checking model paths...")
            
            # Check for model files
            possible_paths = [
                Path("mobile_models/qwen2_5_0_5b"),
                Path("../mobile_models/qwen2_5_0_5b"),
                Path("../../agent/mobile_models/qwen2_5_0_5b"),
                Path("mobile_models/quantized_tinyllama_health"),
                Path("../mobile_models/quantized_tinyllama_health"),
                Path("../../agent/mobile_models/quantized_tinyllama_health")
            ]
            
            for path in possible_paths:
                if path.exists():
                    print(f"   ✅ Found model at: {path}")
                    break
            else:
                print("   ❌ No model found in any expected location")
        
        # Test multiple scenarios with global applicability
        test_scenarios = [
            "I have severe chest pain and shortness of breath",
            "I fell down the stairs and hit my head",
            "I have a high fever with body aches",
            "I feel dizzy and nauseous",
            "I have a minor headache"
        ]
        
        print(f"\n🔍 Testing Multiple Scenarios with Global Advice:")
        for i, test_query in enumerate(test_scenarios, 1):
            print(f"\n--- Test {i}: {test_query} ---")
            symptoms = get_symptom_context(test_query)
            print(f"Detected Symptoms: {symptoms}")
            
            result = rag_system.query_health_emergency(test_query)
            
            print(f"Emergency Type: {result.get('emergency_type', 'Unknown')}")
            print(f"AI Response: {result.get('ai_response', 'NOT_FOUND')[:100]}...")
            print(f"Natural Response: {result.get('natural_response', 'NOT_FOUND')[:100]}...")
        
        # Detailed analysis of first scenario
        print(f"\n🔍 Detailed Analysis of First Scenario:")
        test_query = test_scenarios[0]
        print(f"Query: {test_query}")
        
        result = rag_system.query_health_emergency(test_query)
        
        print(f"\n📋 Debug Information:")
        print(f"   Emergency Type: {result.get('emergency_type', 'Unknown')}")
        print(f"   AI Response Available: {'ai_response' in result}")
        print(f"   AI Response Value: {result.get('ai_response', 'NOT_FOUND')}")
        print(f"   AI Response Type: {type(result.get('ai_response'))}")
        print(f"   AI Response Length: {len(str(result.get('ai_response', '')))}")
        print(f"   Natural Response: {result.get('natural_response', 'NOT_FOUND')[:100]}...")
        
        # Check if AI response is being used
        ai_response = result.get('ai_response')
        natural_response = result.get('natural_response')
        
        if ai_response and ai_response != "Model not available for text generation." and ai_response.strip():
            print(f"\n✅ AI Response is available and should be used")
            print(f"   AI Response: {ai_response[:200]}...")
        else:
            print(f"\n❌ AI Response is NOT available or empty")
            print(f"   This is why you're getting rule-based responses")
        
        # Show what the system is actually returning
        print(f"\n🎯 Final Response Analysis:")
        print(f"   What you see: {natural_response[:200]}...")
        print(f"   Is this AI-generated? {'No' if 'Based on your symptoms' in natural_response else 'Possibly'}")
        
        # Global applicability summary
        print(f"\n🌍 Global Advice Summary:")
        print(f"   ✅ Emergency responses direct to hospital/emergency services")
        print(f"   ✅ Non-emergency responses suggest seeing doctor")
        print(f"   ✅ Symptom-specific guidance when applicable")
        print(f"   ✅ Concise, actionable advice (2-3 sentences max)")
        print(f"   ✅ Language suitable for global audience")
        
        return True
        
    except Exception as e:
        print(f"❌ Error debugging Windows RAG system: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_windows_rag()
    sys.exit(0 if success else 1)

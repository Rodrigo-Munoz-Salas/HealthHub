#!/usr/bin/env python3
"""
Test Model Loading on Linux
Verifies that the LLM model loads correctly with the fixed path detection and loading logic
"""


import sys
import logging
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_model_loading():
   """Test model loading with debug information"""
   print("🧪 Testing Model Loading on Linux...")
   print("=" * 60)
  
   try:
       # Import the Windows RAG system (which now has Linux compatibility)
       from windows_rag_system import WindowsRAGSystem
      
       # Initialize with explicit paths for better debugging
       print("🔧 Initializing Windows RAG System with Linux compatibility...")
       rag_system = WindowsRAGSystem(
           models_dir="../mobile_models",
           data_dir="../mobile_rag_ready"
       )
      
       # Get system status
       status = rag_system.get_system_status()
      
       print(f"\n📊 System Status:")
       print(f"   Guidelines: {status['guidelines_loaded']}")
       print(f"   Emergency Protocols: {status['emergency_protocols']}")
       print(f"   LLM Model: {'✅' if status['llm_model_loaded'] else '❌'}")
       print(f"   Embedding Model: {'✅' if status['embedding_model_loaded'] else '❌'}")
       print(f"   Vector Index: {'✅' if status['vector_index_built'] else '❌'}")
       print(f"   Device: {status['device']}")
      
       if status['llm_model_loaded']:
           print("\n✅ Model loading successful!")
           print("🎉 The LLM model is now properly loaded and ready for use!")
          
           # Test a simple query to verify the model works
           print("\n🔍 Testing model with a simple query...")
           test_query = "I have chest pain"
           result = rag_system.query_health_emergency(test_query)
          
           print(f"Query: {test_query}")
           print(f"Response: {result.get('natural_response', 'No response available')}")
           print(f"Emergency Type: {result.get('emergency_type', 'Unknown')}")
           print(f"Call 911: {'YES' if result.get('call_911') else 'NO'}")
          
       else:
           print("\n❌ Model loading failed!")
           print("💡 Check the debug output above for specific error messages")
           return False
          
       return True
      
   except Exception as e:
       print(f"❌ Error during model loading test: {e}")
       import traceback
       traceback.print_exc()
       return False


def test_path_detection():
   """Test path detection logic specifically"""
   print("\n🔍 Testing Path Detection Logic...")
   print("-" * 40)
  
   from pathlib import Path
  
   # Test the same paths that the system checks
   possible_paths = [
       Path("../mobile_models/quantized_tinyllama_health"),
       Path("../../mobile_models/quantized_tinyllama_health"),
       Path("../agent/mobile_models/quantized_tinyllama_health"),
       Path("../../agent/mobile_models/quantized_tinyllama_health"),
       Path("../mobile_models/qwen2_5_0_5b"),
       Path("../../mobile_models/qwen2_5_0_5b"),
       Path("../agent/mobile_models/qwen2_5_0_5b"),
       Path("../../agent/mobile_models/qwen2_5_0_5b"),
   ]
  
   found_paths = []
   for path in possible_paths:
       exists = path.exists()
       print(f"   {path.absolute()}: {'✅' if exists else '❌'}")
       if exists:
           found_paths.append(path)
  
   if found_paths:
       print(f"\n✅ Found {len(found_paths)} model path(s):")
       for path in found_paths:
           print(f"   - {path.absolute()}")
   else:
       print("\n❌ No model paths found!")
       print("💡 Make sure the mobile_models directory exists and contains the model files")
  
   return len(found_paths) > 0


if __name__ == "__main__":
   print("🚀 Linux Model Loading Test")
   print("=" * 60)
  
   # Test path detection first
   path_test_passed = test_path_detection()
  
   if path_test_passed:
       # Test model loading
       model_test_passed = test_model_loading()
      
       if model_test_passed:
           print("\n🎉 All tests passed! The model loading issue has been fixed!")
           sys.exit(0)
       else:
           print("\n❌ Model loading test failed!")
           sys.exit(1)
   else:
       print("\n❌ Path detection test failed!")
       print("💡 Check that the mobile_models directory exists and contains model files")
       sys.exit(1)



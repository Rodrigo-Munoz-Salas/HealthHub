#!/usr/bin/env python3
"""
Debug JSON Error in Model Loading
Identifies exactly where the JSON parsing error occurs
"""


import sys
import logging
import json
from pathlib import Path


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def debug_model_loading():
   """Debug the exact point where JSON error occurs"""
   print("🔍 Debugging JSON Error in Model Loading...")
   print("=" * 60)
  
   try:
       from transformers import AutoTokenizer, AutoModelForCausalLM
       import torch
      
       # Find the model path
       possible_paths = [
           Path("../mobile_models/quantized_tinyllama_health"),
           Path("../../mobile_models/quantized_tinyllama_health"),
           Path("../agent/mobile_models/quantized_tinyllama_health"),
           Path("../../agent/mobile_models/quantized_tinyllama_health"),
       ]
      
       model_path = None
       for path in possible_paths:
           print(f"🔍 Checking: {path.absolute()}")
           if path.exists():
               model_path = path
               print(f"✅ Found model at: {path.absolute()}")
               break
      
       if model_path is None:
           print("❌ No model path found!")
           return False
      
       # Check all JSON files in the model directory
       print(f"\n📋 Checking JSON files in {model_path}:")
       json_files = [
           "config.json",
           "tokenizer_config.json",
           "generation_config.json",
           "special_tokens_map.json"
       ]
      
       for json_file in json_files:
           file_path = model_path / json_file
           if file_path.exists():
               try:
                   with open(file_path, 'r') as f:
                       data = json.load(f)
                   print(f"   ✅ {json_file}: Valid JSON")
               except json.JSONDecodeError as e:
                   print(f"   ❌ {json_file}: JSON Error - {e}")
                   print(f"      File content preview:")
                   with open(file_path, 'r') as f:
                       content = f.read()[:200]
                       print(f"      {content}...")
               except Exception as e:
                   print(f"   ❌ {json_file}: Error - {e}")
           else:
               print(f"   ⚠️ {json_file}: Not found")
      
       # Test tokenizer loading step by step
       print(f"\n🔄 Testing tokenizer loading...")
       try:
           print("   Step 1: Loading tokenizer...")
           tokenizer = AutoTokenizer.from_pretrained(
               str(model_path),
               trust_remote_code=True
           )
           print("   ✅ Tokenizer loaded successfully")
       except Exception as e:
           print(f"   ❌ Tokenizer loading failed: {e}")
           print(f"   Error type: {type(e).__name__}")
           return False
      
       # Test model loading step by step
       print(f"\n🔄 Testing model loading...")
       try:
           print("   Step 1: Loading model with basic parameters...")
           model = AutoModelForCausalLM.from_pretrained(
               str(model_path),
               trust_remote_code=True,
               torch_dtype=torch.float32,
               device_map="cpu"
           )
           print("   ✅ Model loaded successfully")
       except Exception as e:
           print(f"   ❌ Model loading failed: {e}")
           print(f"   Error type: {type(e).__name__}")
          
           # Try with different parameters
           print(f"\n🔄 Trying fallback model loading...")
           try:
               model = AutoModelForCausalLM.from_pretrained(
                   str(model_path),
                   trust_remote_code=True,
                   torch_dtype=torch.float32,
                   device_map="cpu",
                   low_cpu_mem_usage=True
               )
               print("   ✅ Fallback model loading successful")
           except Exception as e2:
               print(f"   ❌ Fallback model loading also failed: {e2}")
               print(f"   Error type: {type(e2).__name__}")
               return False
      
       print("\n✅ All tests passed!")
       return True
      
   except Exception as e:
       print(f"❌ Unexpected error: {e}")
       import traceback
       traceback.print_exc()
       return False


def check_transformers_version():
   """Check transformers library version and compatibility"""
   print("\n📦 Checking Transformers Library...")
   try:
       import transformers
       print(f"   Transformers version: {transformers.__version__}")
      
       # Check if version is compatible
       version_parts = transformers.__version__.split('.')
       major = int(version_parts[0])
       minor = int(version_parts[1])
      
       if major < 4 or (major == 4 and minor < 30):
           print("   ⚠️ Warning: Transformers version might be too old")
           print("   💡 Consider upgrading: pip install transformers>=4.30.0")
       else:
           print("   ✅ Transformers version looks good")
          
   except Exception as e:
       print(f"   ❌ Error checking transformers version: {e}")


if __name__ == "__main__":
   print("🚀 JSON Error Debug Tool")
   print("=" * 60)
  
   check_transformers_version()
   success = debug_model_loading()
  
   if success:
       print("\n🎉 Debug completed successfully!")
       sys.exit(0)
   else:
       print("\n❌ Debug found issues!")
       sys.exit(1)



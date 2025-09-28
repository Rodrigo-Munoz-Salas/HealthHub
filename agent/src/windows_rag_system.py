"""
Windows-Optimized RAG-Anything Implementation
Complete RAG system optimized for Windows with TinyLlama, embeddings, vector database, and RAG-Anything server integration
"""

import json
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import os
import warnings
import requests

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

class RAGAnythingClient:
    """RAG-Anything server client for retrieving additional context"""
    
    def __init__(self, base_url: str = "http://localhost:9999"):
        self.base_url = base_url.rstrip("/")
        self.available = False
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to RAG-Anything server"""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            if resp.status_code == 200:
                self.available = True
                logger.info("✅ RAG-Anything server connected")
            else:
                logger.warning("⚠️ RAG-Anything server responded with non-200 status")
        except Exception as e:
            logger.warning(f"⚠️ RAG-Anything server unavailable: {e}")
            self.available = False
    
    def retrieve(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve context from RAG-Anything server
        
        Args:
            query: Search query
            k: Number of results to retrieve
            
        Returns:
            List of context documents
        """
        if not self.available:
            return []
        
        try:
            resp = requests.post(
                f"{self.base_url}/retrieve", 
                json={"query": query, "k": k}, 
                timeout=10
            )
            resp.raise_for_status()
            data = resp.json()
            
            # Normalize response format
            results = []
            for item in data:
                results.append({
                    "content": item.get("content", ""),
                    "source_id": item.get("source_id", "rag-anything"),
                    "score": item.get("score", 0.0),
                    "title": item.get("title", "RAG-Anything Result"),
                    "source": "rag-anything"
                })
            
            logger.info(f"📚 Retrieved {len(results)} documents from RAG-Anything")
            return results
            
        except Exception as e:
            logger.warning(f"⚠️ Error retrieving from RAG-Anything: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if RAG-Anything server is available"""
        return self.available

class WindowsRAGSystem:
    """Windows-optimized RAG system with TinyLlama, embeddings, and vector search"""
    
    def __init__(self, 
                 models_dir: str = "mobile_models", 
                 data_dir: str = "mobile_rag_ready",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 rag_anything_url: str = "http://localhost:9999"):
        """
        Initialize Windows-optimized RAG system with RAG-Anything integration
        
        Args:
            models_dir: Directory containing TinyLlama model
            data_dir: Directory containing health guidelines and protocols
            embedding_model_name: Sentence transformer model for embeddings
            rag_anything_url: URL of RAG-Anything server
        """
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.embedding_model_name = embedding_model_name
        
        # Initialize components
        self.llm_model = None
        self.llm_tokenizer = None
        self.embedding_model = None
        self.vector_index = None
        self.guidelines = {}
        self.emergency_protocols = {}
        self.doc_ids = []
        
        # Initialize RAG-Anything client
        self.rag_anything_client = RAGAnythingClient(rag_anything_url)
        
        # Windows-specific optimizations
        self._setup_windows_optimizations()
        
        # Load all components
        self._load_health_data()
        self._load_embedding_model()
        self._load_tinyllama_model_windows()
        self._build_vector_index()
        
        logger.info("🏥 Windows RAG System Initialized!")
        logger.info(f"📚 Guidelines: {len(self.guidelines)}")
        logger.info(f"🚨 Emergency Protocols: {len(self.emergency_protocols)}")
        logger.info(f"🤖 TinyLlama Model: {'Loaded' if self.llm_model is not None else 'Not Available'}")
        logger.info(f"🔍 Embedding Model: {'Loaded' if self.embedding_model is not None else 'Not Available'}")
        logger.info(f"📊 Vector Index: {'Built' if self.vector_index is not None else 'Not Available'}")
        logger.info(f"🌐 RAG-Anything Server: {'Connected' if self.rag_anything_client.is_available() else 'Not Available'}")
    
    def _setup_windows_optimizations(self):
        """Setup Windows-specific optimizations"""
        # Set environment variables for better Windows compatibility
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["OMP_NUM_THREADS"] = "4"
        
        # Windows-specific torch settings
        if torch.cuda.is_available():
            logger.info("🚀 CUDA available - using GPU acceleration")
            self.device = "cuda"
        else:
            logger.info("💻 Using CPU mode")
            self.device = "cpu"
    
    def _load_health_data(self):
        """Load health guidelines and emergency protocols"""
        try:
            # Load guidelines
            guidelines_path = self.data_dir / "processed_guidelines.json"
            if guidelines_path.exists():
                with open(guidelines_path, 'r', encoding='utf-8') as f:
                    self.guidelines = json.load(f)
                logger.info(f"✅ Loaded {len(self.guidelines)} guidelines")
            else:
                logger.warning("⚠️ Guidelines file not found")
            
            # Load emergency protocols
            protocols_path = self.data_dir / "emergency_protocols.json"
            if protocols_path.exists():
                with open(protocols_path, 'r', encoding='utf-8') as f:
                    self.emergency_protocols = json.load(f)
                logger.info(f"✅ Loaded {len(self.emergency_protocols)} emergency protocols")
            else:
                logger.warning("⚠️ Emergency protocols file not found")
                
        except Exception as e:
            logger.error(f"❌ Error loading health data: {e}")
    
    def _load_embedding_model(self):
        """Load sentence transformer model for embeddings"""
        try:
            logger.info("🔄 Loading embedding model...")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            logger.info("✅ Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"❌ Error loading embedding model: {e}")
            self.embedding_model = None
    
    def _load_tinyllama_model_windows(self):
        """Load TinyLlama model optimized for Windows"""
        try:
            # Find model path
            possible_paths = [
                self.models_dir / "quantized_tinyllama_health",
                Path("../mobile_models/quantized_tinyllama_health"),
                Path("../../agent/mobile_models/quantized_tinyllama_health"),
                Path("mobile_models/quantized_tinyllama_health")
            ]
            
            model_path = None
            for path in possible_paths:
                if path.exists():
                    model_path = path
                    break
            
            if model_path is None:
                logger.warning("⚠️ TinyLlama model not found, using fallback responses")
                return
            
            logger.info("🔄 Loading TinyLlama model (Windows optimized)...")
            
            # Load tokenizer
            self.llm_tokenizer = AutoTokenizer.from_pretrained(
                str(model_path),
                trust_remote_code=True
            )
            
            # Windows-optimized model loading
            if self.device == "cuda":
                # GPU loading with quantization support
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                )
            else:
                # CPU loading without quantization issues
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    load_in_8bit=False,
                    load_in_4bit=False
                )
            
            # Set pad token
            if self.llm_tokenizer.pad_token is None:
                self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            
            logger.info(f"✅ TinyLlama model loaded successfully ({self.device} mode)")
            
        except Exception as e:
            logger.warning(f"⚠️ TinyLlama model loading failed: {e}")
            logger.info("💡 Using rule-based responses instead of AI-generated text")
            self.llm_model = None
            self.llm_tokenizer = None
    
    def _build_vector_index(self):
        """Build FAISS vector index from health guidelines"""
        try:
            if not self.embedding_model or not self.guidelines:
                logger.warning("⚠️ Cannot build vector index: missing embedding model or guidelines")
                return
            
            logger.info("🔄 Building vector index...")
            
            # Prepare documents
            documents = []
            doc_ids = []
            
            for guideline_id, guideline in self.guidelines.items():
                content = guideline.get("content", "")
                if content:
                    documents.append(content)
                    doc_ids.append(guideline_id)
            
            if not documents:
                logger.warning("⚠️ No documents to index")
                return
            
            # Generate embeddings
            embeddings = self.embedding_model.encode(documents)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.vector_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            self.vector_index.add(embeddings.astype('float32'))
            
            # Store document IDs
            self.doc_ids = doc_ids
            
            logger.info(f"✅ Vector index built with {len(documents)} documents")
            
        except Exception as e:
            logger.error(f"❌ Error building vector index: {e}")
            self.vector_index = None
    
    def _generate_response(self, prompt: str, max_length: int = 200) -> str:
        """Generate response using TinyLlama model"""
        if self.llm_model is None or self.llm_tokenizer is None:
            return "Model not available for text generation."
        
        try:
            # Tokenize input
            inputs = self.llm_tokenizer(
                prompt, 
                return_tensors="pt", 
                truncation=True, 
                max_length=512
            )
            
            # Move to device
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.llm_tokenizer.eos_token_id,
                    eos_token_id=self.llm_tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.llm_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"❌ Error generating response: {e}")
            return "Error generating response."
    
    def _vector_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform vector search on health guidelines"""
        if not self.vector_index or not self.embedding_model:
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            faiss.normalize_L2(query_embedding)
            
            # Search vector index
            scores, indices = self.vector_index.search(query_embedding.astype('float32'), k)
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx < len(self.doc_ids):
                    guideline_id = self.doc_ids[idx]
                    guideline = self.guidelines.get(guideline_id, {})
                    
                    results.append({
                        "guideline_id": guideline_id,
                        "title": guideline.get("title", "Unknown"),
                        "content": guideline.get("content", ""),
                        "score": float(score),
                        "emergency_level": guideline.get("emergency_level", "medium"),
                        "source": "local_vector_db"
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error in vector search: {e}")
            return []
    
    def _hybrid_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid search combining local vector search and RAG-Anything"""
        all_results = []
        
        # 1. Local vector search
        local_results = self._vector_search(query, k)
        all_results.extend(local_results)
        
        # 2. RAG-Anything search
        if self.rag_anything_client.is_available():
            rag_results = self.rag_anything_client.retrieve(query, k)
            all_results.extend(rag_results)
        else:
            logger.info("🌐 RAG-Anything server not available, using local results only")
        
        # 3. Combine and deduplicate results
        combined_results = self._combine_and_rank_results(all_results, query)
        
        return combined_results[:k]
    
    def _combine_and_rank_results(self, results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
        """Combine and rank results from different sources"""
        # Remove duplicates based on content similarity
        unique_results = []
        seen_contents = set()
        
        for result in results:
            content = result.get("content", "")
            # Simple deduplication based on content hash
            content_hash = hash(content[:100])  # Use first 100 chars for deduplication
            
            if content_hash not in seen_contents:
                seen_contents.add(content_hash)
                unique_results.append(result)
        
        # Sort by score (if available) or by source priority
        def sort_key(result):
            score = result.get("score", 0.0)
            source = result.get("source", "unknown")
            
            # Prioritize emergency protocols and local results
            if source == "emergency_protocols":
                return (1.0, score)
            elif source == "local_vector_db":
                return (0.8, score)
            elif source == "rag-anything":
                return (0.6, score)
            else:
                return (0.4, score)
        
        return sorted(unique_results, key=sort_key, reverse=True)
    
    def _detect_emergency_type(self, query: str) -> Optional[str]:
        """Detect emergency type from query"""
        query_lower = query.lower()
        
        emergency_keywords = {
            "chest_pain": ["chest pain", "heart attack", "cardiac", "heart", "chest"],
            "fainting": ["fainted", "fainting", "unconscious", "passed out", "collapsed"],
            "burn": ["burn", "burned", "fire", "hot", "scald", "thermal"],
            "choking": ["choking", "can't breathe", "blocked airway", "suffocating"],
            "stroke": ["stroke", "facial droop", "slurred speech", "weakness", "paralysis"],
            "shortness_breath": ["shortness of breath", "can't breathe", "breathing difficulty", "dyspnea"]
        }
        
        for emergency_type, keywords in emergency_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                return emergency_type
        
        return None
    
    def _create_rag_prompt(self, query: str, context: List[Dict[str, Any]]) -> str:
        """Create comprehensive RAG prompt with context"""
        # Extract relevant context
        context_text = ""
        for i, item in enumerate(context[:3], 1):
            content = item.get("content", "")
            title = item.get("title", "Health Information")
            context_text += f"{i}. {title}: {content[:200]}...\n"
        
        prompt = f"""You are a medical assistant helping with a health emergency. Use the provided context to give accurate, helpful advice.

PATIENT QUERY: "{query}"

RELEVANT MEDICAL CONTEXT:
{context_text}

INSTRUCTIONS:
- Provide a natural, conversational response (2-3 sentences)
- Explain what the symptoms could mean based on the context
- Give clear guidance on what to do
- Be reassuring but clear about urgency
- If it's an emergency, emphasize calling 911

RESPONSE:"""
        
        return prompt
    
    def query_health_emergency(self, query: str) -> Dict[str, Any]:
        """Main query function for health emergencies with RAG"""
        start_time = time.time()
        
        try:
            logger.info(f"🚨 Health Emergency Query: {query[:100]}...")
            
            # Detect emergency type
            emergency_type = self._detect_emergency_type(query)
            
            # Get relevant information
            if emergency_type and emergency_type in self.emergency_protocols:
                # Emergency protocol response
                protocol = self.emergency_protocols[emergency_type]
                
                # Get additional context from vector search
                vector_results = self._vector_search(query, k=3)
                
                # Generate AI response with context
                ai_response = None
                if self.llm_model is not None:
                    prompt = self._create_rag_prompt(query, vector_results)
                    ai_response = self._generate_response(prompt, max_length=200)
                
                response = {
                    "emergency_type": emergency_type,
                    "protocol": protocol,
                    "immediate_actions": protocol.get("immediate_actions", [])[:3],
                    "warning_signs": protocol.get("warning_signs", []),
                    "call_911": protocol.get("call_911", True),
                    "confidence": 0.9,
                    "source": "emergency_protocols",
                    "vector_results": vector_results,
                    "ai_response": ai_response,
                    "processing_time": time.time() - start_time
                }
            else:
                # General health query with hybrid RAG search
                vector_results = self._hybrid_search(query, k=5)
                
                # Generate AI response with context
                ai_response = None
                if self.llm_model is not None:
                    prompt = self._create_rag_prompt(query, vector_results)
                    ai_response = self._generate_response(prompt, max_length=200)
                
                response = {
                    "emergency_type": "general_health",
                    "vector_results": vector_results,
                    "call_911": any(r.get("emergency_level") == "critical" for r in vector_results),
                    "confidence": max([r.get("score", 0) for r in vector_results], default=0.3),
                    "source": "rag_search",
                    "ai_response": ai_response,
                    "processing_time": time.time() - start_time
                }
            
            # Add natural language response
            response["natural_response"] = self._format_natural_response(response, query)
            
            return response
            
        except Exception as e:
            logger.error(f"❌ Error in health emergency query: {e}")
            return {
                "emergency_type": "error",
                "error": str(e),
                "call_911": True,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "natural_response": "I'm unable to process this health emergency. Please call 911 immediately."
            }
    
    def _format_natural_response(self, response: Dict[str, Any], query: str) -> str:
        """Format natural language response"""
        # Use AI response if available
        if response.get("ai_response") and response["ai_response"] != "Model not available for text generation.":
            return response["ai_response"]
        
        # Fallback to rule-based responses
        emergency_type = response.get("emergency_type", "general_health")
        call_911 = response.get("call_911", False)
        
        if emergency_type == "chest_pain":
            if call_911:
                return "Based on your symptoms, this appears to be a potential heart attack or cardiac emergency. Chest pain with breathing difficulties is a serious medical emergency that requires immediate attention. You should call 911 right away and try to stay calm while waiting for help."
            else:
                return "Your chest pain symptoms could indicate several conditions ranging from heartburn to anxiety. However, any chest pain should be taken seriously and evaluated by a healthcare provider. Monitor your symptoms closely and seek medical attention if they worsen."
        
        elif emergency_type == "shortness_breath":
            return "Your symptoms of shortness of breath could indicate several serious conditions including respiratory problems, heart issues, or shock. These symptoms suggest your body may not be getting enough oxygen, which is a medical emergency. You should call 911 immediately and try to stay calm while waiting for help."
        
        elif emergency_type == "fainting":
            if call_911:
                return "Fainting with potential head injury is a serious medical emergency that requires immediate attention. Loss of consciousness can indicate various serious conditions including head trauma, cardiac issues, or neurological problems. Call 911 immediately and while waiting, check if the person is breathing."
            else:
                return "Fainting episodes can have various causes including dehydration, low blood pressure, or stress. However, any loss of consciousness should be evaluated by a healthcare provider to rule out serious conditions. Monitor the person closely and seek medical attention if symptoms persist."
        
        elif emergency_type == "choking":
            return "Choking is a life-threatening emergency that requires immediate action. When someone cannot speak or breathe due to a blocked airway, every second counts. Call 911 immediately and perform the Heimlich maneuver if you're trained to do so, or encourage the person to cough forcefully."
        
        elif emergency_type == "stroke":
            return "Facial drooping is a classic sign of stroke, which is a medical emergency that requires immediate treatment. Time is critical with strokes - the sooner treatment begins, the better the outcome. Call 911 immediately and note the time when symptoms started, as this information is crucial for treatment decisions."
        
        else:
            # General health response with context
            vector_results = response.get("vector_results", [])
            if vector_results:
                best_result = vector_results[0]
                content = best_result.get("content", "")
                if len(content) > 200:
                    content = content[:200] + "..."
                return f"Based on your symptoms, here's relevant health information: {content}"
            else:
                return "Your symptoms require medical attention and should be evaluated by a healthcare provider. While I cannot provide a specific diagnosis, it's important to take your symptoms seriously. If you're experiencing severe or worsening symptoms, call 911 or seek immediate medical care."
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "guidelines_loaded": len(self.guidelines),
            "emergency_protocols": len(self.emergency_protocols),
            "llm_model_loaded": self.llm_model is not None,
            "embedding_model_loaded": self.embedding_model is not None,
            "vector_index_built": self.vector_index is not None,
            "rag_anything_available": self.rag_anything_client.is_available(),
            "rag_anything_url": self.rag_anything_client.base_url,
            "device": self.device,
            "system_ready": True
        }


def test_windows_rag_system():
    """Test the Windows RAG system"""
    print("🧪 Testing Windows RAG System")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    try:
        # Initialize the Windows RAG system
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
        print(f"   RAG-Anything Server: {'✅' if status['rag_anything_available'] else '❌'} ({status['rag_anything_url']})")
        print(f"   Device: {status['device']}")
        
        # Test queries
        test_queries = [
            "I have severe chest pain and can't breathe properly",
            "Someone just fainted and hit their head",
            "A person is choking on food and can't speak",
            "My neighbor is showing signs of stroke with facial drooping",
            "I have shortness of breath, pale skin, and cold skin"
        ]
        
        print(f"\n🚨 Testing Health Emergency Queries:")
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: {query}")
            print("-" * 50)
            
            result = rag_system.query_health_emergency(query)
            
            print(f"🤖 NATURAL RESPONSE:")
            print(f"   {result.get('natural_response', 'No response available')}")
            
            print(f"\n📋 TECHNICAL ANALYSIS:")
            print(f"   Emergency Type: {result.get('emergency_type', 'Unknown').replace('_', ' ').title()}")
            print(f"   Call 911: {'YES - IMMEDIATELY' if result.get('call_911') else 'NO - Monitor situation'}")
            print(f"   Confidence: {result.get('confidence', 0.0):.1%}")
            print(f"   Processing Time: {result.get('processing_time', 0.0):.2f}s")
            
            # Show immediate actions if available
            immediate_actions = result.get('immediate_actions', [])
            if immediate_actions:
                print(f"\n⚡ IMMEDIATE ACTIONS:")
                for j, action in enumerate(immediate_actions, 1):
                    print(f"   {j}. {action}")
            
            # Show vector search results if available
            vector_results = result.get('vector_results', [])
            if vector_results:
                print(f"\n📚 RELEVANT HEALTH INFORMATION:")
                for j, result_item in enumerate(vector_results[:2], 1):
                    content = result_item.get('content', '')
                    if len(content) > 150:
                        content = content[:150] + "..."
                    print(f"   {j}. {content}")
                    print(f"      Source: {result_item.get('title', 'Health Guidelines')}")
        
        print(f"\n✅ Windows RAG System Test Completed!")
        print(f"💡 The system is ready for health emergency assistance on Windows!")
        
    except Exception as e:
        print(f"❌ Error testing Windows RAG system: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_windows_rag_system()

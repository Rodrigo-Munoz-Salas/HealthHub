"""
Mobile RAG Test Server
FastAPI server for testing mobile RAG system with prediction and rag_client integration
"""

import json
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import our mobile RAG client
from mobile_rag_client import MobileRAGClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Mobile RAG Test Server",
    description="Test server for mobile health emergency RAG system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize mobile RAG client
mobile_rag_client = None

@app.on_event("startup")
async def startup_event():
    """Initialize the mobile RAG client on startup"""
    global mobile_rag_client
    try:
        mobile_rag_client = MobileRAGClient()
        logger.info("🚀 Mobile RAG Test Server started successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize mobile RAG client: {e}")
        raise

# Pydantic models for request/response
class EmergencyQuery(BaseModel):
    query: str = Field(..., description="Health emergency query", min_length=1)
    user_profile: Optional[Dict[str, Any]] = Field(None, description="Optional user profile information")

class VectorSearchQuery(BaseModel):
    query: str = Field(..., description="Search query", min_length=1)
    k: int = Field(5, description="Number of results to return", ge=1, le=20)

class TestResponse(BaseModel):
    success: bool
    data: Dict[str, Any]
    processing_time: float
    timestamp: float

# API Endpoints

@app.get("/")
async def root():
    """Root endpoint with basic information"""
    return {
        "message": "Mobile RAG Test Server",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "test_emergency": "/test/emergency",
            "test_vector_search": "/test/vector-search",
            "test_protocols": "/test/protocols",
            "comprehensive_test": "/test/comprehensive"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if mobile_rag_client is None:
        raise HTTPException(status_code=503, detail="Mobile RAG client not initialized")
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "mobile_rag_ready": mobile_rag_client is not None
    }

@app.get("/status")
async def get_system_status():
    """Get comprehensive system status"""
    if mobile_rag_client is None:
        raise HTTPException(status_code=503, detail="Mobile RAG client not initialized")
    
    try:
        status = mobile_rag_client.get_system_status()
        return TestResponse(
            success=True,
            data=status,
            processing_time=0.0,
            timestamp=time.time()
        )
    except Exception as e:
        logger.error(f"❌ Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/emergency")
async def test_emergency_query(request: EmergencyQuery):
    """Test emergency query processing"""
    if mobile_rag_client is None:
        raise HTTPException(status_code=503, detail="Mobile RAG client not initialized")
    
    try:
        start_time = time.time()
        result = mobile_rag_client.test_emergency_query(
            query=request.query,
            user_profile=request.user_profile
        )
        processing_time = time.time() - start_time
        
        return TestResponse(
            success=result.get("success", True),
            data=result,
            processing_time=processing_time,
            timestamp=time.time()
        )
    except Exception as e:
        logger.error(f"❌ Error testing emergency query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test/vector-search")
async def test_vector_search(request: VectorSearchQuery):
    """Test vector search capabilities"""
    if mobile_rag_client is None:
        raise HTTPException(status_code=503, detail="Mobile RAG client not initialized")
    
    try:
        start_time = time.time()
        result = mobile_rag_client.test_vector_search(
            query=request.query,
            k=request.k
        )
        processing_time = time.time() - start_time
        
        return TestResponse(
            success=result.get("success", True),
            data=result,
            processing_time=processing_time,
            timestamp=time.time()
        )
    except Exception as e:
        logger.error(f"❌ Error testing vector search: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test/protocols")
async def test_emergency_protocols():
    """Test emergency protocols functionality"""
    if mobile_rag_client is None:
        raise HTTPException(status_code=503, detail="Mobile RAG client not initialized")
    
    try:
        start_time = time.time()
        result = mobile_rag_client.test_emergency_protocols()
        processing_time = time.time() - start_time
        
        return TestResponse(
            success=result.get("success", True),
            data=result,
            processing_time=processing_time,
            timestamp=time.time()
        )
    except Exception as e:
        logger.error(f"❌ Error testing emergency protocols: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test/comprehensive")
async def run_comprehensive_test():
    """Run comprehensive test of all capabilities"""
    if mobile_rag_client is None:
        raise HTTPException(status_code=503, detail="Mobile RAG client not initialized")
    
    try:
        start_time = time.time()
        result = mobile_rag_client.run_comprehensive_test()
        processing_time = time.time() - start_time
        
        return TestResponse(
            success=True,
            data=result,
            processing_time=processing_time,
            timestamp=time.time()
        )
    except Exception as e:
        logger.error(f"❌ Error running comprehensive test: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/test/sample-queries")
async def get_sample_queries():
    """Get sample emergency queries for testing"""
    return {
        "sample_queries": [
            {
                "query": "Someone is having severe chest pain and can't breathe",
                "expected_type": "chest_pain",
                "emergency_level": "critical"
            },
            {
                "query": "A person fainted and is unconscious",
                "expected_type": "fainting",
                "emergency_level": "critical"
            },
            {
                "query": "There's a severe burn on someone's arm",
                "expected_type": "burn",
                "emergency_level": "high"
            },
            {
                "query": "Someone is choking and can't speak",
                "expected_type": "choking",
                "emergency_level": "critical"
            },
            {
                "query": "Person showing signs of stroke with facial droop",
                "expected_type": "stroke",
                "emergency_level": "critical"
            },
            {
                "query": "Someone has a high fever and is very weak",
                "expected_type": "general_health",
                "emergency_level": "medium"
            }
        ],
        "usage": "Use these queries with the /test/emergency endpoint"
    }

# Background task endpoints
@app.post("/test/batch-emergency")
async def test_batch_emergency_queries(background_tasks: BackgroundTasks):
    """Test multiple emergency queries in background"""
    if mobile_rag_client is None:
        raise HTTPException(status_code=503, detail="Mobile RAG client not initialized")
    
    sample_queries = [
        "Someone is having severe chest pain and can't breathe",
        "A person fainted and is unconscious",
        "There's a severe burn on someone's arm",
        "Someone is choking and can't speak",
        "Person showing signs of stroke with facial droop"
    ]
    
    def run_batch_tests():
        results = []
        for query in sample_queries:
            try:
                result = mobile_rag_client.test_emergency_query(query)
                results.append(result)
            except Exception as e:
                results.append({"query": query, "error": str(e), "success": False})
        
        # Save results to file
        with open("batch_test_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"📊 Batch test completed: {len(results)} queries processed")
    
    background_tasks.add_task(run_batch_tests)
    
    return {
        "message": "Batch emergency query test started",
        "queries_count": len(sample_queries),
        "status": "running_in_background"
    }

if __name__ == "__main__":
    print("🏥 Starting Mobile RAG Test Server...")
    print("=" * 50)
    print("Available endpoints:")
    print("  GET  / - Root endpoint")
    print("  GET  /health - Health check")
    print("  GET  /status - System status")
    print("  POST /test/emergency - Test emergency query")
    print("  POST /test/vector-search - Test vector search")
    print("  GET  /test/protocols - Test emergency protocols")
    print("  GET  /test/comprehensive - Run comprehensive test")
    print("  GET  /test/sample-queries - Get sample queries")
    print("  POST /test/batch-emergency - Test batch queries")
    print()
    print("Server will start on http://localhost:8000")
    print("API documentation available at http://localhost:8000/docs")
    print()
    
    uvicorn.run(
        "mobile_test_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# Mobile RAG Testing System

This system provides a comprehensive testing framework for the mobile health emergency RAG system, integrating with your existing `prediction.py` and `rag_client.py` files.

## Overview

The mobile RAG testing system allows you to:
- Test health emergency queries with the mobile RAG system
- Integrate with your existing RAG-Anything server
- Simulate mobile device behavior
- Test vector search capabilities
- Run comprehensive system tests

## Files Created

### Core Components
- `mobile_rag_client.py` - Main client that integrates prediction and RAG capabilities
- `mobile_test_server.py` - FastAPI server with testing endpoints
- `test_mobile_rag.py` - Test script for direct and HTTP testing
- `start_mobile_test.py` - Startup script for the test server

## Quick Start

### 1. Start the Test Server
```bash
cd /Users/darkknight/Desktop/HealthHub/agent/src
python start_mobile_test.py
```

### 2. Test the System
```bash
# Run comprehensive tests
python test_mobile_rag.py

# Or test specific endpoints via HTTP
curl -X POST http://localhost:8000/test/emergency \
  -H 'Content-Type: application/json' \
  -d '{"query": "Someone is having chest pain"}'
```

## API Endpoints

### Health & Status
- `GET /health` - Health check
- `GET /status` - System status and configuration

### Testing Endpoints
- `POST /test/emergency` - Test emergency query processing
- `POST /test/vector-search` - Test vector search capabilities
- `GET /test/protocols` - Test emergency protocols
- `GET /test/comprehensive` - Run comprehensive system test
- `GET /test/sample-queries` - Get sample queries for testing

### Background Tasks
- `POST /test/batch-emergency` - Test multiple queries in background

## Usage Examples

### Test Emergency Query
```python
import requests

# Test emergency query
response = requests.post("http://localhost:8000/test/emergency", json={
    "query": "Someone is having severe chest pain and can't breathe",
    "user_profile": {
        "name": "Test User",
        "age": 30,
        "conditions": ["None"]
    }
})

print(response.json())
```

### Test Vector Search
```python
# Test vector search
response = requests.post("http://localhost:8000/test/vector-search", json={
    "query": "heart attack symptoms",
    "k": 5
})

print(response.json())
```

### Direct Testing (No Server)
```python
from mobile_rag_client import MobileRAGClient

# Initialize client
client = MobileRAGClient()

# Test emergency query
result = client.test_emergency_query("Someone is having chest pain")
print(result)
```

## Sample Queries

The system comes with pre-configured sample queries for testing:

1. **Chest Pain**: "Someone is having severe chest pain and can't breathe"
2. **Fainting**: "A person fainted and is unconscious"
3. **Burns**: "There's a severe burn on someone's arm"
4. **Choking**: "Someone is choking and can't speak"
5. **Stroke**: "Person showing signs of stroke with facial droop"
6. **General Health**: "Someone has a high fever and is very weak"

## System Integration

### Mobile RAG System
- Uses pre-built guidelines from `mobile_rag_ready/processed_guidelines.json`
- Loads emergency protocols from `mobile_rag_ready/emergency_protocols.json`
- Accesses vector database from `mobile_rag_ready/vector_database.pkl`

### RAG-Anything Server
- Connects to RAG-Anything server (default: http://localhost:9999)
- Falls back gracefully if server is unavailable
- Integrates with your existing `rag_client.py`

### Prediction Integration
- Uses your existing prediction logic from `prediction.py`
- Maintains compatibility with your model structure
- Supports both local and remote model inference

## Configuration

### Environment Variables
- `RAG_ANYTHING_URL` - RAG-Anything server URL (default: http://localhost:9999)
- `MOBILE_RAG_DIR` - Mobile RAG data directory (default: mobile_rag_ready)

### Server Configuration
- Host: 0.0.0.0 (all interfaces)
- Port: 8000
- CORS: Enabled for all origins
- Logging: INFO level

## Testing Workflow

1. **Start Server**: `python start_mobile_test.py`
2. **Check Health**: `curl http://localhost:8000/health`
3. **Test Emergency**: Use `/test/emergency` endpoint
4. **Test Vector Search**: Use `/test/vector-search` endpoint
5. **Run Comprehensive Test**: Use `/test/comprehensive` endpoint
6. **Review Results**: Check response data and processing times

## Response Format

All endpoints return a consistent response format:

```json
{
  "success": true,
  "data": {
    "query": "test query",
    "mobile_rag_response": {...},
    "rag_context": [...],
    "processing_time": 0.123,
    "timestamp": 1234567890.123
  },
  "processing_time": 0.123,
  "timestamp": 1234567890.123
}
```

## Troubleshooting

### Common Issues

1. **Server won't start**: Check if port 8000 is available
2. **Mobile RAG not loading**: Verify `mobile_rag_ready` directory exists
3. **RAG-Anything connection failed**: Check if RAG server is running on port 9999
4. **Import errors**: Ensure all dependencies are installed

### Debug Mode
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python start_mobile_test.py
```

## Performance Metrics

The system tracks:
- Query processing time
- Vector search performance
- Emergency detection accuracy
- RAG server response times
- System resource usage

## Next Steps

1. **Customize Queries**: Add your own test queries
2. **Extend Endpoints**: Add new testing capabilities
3. **Performance Tuning**: Optimize for your use case
4. **Integration**: Connect with your existing systems

## Support

For issues or questions:
1. Check the logs for error messages
2. Verify all required files are present
3. Test individual components separately
4. Review the API documentation at http://localhost:8000/docs

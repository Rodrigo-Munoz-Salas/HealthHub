# Windows RAG System

A complete RAG (Retrieval-Augmented Generation) system optimized for Windows, featuring TinyLlama model, local embeddings, and vector database for health emergency assistance.

## 🚀 Features

- **Windows-Optimized**: Specifically designed for Windows systems
- **TinyLlama Integration**: Local AI model for text generation
- **Vector Search**: FAISS-based similarity search
- **Health Guidelines**: 55+ health emergency protocols
- **Emergency Detection**: Automatic emergency type classification
- **GPU Support**: CUDA acceleration when available
- **Fallback Responses**: Rule-based responses when model unavailable

## 📋 Requirements

- **Windows 10/11**
- **Python 3.8+**
- **8GB+ RAM** (recommended)
- **NVIDIA GPU** (optional, for CUDA acceleration)

## 🛠️ Installation

### Quick Setup

```bash
# 1. Clone the repository
git clone <your-repo>
cd HealthHub/agent

# 2. Run the Windows setup script
python setup_windows.py
```

### Manual Setup

```bash
# 1. Install requirements
pip install -r requirements_windows.txt

# 2. Test the system
python windows_rag_system.py
```

## 🎯 Usage

### Basic Usage

```bash
# Test with a health emergency query
python test_windows_prompt.py "I have severe chest pain and shortness of breath"
```

### Example Queries

```bash
# Heart attack symptoms
python test_windows_prompt.py "I have severe chest pain and can't breathe"

# Choking emergency
python test_windows_prompt.py "Someone is choking on food and can't speak"

# Stroke symptoms
python test_windows_prompt.py "My neighbor is showing signs of stroke"

# General health concern
python test_windows_prompt.py "I have a headache and feel dizzy"
```

### Programmatic Usage

```python
from windows_rag_system import WindowsRAGSystem

# Initialize the system
rag_system = WindowsRAGSystem()

# Query the system
result = rag_system.query_health_emergency("I have chest pain")

# Get response
print(result['natural_response'])
print(f"Call 911: {result['call_911']}")
```

## 🔧 System Components

### 1. TinyLlama Model
- **Purpose**: Local text generation
- **Optimization**: Windows-specific loading
- **Fallback**: Rule-based responses when unavailable

### 2. Vector Database
- **Technology**: FAISS (Facebook AI Similarity Search)
- **Content**: 55 health guidelines
- **Search**: Semantic similarity matching

### 3. Embeddings
- **Model**: all-MiniLM-L6-v2
- **Purpose**: Convert text to vectors for search
- **Performance**: Fast and accurate

### 4. Emergency Protocols
- **Count**: 5 emergency protocols
- **Types**: Chest pain, choking, stroke, fainting, burns
- **Features**: Immediate actions, warning signs, 911 recommendations

## 📊 System Status

The system provides detailed status information:

```python
status = rag_system.get_system_status()
print(f"Guidelines: {status['guidelines_loaded']}")
print(f"Emergency Protocols: {status['emergency_protocols']}")
print(f"LLM Model: {status['llm_model_loaded']}")
print(f"Device: {status['device']}")  # 'cuda' or 'cpu'
```

## 🚨 Emergency Detection

The system automatically detects emergency types:

- **Chest Pain**: Heart attack, cardiac issues
- **Choking**: Airway obstruction
- **Stroke**: Neurological emergency
- **Fainting**: Loss of consciousness
- **Burns**: Thermal injuries
- **Shortness of Breath**: Respiratory emergencies

## 🔍 RAG Pipeline

1. **Query Processing**: Analyze the health emergency query
2. **Emergency Detection**: Classify emergency type
3. **Vector Search**: Find relevant health information
4. **Context Retrieval**: Get additional context from guidelines
5. **AI Generation**: Generate response using TinyLlama
6. **Response Formatting**: Create natural language response

## ⚡ Performance Optimizations

### Windows-Specific Optimizations

- **Environment Variables**: Optimized for Windows
- **Threading**: Configured for Windows threading
- **Memory Management**: Efficient memory usage
- **CUDA Support**: Automatic GPU detection

### Device Selection

```python
# Automatic device selection
if torch.cuda.is_available():
    device = "cuda"  # GPU acceleration
else:
    device = "cpu"   # CPU mode
```

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```bash
   # Solution: Check model files exist
   ls mobile_models/quantized_tinyllama_health/
   ```

2. **CUDA Issues**
   ```bash
   # Solution: Install CPU-only version
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
   ```

3. **Memory Issues**
   ```bash
   # Solution: Reduce batch size or use CPU mode
   export CUDA_VISIBLE_DEVICES=""
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed logging
rag_system = WindowsRAGSystem()
```

## 📈 Performance Benchmarks

| Component | CPU Mode | GPU Mode |
|-----------|----------|----------|
| Model Loading | 5-10s | 3-5s |
| Query Processing | 0.5-2s | 0.2-0.5s |
| Vector Search | 0.1-0.3s | 0.1-0.3s |
| Memory Usage | 2-4GB | 4-6GB |

## 🔒 Security & Privacy

- **Local Processing**: All data processed locally
- **No External Calls**: No data sent to external servers
- **Privacy First**: Health data stays on your machine
- **Offline Capable**: Works without internet connection

## 📚 API Reference

### WindowsRAGSystem Class

```python
class WindowsRAGSystem:
    def __init__(self, models_dir, data_dir, embedding_model_name)
    def query_health_emergency(self, query: str) -> Dict[str, Any]
    def get_system_status(self) -> Dict[str, Any]
```

### Response Format

```python
{
    "emergency_type": "chest_pain",
    "natural_response": "AI-generated response...",
    "call_911": True,
    "confidence": 0.9,
    "immediate_actions": ["Call 911", "Sit down", "Loosen clothing"],
    "warning_signs": ["Shortness of breath", "Nausea"],
    "vector_results": [...],
    "processing_time": 0.5
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test on Windows
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review the system status
3. Check Windows compatibility
4. Verify Python version

---

**Note**: This system is optimized for Windows and may not work correctly on macOS or Linux systems.

# 📱 Mobile Health Emergency RAG System

## 🚀 Quick Start

1. **Copy files to mobile device:**
   - Copy `mobile_rag_ready/` folder
   - Copy `mobile_models/` folder

2. **Install dependencies on mobile:**
   ```bash
   pip install torch>=2.2.0
   pip install transformers>=4.36.0
   pip install sentence-transformers>=2.2.2
   pip install numpy>=1.24.3
   ```

3. **Run the system:**
   ```bash
   python mobile_health_rag.py
   ```

## 🏥 Usage

```python
from mobile_health_rag import MobileHealthRAG

# Initialize system
mobile_rag = MobileHealthRAG()

# Query health emergency
response = mobile_rag.query_emergency(
    "Someone is having chest pain and shortness of breath"
)

print(f"Emergency Type: {response['emergency_type']}")
print(f"Immediate Actions: {response['immediate_actions']}")
print(f"Call 911: {response['call_911']}")
```

## 🚨 Emergency Protocols

- **Chest Pain**: Heart attack protocol
- **Fainting**: Unconsciousness protocol
- **Severe Burns**: Burn treatment protocol
- **Choking**: Airway obstruction protocol
- **Stroke**: Stroke recognition protocol

## 📊 Performance

- **Startup Time**: < 5 seconds
- **Query Time**: < 1 second
- **Memory Usage**: < 500MB
- **Storage Required**: ~700MB

## 🎯 Features

- ✅ Pre-built emergency protocols
- ✅ Pre-processed health guidelines
- ✅ Pre-computed vector database
- ✅ Instant emergency responses
- ✅ Offline capable
- ✅ Mobile optimized

**Ready for life-saving health emergency assistance!** 🏥📱

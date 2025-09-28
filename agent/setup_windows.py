"""
Windows Setup Script for RAG System
Automatically sets up the Windows-optimized RAG system
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def check_windows():
    """Check if running on Windows"""
    if platform.system() != "Windows":
        print("❌ This script is designed for Windows systems only.")
        print(f"   Current system: {platform.system()}")
        return False
    return True

def check_python_version():
    """Check Python version"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required. Current version: {version.major}.{version.minor}")
        return False
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """Install Windows-optimized requirements"""
    print("📦 Installing Windows-optimized requirements...")
    
    try:
        # Install requirements
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", "requirements_windows.txt"
        ])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_cuda_support():
    """Check for CUDA support"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"🚀 CUDA available! GPU count: {torch.cuda.device_count()}")
            print(f"   Current device: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("💻 CUDA not available, using CPU mode")
            return False
    except ImportError:
        print("⚠️ PyTorch not installed yet")
        return False

def test_system():
    """Test the Windows RAG system"""
    print("🧪 Testing Windows RAG System...")
    
    try:
        from windows_rag_system import WindowsRAGSystem
        
        # Initialize system
        rag_system = WindowsRAGSystem()
        
        # Check status
        status = rag_system.get_system_status()
        
        print("\n📊 SYSTEM STATUS:")
        print(f"   Guidelines: {status['guidelines_loaded']}")
        print(f"   Emergency Protocols: {status['emergency_protocols']}")
        print(f"   LLM Model: {'✅' if status['llm_model_loaded'] else '❌'}")
        print(f"   Embedding Model: {'✅' if status['embedding_model_loaded'] else '❌'}")
        print(f"   Vector Index: {'✅' if status['vector_index_built'] else '❌'}")
        print(f"   Device: {status['device']}")
        
        # Test a simple query
        print("\n🔍 Testing with sample query...")
        result = rag_system.query_health_emergency("I have chest pain")
        
        if result.get('natural_response'):
            print("✅ System working correctly!")
            return True
        else:
            print("❌ System test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing system: {e}")
        return False

def main():
    """Main setup function"""
    print("🏥 Windows RAG System Setup")
    print("=" * 50)
    
    # Check system requirements
    if not check_windows():
        return
    
    if not check_python_version():
        return
    
    print("✅ System requirements met!")
    print()
    
    # Install requirements
    if not install_requirements():
        return
    
    print()
    
    # Check CUDA support
    check_cuda_support()
    print()
    
    # Test system
    if test_system():
        print("\n🎉 Windows RAG System setup complete!")
        print("\n💡 Usage:")
        print("   python test_windows_prompt.py \"Your health emergency query\"")
        print("\n📚 Example:")
        print("   python test_windows_prompt.py \"I have severe chest pain\"")
    else:
        print("\n❌ Setup incomplete. Please check the errors above.")

if __name__ == "__main__":
    main()

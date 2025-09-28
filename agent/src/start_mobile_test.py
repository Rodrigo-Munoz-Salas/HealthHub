#!/usr/bin/env python3
"""
Start Mobile RAG Test Server
Simple script to start the mobile RAG test server with proper configuration
"""

import os
import sys
import subprocess
from pathlib import Path

def check_requirements():
    """Check if required files and directories exist"""
    print("🔍 Checking requirements...")
    
    # Look for mobile_rag_ready directory in multiple possible locations
    possible_paths = [
        Path("mobile_rag_ready"),  # Current directory
        Path("../mobile_rag_ready"),  # Parent directory
        Path("../../mobile_rag_ready"),  # Two levels up
        Path("../../agent/mobile_rag_ready"),  # Agent subdirectory
    ]
    
    mobile_rag_dir = None
    for path in possible_paths:
        if path.exists():
            mobile_rag_dir = path
            break
    
    if not mobile_rag_dir:
        print("❌ mobile_rag_ready directory not found")
        print("💡 Searched in:")
        for path in possible_paths:
            print(f"   - {path.absolute()}")
        return False
    
    print(f"✅ Found mobile_rag_ready at: {mobile_rag_dir.absolute()}")
    
    # Check for required files
    required_files = [
        mobile_rag_dir / "processed_guidelines.json",
        mobile_rag_dir / "emergency_protocols.json",
        mobile_rag_dir / "vector_database.pkl"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing required files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        return False
    
    print("✅ All requirements met")
    return True

def start_server():
    """Start the mobile RAG test server"""
    print("🚀 Starting Mobile RAG Test Server...")
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    server_script = script_dir / "mobile_test_server.py"
    
    if not server_script.exists():
        print(f"❌ Server script not found: {server_script}")
        return
    
    try:
        # Start the server
        subprocess.run([
            sys.executable, str(server_script)
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting server: {e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

def main():
    """Main function"""
    print("🏥 Mobile RAG Test Server Startup")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n💡 Please ensure all required files are present before starting the server")
        return
    
    print("\n📋 Server Information:")
    print("   - Server URL: http://localhost:8000")
    print("   - API Documentation: http://localhost:8000/docs")
    print("   - Health Check: http://localhost:8000/health")
    print("   - System Status: http://localhost:8000/status")
    print()
    print("📋 Available Test Endpoints:")
    print("   - POST /test/emergency - Test emergency queries")
    print("   - POST /test/vector-search - Test vector search")
    print("   - GET  /test/protocols - Test emergency protocols")
    print("   - GET  /test/comprehensive - Run comprehensive test")
    print("   - GET  /test/sample-queries - Get sample queries")
    print()
    print("💡 To test the system, run: python test_mobile_rag.py")
    print()
    
    # Start server
    start_server()

if __name__ == "__main__":
    main()

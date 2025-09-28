"""
Test Script for Mobile RAG System
Demonstrates how to test prompts with the mobile RAG system
"""

import json
import time
import requests
from typing import Dict, List, Any

def test_mobile_rag_direct():
    """Test mobile RAG system directly (without server)"""
    print("🏥 Testing Mobile RAG System Directly")
    print("=" * 50)
    
    try:
        from mobile_rag_client import MobileRAGClient
        
        # Initialize client
        client = MobileRAGClient()
        
        # Test emergency queries
        test_queries = [
            "Someone is having severe chest pain and can't breathe",
            "A person fainted and is unconscious", 
            "There's a severe burn on someone's arm",
            "Someone is choking and can't speak",
            "Person showing signs of stroke with facial droop"
        ]
        
        print("\n🚨 Testing Emergency Queries:")
        for i, query in enumerate(test_queries, 1):
            print(f"\n{i}. Query: {query}")
            result = client.test_emergency_query(query)
            
            if result.get("success"):
                mobile_response = result.get("mobile_rag_response", {})
                print(f"   Emergency Type: {mobile_response.get('emergency_type', 'unknown')}")
                print(f"   Call 911: {mobile_response.get('call_911', False)}")
                print(f"   Confidence: {mobile_response.get('confidence', 0.0):.2f}")
                print(f"   Processing Time: {result.get('processing_time', 0.0):.2f}s")
                
                if mobile_response.get('immediate_actions'):
                    print(f"   Immediate Actions: {mobile_response['immediate_actions'][:2]}")
            else:
                print(f"   ❌ Error: {result.get('error', 'Unknown error')}")
        
        # Test vector search
        print("\n🔍 Testing Vector Search:")
        search_result = client.test_vector_search("heart attack symptoms", k=3)
        if search_result.get("success"):
            print(f"   Query: {search_result.get('query')}")
            print(f"   Mobile RAG Results: {len(search_result.get('mobile_rag_results', []))}")
            print(f"   RAG Server Results: {len(search_result.get('rag_server_results', []))}")
            print(f"   Processing Time: {search_result.get('processing_time', 0.0):.2f}s")
        else:
            print(f"   ❌ Error: {search_result.get('error', 'Unknown error')}")
        
        # Get system status
        print("\n📊 System Status:")
        status = client.get_system_status()
        print(f"   Mobile RAG Ready: {status.get('system_ready', False)}")
        print(f"   Guidelines Loaded: {status.get('mobile_rag_status', {}).get('guidelines_loaded', 0)}")
        print(f"   Emergency Protocols: {status.get('mobile_rag_status', {}).get('emergency_protocols', 0)}")
        print(f"   RAG Server Status: {status.get('rag_server_status', 'unknown')}")
        
    except Exception as e:
        print(f"❌ Error testing mobile RAG system: {e}")

def test_mobile_rag_server():
    """Test mobile RAG system via HTTP server"""
    print("\n🌐 Testing Mobile RAG System via HTTP Server")
    print("=" * 50)
    
    base_url = "http://localhost:8000"
    
    # Test health check
    try:
        response = requests.get(f"{base_url}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Server is healthy")
        else:
            print(f"❌ Server health check failed: {response.status_code}")
            return
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("💡 Make sure to start the server first: python mobile_test_server.py")
        return
    
    # Test system status
    try:
        response = requests.get(f"{base_url}/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("📊 System Status:")
            print(f"   Success: {data.get('success', False)}")
            print(f"   Processing Time: {data.get('processing_time', 0.0):.2f}s")
            
            status_data = data.get('data', {})
            mobile_status = status_data.get('mobile_rag_status', {})
            print(f"   Guidelines: {mobile_status.get('guidelines_loaded', 0)}")
            print(f"   Emergency Protocols: {mobile_status.get('emergency_protocols', 0)}")
            print(f"   RAG Server: {status_data.get('rag_server_status', 'unknown')}")
        else:
            print(f"❌ Status check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking status: {e}")
    
    # Test emergency query
    try:
        test_query = {
            "query": "Someone is having severe chest pain and can't breathe",
            "user_profile": {
                "name": "Test User",
                "age": 30,
                "conditions": ["None"]
            }
        }
        
        response = requests.post(f"{base_url}/test/emergency", json=test_query, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("\n🚨 Emergency Query Test:")
            print(f"   Success: {data.get('success', False)}")
            print(f"   Processing Time: {data.get('processing_time', 0.0):.2f}s")
            
            result_data = data.get('data', {})
            mobile_response = result_data.get('mobile_rag_response', {})
            print(f"   Emergency Type: {mobile_response.get('emergency_type', 'unknown')}")
            print(f"   Call 911: {mobile_response.get('call_911', False)}")
            print(f"   Confidence: {mobile_response.get('confidence', 0.0):.2f}")
        else:
            print(f"❌ Emergency query test failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing emergency query: {e}")
    
    # Test vector search
    try:
        search_query = {
            "query": "heart attack symptoms",
            "k": 3
        }
        
        response = requests.post(f"{base_url}/test/vector-search", json=search_query, timeout=10)
        if response.status_code == 200:
            data = response.json()
            print("\n🔍 Vector Search Test:")
            print(f"   Success: {data.get('success', False)}")
            print(f"   Processing Time: {data.get('processing_time', 0.0):.2f}s")
            
            result_data = data.get('data', {})
            print(f"   Mobile RAG Results: {len(result_data.get('mobile_rag_results', []))}")
            print(f"   RAG Server Results: {len(result_data.get('rag_server_results', []))}")
        else:
            print(f"❌ Vector search test failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error testing vector search: {e}")
    
    # Test comprehensive test
    try:
        response = requests.get(f"{base_url}/test/comprehensive", timeout=30)
        if response.status_code == 200:
            data = response.json()
            print("\n🧪 Comprehensive Test:")
            print(f"   Success: {data.get('success', False)}")
            print(f"   Processing Time: {data.get('processing_time', 0.0):.2f}s")
            
            result_data = data.get('data', {})
            test_queries = result_data.get('test_queries', [])
            print(f"   Test Queries Processed: {len(test_queries)}")
            
            successful_queries = sum(1 for q in test_queries if q.get('success', False))
            print(f"   Successful Queries: {successful_queries}/{len(test_queries)}")
        else:
            print(f"❌ Comprehensive test failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error running comprehensive test: {e}")

def main():
    """Main test function"""
    print("🏥 Mobile RAG System Test Suite")
    print("=" * 60)
    
    # Test direct access
    test_mobile_rag_direct()
    
    # Test via HTTP server
    test_mobile_rag_server()
    
    print("\n✅ Test suite completed!")
    print("\n💡 To start the HTTP server, run:")
    print("   python mobile_test_server.py")
    print("\n💡 To test specific queries, use:")
    print("   curl -X POST http://localhost:8000/test/emergency \\")
    print("     -H 'Content-Type: application/json' \\")
    print("     -d '{\"query\": \"Someone is having chest pain\"}'")

if __name__ == "__main__":
    main()

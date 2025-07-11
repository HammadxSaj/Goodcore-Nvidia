"""Test NVIDIA services integration"""

import asyncio
import sys
sys.path.append('src')

from core.nvidia_services import nvidia_client
from core.query_processor import query_processor
from utils.logging_config import setup_logging

async def test_nvidia_integration():
    """Test NVIDIA services integration"""
    
    setup_logging("INFO")
    print("🧪 Testing NVIDIA Services Integration...\n")
    
    # Test 1: Service health check
    print("1️⃣ Testing service health...")
    health_result = await nvidia_client.health_check()
    print(f"Health status: {health_result['status']}")
    if health_result['status'] == 'unhealthy':
        print(f"Health check failed: {health_result.get('error', 'Unknown error')}")
        print("⚠️  NVIDIA services may not be running. Starting basic tests anyway...\n")
    else:
        print("✅ Services are healthy\n")
    
    # Test 2: Complete service test
    print("2️⃣ Testing all services...")
    service_results = await nvidia_client.test_all_services()
    
    print("📊 Service Test Results:")
    for service, result in service_results.items():
        if isinstance(result, dict):
            status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
            print(f"{service}: {status}")
            if not result.get('success', False):
                print(f"  Error: {result.get('error', 'Unknown error')}")
    print()
    
    # Test 3: Query processing
    print("3️⃣ Testing query processing...")
    
    test_queries = [
        "Find cloud computing experts",
        "Who can speak about AI to executives?",
        "Technical speakers with DevOps experience"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Query {i}: {query}")
        
        try:
            result = await query_processor.process_query(query)
            print(f"✅ Processing successful")
            print(f"  Intent: {result['llm_analysis'].get('intent', 'Unknown')}")
            print(f"  Technologies: {result['llm_analysis'].get('technologies', [])}")
            print(f"  Expertise areas: {result['llm_analysis'].get('expertise', [])}")
            print(f"  Target audiences: {result['llm_analysis'].get('audiences', [])}")
            print(f"  Roles mentioned: {result['llm_analysis'].get('roles', [])}")
            print(f"  Has embedding: {'Yes' if result['query_embedding'] else 'No'}")
            print(f"  Boost factors applied: {len(result['boost_factors'])}")
            
        except Exception as e:
            print(f"❌ Processing failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Test 4: Direct embedding test
    print("\n4️⃣ Testing direct embedding generation...")
    
    test_texts = [
        "Expert in cloud computing and artificial intelligence",
        "Technical speaker with DevOps and automation experience",
        "Executive presenter specializing in digital transformation"
    ]
    
    try:
        embedding_response = await nvidia_client.generate_embeddings(test_texts)
        
        if embedding_response.success:
            print("✅ Embedding generation successful")
            print(f"  Generated embeddings for {len(test_texts)} texts")
            print(f"  Embedding dimension: {len(embedding_response.embeddings[0])}")
            print(f"  Token usage: {embedding_response.usage}")
        else:
            print(f"❌ Embedding generation failed: {embedding_response.error}")
            
    except Exception as e:
        print(f"❌ Embedding test failed: {e}")
    
    # Test 5: LLM response test
    print("\n5️⃣ Testing LLM response generation...")
    
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant for speaker recommendations."},
            {"role": "user", "content": "What makes a good technical speaker?"}
        ]
        
        llm_response = await nvidia_client.generate_llm_response(messages)
        
        if llm_response.success:
            print("✅ LLM response generation successful")
            print(f"  Response length: {len(llm_response.content)} characters")
            print(f"  Token usage: {llm_response.usage}")
            print(f"  Response preview: {llm_response.content[:200]}...")
        else:
            print(f"❌ LLM response failed: {llm_response.error}")
            
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
    
    print("\n🎯 NVIDIA Integration Testing Complete!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_nvidia_integration())
    if success:
        print("\n✅ All tests completed!")
    else:
        print("\n❌ Some tests failed!")
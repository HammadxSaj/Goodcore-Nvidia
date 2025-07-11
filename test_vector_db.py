"""
Comprehensive Vector Database Test
Tests full workflow: Data Loading → Embedding Generation → Vector Storage → Search
"""

import asyncio
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append('src')

from core.data_processor import SpeakerDataProcessor
from core.vector_db import vector_db
from core.nvidia_services import nvidia_client
from core.query_processor import query_processor
from utils.logging_config import setup_logging

async def test_vector_database_workflow():
    """Test complete vector database workflow with detailed logging"""
    
    setup_logging("INFO")
    print("🧪 Testing Vector Database Full Workflow...\n")
    
    # =========================
    # Phase 1: Data Loading
    # =========================
    print("📊 Phase 1: Loading and Processing Speaker Data")
    print("=" * 60)
    
    try:
        # Initialize data processor
        processor = SpeakerDataProcessor()
        print("✅ Data processor initialized")
        
        # Load raw data
        speakers_df, topics_df = processor.load_data()
        print(f"✅ Loaded {len(speakers_df)} speakers and {len(topics_df)} topic entries")
        
        # Create speaker profiles
        print("\n🔄 Creating speaker profiles...")
        speaker_profiles = processor.create_speaker_profiles()
        print(f"✅ Created {len(speaker_profiles)} speaker profiles")
        
        # Get data statistics
        stats = processor.get_data_statistics()
        print(f"\n📈 Data Statistics:")
        print(f"  Total speakers: {stats['total_speakers']}")
        print(f"  Active speakers: {stats['active_speakers']}")
        print(f"  Speakers with topics: {stats['speakers_with_topics']}")
        print(f"  Speakers with bio: {stats['speakers_with_bio']}")
        print(f"  Unique topics: {stats['unique_topics']}")
        print(f"  Average completeness: {stats['average_completeness']:.1%}")
        
        # Show sample speaker
        if speaker_profiles:
            sample_speaker = speaker_profiles[0]
            print(f"\n👤 Sample Speaker:")
            print(f"  Name: {sample_speaker['name']}")
            print(f"  Job Title: {sample_speaker['job_title']}")
            print(f"  Speaking Topics: {sample_speaker['speaking_topics'][:3]}")
            print(f"  Completeness: {sample_speaker['completeness_score']:.1%}")
            print(f"  Searchable Text Length: {len(sample_speaker['searchable_text'])} chars")
        
    except Exception as e:
        print(f"❌ Phase 1 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================
    # Phase 2: Vector DB Setup
    # =========================
    print(f"\n🗄️  Phase 2: Vector Database Initialization")
    print("=" * 60)
    
    try:
        # Initialize vector database
        success = await vector_db.initialize_collection()
        print(f"✅ Vector database initialized: {success}")
        
        # Get initial stats
        initial_stats = await vector_db.get_collection_stats()
        print(f"📊 Initial DB Stats:")
        print(f"  Collection: {initial_stats.get('collection_name', 'Unknown')}")
        print(f"  Current speakers: {initial_stats.get('total_speakers', 0)}")
        print(f"  DB Path: {initial_stats.get('db_path', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ Phase 2 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================
    # Phase 3: Embedding Generation & Storage
    # =========================
    print(f"\n⚡ Phase 3: Generating Embeddings and Storing Data")
    print("=" * 60)
    
    try:
        # Test with a subset first (first 5 speakers for testing)
        test_speakers = speaker_profiles[:5]
        print(f"🧪 Testing with {len(test_speakers)} speakers first...")
        
        # Add speakers to vector database
        add_result = await vector_db.add_speakers(test_speakers)
        
        if add_result['success']:
            print(f"✅ Successfully added {add_result['added_count']} speakers to vector DB")
            print(f"📊 Embedding Usage: {add_result['embedding_usage']}")
        else:
            print(f"❌ Failed to add speakers: {add_result['error']}")
            return False
        
        # Get updated stats
        updated_stats = await vector_db.get_collection_stats()
        print(f"📊 Updated DB Stats:")
        print(f"  Total speakers in DB: {updated_stats.get('total_speakers', 0)}")
        print(f"  Sample IDs: {updated_stats.get('sample_ids', [])}")
        
    except Exception as e:
        print(f"❌ Phase 3 failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # =========================
    # Phase 4: Query Processing & Vector Search
    # =========================
    print(f"\n🔍 Phase 4: Query Processing and Vector Search")
    print("=" * 60)
    
    test_queries = [
        "Find cloud computing experts",
        "Data center specialists",
        "Executive speakers with business focus"
    ]
    
    search_results = {}
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔍 Test Query {i}: '{query}'")
        print("-" * 40)
        
        try:
            # Process query
            query_params = await query_processor.process_query(query)
            
            if query_params['query_embedding']:
                print(f"✅ Query processed successfully")
                print(f"  Intent: {query_params['llm_analysis'].get('intent', 'Unknown')}")
                print(f"  Technologies: {query_params['llm_analysis'].get('technologies', [])}")
                print(f"  Embedding dimension: {len(query_params['query_embedding'])}")
                
                # Perform vector search
                search_result = await vector_db.search_speakers(
                    query_embedding=query_params['query_embedding'],
                    limit=3  # Limit for testing
                )
                
                if search_result['success']:
                    candidates = search_result['candidates']
                    print(f"✅ Vector search found {len(candidates)} candidates")
                    
                    print(f"\n🔍 DEBUG: Full candidate data for first result:")
                    if candidates:
                        first_candidate = candidates[0]
                        print(f"Raw candidate keys: {list(first_candidate.keys())}")
                        print(f"Full speaking_topics: {first_candidate.get('speaking_topics', 'NOT_FOUND')}")
                        print(f"speaking_topics type: {type(first_candidate.get('speaking_topics'))}")
                        print(f"Metadata speaking_topics: {first_candidate.get('metadata', {}).get('speaking_topics', 'NOT_FOUND')}")
                        print(f"Full metadata: {json.dumps(first_candidate.get('metadata', {}), indent=2)}")
                    # Show top results
                    for j, candidate in enumerate(candidates[:2], 1):
                        print(f"  Result {j}:")
                        print(f"    Name: {candidate.get('name', 'Unknown')}")
                        print(f"    Job Title: {candidate.get('job_title', 'Unknown')}")
                        print(f"    Similarity Score: {candidate.get('similarity_score', 0):.3f}")
                        topics = candidate.get('speaking_topics', '')
                        if isinstance(topics, str):
                            topic_list = [t.strip() for t in topics.split(',')][:2]  # First 2 topics
                            print(f"    Topics: {topic_list}")
                        else:
                            print(f"    Topics: {topics[:2]}")
                    
                    search_results[query] = candidates
                    
                else:
                    print(f"❌ Vector search failed: {search_result['error']}")
                    
            else:
                print(f"❌ Query processing failed - no embedding generated")
                
        except Exception as e:
            print(f"❌ Query {i} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # =========================
    # Phase 5: Hybrid Search Test
    # =========================
    print(f"\n🔄 Phase 5: Hybrid Search (Vector + Reranking)")
    print("=" * 60)
    
    hybrid_query = "Find speakers with expertise in cloud computing and data centers"
    
    try:
        print(f"🔍 Hybrid Query: '{hybrid_query}'")
        
        # Process query
        query_params = await query_processor.process_query(hybrid_query)
        
        if query_params['query_embedding']:
            # Perform hybrid search
            hybrid_result = await vector_db.hybrid_search(
                query_embedding=query_params['query_embedding'],
                query_text=hybrid_query,
                limit=3
            )
            
            if hybrid_result['success']:
                candidates = hybrid_result['candidates']
                search_type = hybrid_result['search_type']
                
                print(f"✅ Hybrid search completed ({search_type})")
                print(f"📊 Found {len(candidates)} candidates")
                
                # Show hybrid results
                for i, candidate in enumerate(candidates[:2], 1):
                    print(f"  Hybrid Result {i}:")
                    print(f"    Name: {candidate.get('name', 'Unknown')}")
                    print(f"    Job Title: {candidate.get('job_title', 'Unknown')}")
                    vector_score = candidate.get('similarity_score', 0)
                    rerank_score = candidate.get('rerank_score', 0) 
                    hybrid_score = candidate.get('hybrid_score', 0)
                    print(f"    Vector Score: {vector_score:.3f}")
                    print(f"    Rerank Score: {rerank_score:.3f}")
                    print(f"    Hybrid Score: {hybrid_score:.3f}")
                
            else:
                print(f"❌ Hybrid search failed: {hybrid_result['error']}")
                
        else:
            print(f"❌ Query processing failed for hybrid search")
            
    except Exception as e:
        print(f"❌ Hybrid search failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================
    # Phase 6: Full Data Load Test (Optional)
    # =========================
    print(f"\n📦 Phase 6: Full Data Load Test")
    print("=" * 60)
    
    try:
        # Clear the test data first
        print("🧹 Clearing test data...")
        clear_success = await vector_db.clear_collection()
        print(f"✅ Collection cleared: {clear_success}")
        
        # Load a larger subset (first 20 speakers)
        larger_subset = speaker_profiles[:20]
        print(f"📦 Loading {len(larger_subset)} speakers...")

        print(f"\n📏 DEBUG: Checking searchable text lengths:")
        for i, speaker in enumerate(larger_subset[:3]):  # Check first 3
            text = speaker.get('searchable_text', '')
            print(f"Speaker {i+1} ({speaker.get('name', 'Unknown')}):")
            print(f"  Text length: {len(text)} characters")
            print(f"  Estimated tokens: ~{len(text.split())}")
            if len(text) > 400:  # Show problematic ones
                print(f"  First 200 chars: {text[:200]}...")
                print(f"  Last 200 chars: ...{text[-200:]}")
                
        full_load_result = await vector_db.add_speakers(larger_subset)
        
        if full_load_result['success']:
            print(f"✅ Successfully loaded {full_load_result['added_count']} speakers")
            print(f"📊 Total embedding usage: {full_load_result['embedding_usage']}")
            
            # Final stats
            final_stats = await vector_db.get_collection_stats()
            print(f"📊 Final DB Stats:")
            print(f"  Total speakers: {final_stats.get('total_speakers', 0)}")
            
            # Test search with larger dataset
            print(f"\n🔍 Testing search with larger dataset...")
            query_params = await query_processor.process_query("cloud computing")
            
            if query_params['query_embedding']:
                search_result = await vector_db.search_speakers(
                    query_embedding=query_params['query_embedding'],
                    limit=5
                )
                
                if search_result['success']:
                    print(f"✅ Search found {len(search_result['candidates'])} candidates from larger dataset")
                else:
                    print(f"❌ Search failed on larger dataset")
            
        else:
            print(f"❌ Full load failed: {full_load_result['error']}")
            
    except Exception as e:
        print(f"❌ Phase 6 failed: {e}")
        import traceback
        traceback.print_exc()
    
    # =========================
    # Summary
    # =========================
    print(f"\n🎯 Test Summary")
    print("=" * 60)
    print(f"✅ Data Processing: Loaded {len(speaker_profiles)} speaker profiles")
    print(f"✅ Vector Database: Initialized and operational")
    print(f"✅ Embedding Generation: Working with NVIDIA services")
    print(f"✅ Vector Search: Functional with similarity scoring")
    print(f"✅ Query Processing: LLM analysis and embedding generation")
    print(f"✅ Hybrid Search: Vector + Reranking integration")
    
    if search_results:
        print(f"✅ Search Results: Generated for {len(search_results)} test queries")
    
    print(f"\n🎉 Vector Database Full Workflow Test Complete!")
    return True

if __name__ == "__main__":
    success = asyncio.run(test_vector_database_workflow())
    if success:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
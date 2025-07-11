"""
API routes for speaker search functionality
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import asyncio
from datetime import datetime
import sys
import os

# Add the src directory to Python path to import our modules
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Import our existing services with correct paths and names
from core.data_processor import SpeakerDataProcessor  # Correct class name
from core.vector_db import VectorDatabase
from core.query_processor import QueryProcessor
from core.nvidia_services import generate_llm_response

logger = logging.getLogger(__name__)

# Pydantic models for API
class SearchQuery(BaseModel):
    query: str
    max_results: int = 5

class SpeakerResult(BaseModel):
    speaker_id: str
    name: str
    job_title: str
    company: Optional[str] = None
    speaking_topics: List[str]
    bio: Optional[str] = None
    specializations: Optional[str] = None
    audiences: Optional[str] = None
    similarity_score: float
    rerank_score: Optional[float] = None

class SearchResponse(BaseModel):
    speakers: List[SpeakerResult]
    explanation: str
    recommendation: str
    query_analysis: Dict[str, Any]
    total_results: int
    search_time_ms: int

# Global instances (will be initialized on startup)
data_processor = None
vector_db = None
query_processor = None
speakers_loaded = False

router = APIRouter()

async def initialize_system():
    """Initialize all components and load data - Following the exact flow from test_vector_db.py"""
    global data_processor, vector_db, query_processor, speakers_loaded
    
    try:
        logger.info("🚀 Initializing Speaker Preference AI System...")
        
        # Phase 1: Initialize components (like in test)
        data_processor = SpeakerDataProcessor()
        vector_db = VectorDatabase()
        query_processor = QueryProcessor()
        
        # Phase 2: Load speaker data (following test flow)
        logger.info("📊 Loading speaker data...")
        
        # Load data using the actual methods from SpeakerDataProcessor
        speakers_df, topics_df = data_processor.load_data()
        logger.info(f"Loaded {len(speakers_df)} speakers and {len(topics_df)} topic entries")
        
        # Create speaker profiles
        speaker_profiles = data_processor.create_speaker_profiles()
        logger.info(f"Created {len(speaker_profiles)} speaker profiles")
        
        # Phase 3: Initialize vector database
        vector_db_ready = await vector_db.initialize_collection()
        if not vector_db_ready:
            raise Exception("Failed to initialize vector database")
        
        # Check if vector DB already has data
        db_stats = await vector_db.get_collection_stats()
        current_count = db_stats.get('total_speakers', 0)
        
        # Phase 4: Load speakers into vector DB if needed
        if current_count == 0:
            logger.info("⚡ Loading all speakers into vector database...")
            # Get all speakers (not just active ones for full dataset)
            all_speakers = data_processor.get_all_speakers()
            
            # Add speakers to vector DB
            result = await vector_db.add_speakers(all_speakers)
            if result["success"]:
                logger.info(f"✅ Loaded {result['added_count']} speakers into vector database")
            else:
                raise Exception(f"Failed to load speakers into vector database: {result.get('error')}")
        else:
            logger.info(f"✅ Vector database already contains {current_count} speakers")
        
        speakers_loaded = True
        logger.info("🎉 System initialization complete!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ System initialization failed: {e}")
        return False

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "system_ready": speakers_loaded,
        "timestamp": datetime.now().isoformat()
    }

@router.get("/stats")
async def get_system_stats():
    """Get system statistics"""
    if not speakers_loaded:
        raise HTTPException(status_code=503, detail="System not ready")
    
    # Use the correct method name from VectorDatabase
    db_stats = await vector_db.get_collection_stats()
    data_stats = data_processor.get_data_statistics()
    
    return {
        "total_speakers": db_stats.get('total_speakers', 0),
        "collection_name": db_stats.get('collection_name', 'unknown'),
        "database_path": db_stats.get('db_path', 'unknown'),
        "data_statistics": data_stats
    }

@router.post("/search", response_model=SearchResponse)
async def search_speakers(search_query: SearchQuery):
    """Search for speakers based on query - Following the exact flow from test"""
    if not speakers_loaded:
        raise HTTPException(status_code=503, detail="System not ready. Please wait for initialization to complete.")
    
    start_time = datetime.now()
    
    try:
        logger.info(f"🔍 Processing search query: {search_query.query}")
        
        # Phase 1: Process query (like in test)
        query_params = await query_processor.process_query(search_query.query)
        
        query_analysis = query_params.get('llm_analysis', {})
        query_embedding = query_params.get('query_embedding')
        
        if not query_embedding:
            raise HTTPException(status_code=500, detail="Failed to generate query embedding")
        
        logger.info(f"Query processed - Intent: {query_analysis.get('intent', 'unknown')}")
        
        # Phase 2: Perform hybrid search (like in test)
        search_results = await vector_db.hybrid_search(
            query_embedding=query_embedding,
            query_text=search_query.query,
            filters=None,  # Can add filters later
            limit=search_query.max_results
        )
        
        if not search_results["success"]:
            raise HTTPException(status_code=500, detail=f"Search failed: {search_results.get('error')}")
        
        candidates = search_results["candidates"]
        logger.info(f"Found {len(candidates)} candidates")
        
        # Phase 3: Convert results to API format
        speaker_results = []
        for candidate in candidates:
            # Extract speaking topics (handle both string and list formats)
            topics = candidate.get('speaking_topics', '')
            if isinstance(topics, str):
                topic_list = [t.strip() for t in topics.split(',') if t.strip()]
            else:
                topic_list = topics if isinstance(topics, list) else []
            
            # Get speaker ID - try different possible field names
            speaker_id = (
                candidate.get('speaker_id') or 
                candidate.get('id') or 
                candidate.get('metadata', {}).get('speaker_id') or
                'unknown'
            )
            
            speaker_result = SpeakerResult(
                speaker_id=str(speaker_id),
                name=candidate.get('name', candidate.get('metadata', {}).get('name', 'Unknown')),
                job_title=candidate.get('job_title', candidate.get('metadata', {}).get('job_title', '')),
                company=candidate.get('company', candidate.get('metadata', {}).get('company')),
                speaking_topics=topic_list,
                bio=candidate.get('bio', candidate.get('metadata', {}).get('bio', '')),
                specializations=candidate.get('specializations', candidate.get('metadata', {}).get('specializations', '')),
                audiences=candidate.get('audiences', candidate.get('metadata', {}).get('audiences', '')),
                similarity_score=candidate.get('similarity_score', 0.0),
                rerank_score=candidate.get('rerank_score')
            )
            speaker_results.append(speaker_result)
        
        # Phase 4: Generate explanation and recommendation using LLM (like in test)
        explanation_prompt = f"""
Based on the search query "{search_query.query}", I found {len(speaker_results)} relevant speakers. 

Here are the speakers found:
{chr(10).join([f"- {s.name} ({s.job_title}): {', '.join(s.speaking_topics[:3]) if s.speaking_topics else 'No topics listed'}" for s in speaker_results])}

Please provide:
1. A comprehensive explanation of why these speakers match the query requirements
2. Your personal recommendation for the top choice and why they are the best fit

Keep it professional and helpful for event organizers making speaker selection decisions.
"""
        
        llm_response = await generate_llm_response([
            {"role": "system", "content": "You are an expert event organizer helping to match speakers with event requirements. Provide clear, actionable insights."},
            {"role": "user", "content": explanation_prompt}
        ])
        
        if llm_response.success:
            llm_content = llm_response.content
            # Split explanation and recommendation
            if "recommendation" in llm_content.lower():
                parts = llm_content.split("recommendation", 1)
                explanation = parts[0].strip()
                recommendation = "Recommendation: " + parts[1].strip() if len(parts) > 1 else llm_content
            else:
                explanation = llm_content
                recommendation = f"Top recommendation: {speaker_results[0].name} - {speaker_results[0].job_title}" if speaker_results else "No recommendations available"
        else:
            explanation = f"Found {len(speaker_results)} speakers matching your criteria for '{search_query.query}'."
            recommendation = f"Top recommendation: {speaker_results[0].name} - {speaker_results[0].job_title}" if speaker_results else "No recommendations available"
        
        # Calculate search time
        search_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return SearchResponse(
            speakers=speaker_results,
            explanation=explanation,
            recommendation=recommendation,
            query_analysis=query_analysis,
            total_results=len(speaker_results),
            search_time_ms=search_time
        )
        
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
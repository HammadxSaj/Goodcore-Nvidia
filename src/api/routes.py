"""
API routes for speaker search functionality
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import asyncio
import json
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


class ErrorResponse(BaseModel):
    error: bool = True
    message: str
    suggestion: Optional[str] = None


# Global instances (will be initialized on startup)
data_processor = None
vector_db = None
query_processor = None
speakers_loaded = False

router = APIRouter()


async def _is_query_relevant(query: str) -> tuple[bool, str]:
    """
    Uses an LLM to check if a query is a relevant request for finding a speaker.
    Returns (is_relevant, error_message) tuple.
    """
    system_prompt = """You are a security and relevance guard for a speaker search system. 

Your ONLY job is to classify user queries and provide context. Respond with JSON containing:
1. "classification" - either "valid_speaker_request" or "irrelevant_request"
2. "category" - for irrelevant requests, categorize as: "general_question", "date_time", "weather", "math", "personal", "joke", "greeting", "other"

Examples of VALID requests:
- "Find me an expert in cloud computing" 
- "Speakers on AI for healthcare"
- "Who can talk about cybersecurity?"

Examples of IRRELEVANT requests:
- "What is the date today?" (category: "date_time")
- "How's the weather?" (category: "weather") 
- "Tell me a joke" (category: "joke")
- "What is 2+2?" (category: "math")
- "Hello" (category: "greeting")

Format: {"classification": "valid_speaker_request"} OR {"classification": "irrelevant_request", "category": "date_time"}"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f'Classify this query: "{query}"'},
    ]

    try:
        response = await generate_llm_response(
            messages, max_tokens=100, temperature=0.1
        )

        if response.success and response.content:
            content = response.content.strip()

            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
            else:
                json_str = content

            result = json.loads(json_str)
            classification = result.get("classification", "").lower()
            category = result.get("category", "other")

            if classification == "valid_speaker_request":
                logger.info(f"✅ Query relevance check PASSED for: '{query}'")
                return True, ""
            elif classification == "irrelevant_request":
                logger.info(
                    f"❌ Query relevance check FAILED for: '{query}' (category: {category})"
                )
                error_message = _generate_error_message(category, query)
                return False, error_message
            else:
                logger.warning(
                    f"⚠️ Unknown classification '{classification}' for query: '{query}'. Blocking by default."
                )
                return False, _generate_error_message("other", query)

    except Exception as e:
        logger.error(f"❌ Error during query relevance check for '{query}': {e}")
        return False, _generate_error_message("other", query)


def _generate_error_message(category: str, query: str) -> str:
    """Generate user-friendly error messages based on query category"""
    messages = {
        "date_time": "I specialize in finding speakers for events, not providing date or time information. Please ask me about speakers for specific topics, industries, or areas of expertise instead.",
        "weather": "I'm designed to help you find the perfect speakers for your events, not weather information. Try asking about speakers for topics like climate science, environmental sustainability, or meteorology if that's relevant to your event.",
        "math": "I'm a speaker search assistant, not a calculator. I'd be happy to help you find speakers who specialize in mathematics, data science, or analytics for your event though!",
        "joke": "While I appreciate humor, I'm focused on helping you find professional speakers for your events. Perhaps you'd like a speaker who specializes in comedy, entertainment, or public speaking techniques?",
        "greeting": "Hello! I'm here to help you find the perfect speakers for your event. Please tell me what topic, industry, or area of expertise you're looking for.",
        "general_question": "I'm specifically designed to help you discover speakers for events and conferences. Please ask me about finding speakers for particular topics, industries, or types of expertise.",
        "personal": "I'm a professional speaker search assistant and don't handle personal questions. I'd be glad to help you find speakers for topics like personal development, leadership, or career advancement if that's what you're interested in.",
        "other": "I can only assist with finding speakers for events and conferences. Please provide a query related to specific topics, industries, expertise areas, or speaker requirements you're looking for.",
    }

    return messages.get(category, messages["other"])


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
        logger.info(
            f"Loaded {len(speakers_df)} speakers and {len(topics_df)} topic entries"
        )

        # Create speaker profiles
        speaker_profiles = data_processor.create_speaker_profiles()
        logger.info(f"Created {len(speaker_profiles)} speaker profiles")

        # Phase 3: Initialize vector database
        vector_db_ready = await vector_db.initialize_collection()
        if not vector_db_ready:
            raise Exception("Failed to initialize vector database")

        # Check if vector DB already has data
        db_stats = await vector_db.get_collection_stats()
        current_count = db_stats.get("total_speakers", 0)

        # Phase 4: Load speakers into vector DB if needed
        if current_count == 0:
            logger.info("⚡ Loading all speakers into vector database...")
            # Get all speakers (not just active ones for full dataset)
            all_speakers = data_processor.get_all_speakers()

            # Add speakers to vector DB
            result = await vector_db.add_speakers(all_speakers)
            if result["success"]:
                logger.info(
                    f"✅ Loaded {result['added_count']} speakers into vector database"
                )
            else:
                raise Exception(
                    f"Failed to load speakers into vector database: {result.get('error')}"
                )
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
        "timestamp": datetime.now().isoformat(),
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
        "total_speakers": db_stats.get("total_speakers", 0),
        "collection_name": db_stats.get("collection_name", "unknown"),
        "database_path": db_stats.get("db_path", "unknown"),
        "data_statistics": data_stats,
    }


@router.post("/search")
async def search_speakers(search_query: SearchQuery):
    """Search for speakers based on query - Following the exact flow from test"""
    if not speakers_loaded:
        return ErrorResponse(
            message="Our speaker database is currently loading. Please try again in a few moments.",
            suggestion="Check the system status and try your search again shortly.",
        )

    # Relevance Guardrail Check
    is_relevant, error_message = await _is_query_relevant(search_query.query)
    if not is_relevant:
        return ErrorResponse(
            message=error_message,
            suggestion="Try asking about speakers for specific topics like 'AI experts', 'marketing professionals', or 'healthcare speakers'.",
        )

    start_time = datetime.now()

    try:
        logger.info(f"🔍 Processing search query: {search_query.query}")

        # Phase 1: Process query (like in test)
        query_params = await query_processor.process_query(search_query.query)

        query_analysis = query_params.get("llm_analysis", {})
        query_embedding = query_params.get("query_embedding")

        if not query_embedding:
            return ErrorResponse(
                message="I'm having trouble understanding your query right now. Please try rephrasing your speaker search request.",
                suggestion="Be specific about the topic, industry, or expertise you're looking for.",
            )

        logger.info(
            f"Query processed - Intent: {query_analysis.get('intent', 'unknown')}"
        )

        # Phase 2: Perform hybrid search (like in test)
        search_results = await vector_db.hybrid_search(
            query_embedding=query_embedding,
            query_text=query_params.get("enhanced_query", search_query.query),
            filters=None,
            limit=search_query.max_results,
        )

        if not search_results["success"]:
            return ErrorResponse(
                message="I encountered an issue while searching our speaker database. Please try again.",
                suggestion="If the problem persists, try a different search query or contact support.",
            )

        candidates = search_results["candidates"]
        logger.info(f"Found {len(candidates)} candidates")

        # Phase 3: Convert results to API format
        speaker_results = []
        for candidate in candidates:
            # Extract speaking topics (handle both string and list formats)
            topics = candidate.get("speaking_topics", "")
            if isinstance(topics, str):
                topic_list = [t.strip() for t in topics.split(",") if t.strip()]
            else:
                topic_list = topics if isinstance(topics, list) else []

            # Get speaker ID - try different possible field names
            speaker_id = (
                candidate.get("speaker_id")
                or candidate.get("id")
                or candidate.get("metadata", {}).get("speaker_id")
                or "unknown"
            )

            speaker_result = SpeakerResult(
                speaker_id=str(speaker_id),
                name=candidate.get(
                    "name", candidate.get("metadata", {}).get("name", "Unknown")
                ),
                job_title=candidate.get(
                    "job_title", candidate.get("metadata", {}).get("job_title", "")
                ),
                company=candidate.get(
                    "company", candidate.get("metadata", {}).get("company")
                ),
                speaking_topics=topic_list,
                bio=candidate.get("bio", candidate.get("metadata", {}).get("bio", "")),
                specializations=candidate.get(
                    "specializations",
                    candidate.get("metadata", {}).get("specializations", ""),
                ),
                audiences=candidate.get(
                    "audiences", candidate.get("metadata", {}).get("audiences", "")
                ),
                similarity_score=candidate.get("similarity_score", 0.0),
                rerank_score=candidate.get("rerank_score"),
            )
            speaker_results.append(speaker_result)

        # Phase 4: Generate explanation and recommendation using LLM (like in test)
        system_prompt = """You are an AI Assistant for Speaker Selection. Your goal is to provide a concise, professional, and helpful summary for event organizers.
    
            **Instructions:**
            1.  **Analyze the Results:** Review the user's query and the list of speakers found.
            2.  **Provide a High-Level Analysis:** In the "Analysis" section, your goal is to provide a strategic overview of the search results *as a group*.
                -   Synthesize information from all returned profiles to identify common themes, shared expertise, or different categories of speakers found (e.g., "The results include both deep technical
    experts and high-level strategic thinkers...").
                -   Explain why this *group* of candidates is a strong starting point for the user's search.
                -   Keep this section to 2-3 sentences. **Do not discuss individual speakers here.**
            3.  **Make a Top Recommendation:** In the "Top Recommendation" section, now focus on a single individual.
                -   Identify the single best speaker from the list who most closely matches the user's query.
                -   Justify your choice in 1-2 sentences, explaining what makes them stand out from the rest of the group.
            4.  **Guardrail:** Base your analysis STRICTLY on the provided speaker information. Do not invent or infer details not present in the context.
            5.  **Tone:** Be concise, professional, and direct.
   
            **Output Format:**
            - You MUST use the following Markdown structure. Do not add any other text.
            ### Analysis
            (Your 2-3 sentence high-level analysis of the group here)
   
            ### Top Recommendation
            (Your 1-2 sentence specific recommendation here)
        """

        # Create a concise context string for the LLM
        speaker_context = "\n".join(
            [
                f"- **{s.name}** ({s.job_title}): Specializes in {s.specializations or 'N/A'}. Key topics: {', '.join(s.speaking_topics[:3]) if s.speaking_topics else 'Not specified'}."
                for s in speaker_results
            ]
        )

        user_prompt = f"""
        **User Query:** "{search_query.query}"

        **Search Results:**
        {speaker_context}

        Please generate the analysis and recommendation based on these results.
        """

        llm_response = await generate_llm_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        explanation = f"Found {len(speaker_results)} speakers matching your criteria for '{search_query.query}'."
        recommendation = (
            f"Top recommendation: {speaker_results[0].name}"
            if speaker_results
            else "No recommendations available."
        )

        if llm_response.success:
            llm_content = llm_response.content
            # Split explanation and recommendation using the new, robust structure
            if "### Top Recommendation" in llm_content:
                parts = llm_content.split("### Top Recommendation")
                exp_part = parts[0].replace("### Analysis", "").strip()
                rec_part = parts[1].strip()

                if exp_part:
                    explanation = exp_part
                if rec_part:
                    recommendation = rec_part
            else:
                # Fallback if the model doesn't follow instructions perfectly
                explanation = llm_content.strip()

        # Calculate search time
        search_time = int((datetime.now() - start_time).total_seconds() * 1000)

        return SearchResponse(
            speakers=speaker_results,
            explanation=explanation,
            recommendation=recommendation,
            query_analysis=query_analysis,
            total_results=len(speaker_results),
            search_time_ms=search_time,
        )

    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return ErrorResponse(
            message="Something went wrong while processing your speaker search. Please try again.",
            suggestion="Try rephrasing your query or search for a different topic.",
        )

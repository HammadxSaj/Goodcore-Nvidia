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
    max_results: int = 10
    conversation_history: Optional[List[Dict[str, str]]] = None
    current_speakers: Optional[List[Dict[str, Any]]] = None


class SpeakerResult(BaseModel):
    speaker_id: str
    name: str
    job_title: str
    company: Optional[str] = None
    speaking_topics: List[str]
    bio: Optional[str] = None
    specializations: Optional[str] = None
    audiences: Optional[str] = None
    centers: Optional[str] = None
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
    system_prompt = """detailed thinking off. You are a security and relevance guard for a speaker search system. 

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


async def _process_refinement_action(
    query: str,
    conversation_history: List[Dict[str, str]],
    current_speakers: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Use LLM to analyze refinement query and determine action type"""

    # Create speaker context
    speaker_context = "\n".join(
        [
            f"Speaker {i+1}: ID={speaker.get('speaker_id', 'unknown')}, Name={speaker.get('name', 'Unknown')}, Title={speaker.get('job_title', '')}, Company={speaker.get('company', '')}, Centers={speaker.get('centers', '')}, Topics={', '.join(speaker.get('speaking_topics', []))}"
            for i, speaker in enumerate(current_speakers)
        ]
    )

    conversation_context = "\n".join(
        [
            f"{msg['role'].title()}: {msg['content']}"
            for msg in conversation_history[-5:]  # Last 5 messages for context
        ]
    )

    system_prompt = """Detailed thinking off. You are a conversational refinement processor for a speaker search system. Analyze the user's request and determine what action to take.

**Action Types:**
1. **ui_modification** - Simple list operations (remove specific speakers, reorder, clear all)
2. **new_search** - Search refinements that require new database queries (filter by location, add criteria, etc.)

**Instructions:**
1. Analyze the user's request in context of the conversation and current speaker list
2. Determine if this is a simple UI modification or requires a new search
3. For UI modifications: specify exactly which speakers to keep by their speaker_id
4. For new searches: extract the refined search criteria while preserving original intent

**Output JSON Format:**
{
  "action_type": "ui_modification" | "new_search",
  "reasoning": "Brief explanation of the action",
  "details": {
    // For ui_modification:
    "speakers_to_keep": ["speaker_id1", "speaker_id2"],
    
    // For new_search:
    "refined_criteria": {
      "intent": "combined intent from conversation",
      "mandatory_criteria": {
        "job_title_contains": [],
        "topics_must_include": [],
        "centers_must_include": []
      }
    },
    "preserve_relevant_speakers": true
  }
}

The response must be valid JSON and contain all required fields. Do not include any additional text or explanations outside the JSON format or before it. Just focus on returning the JSON object as specified.
"""

    user_prompt = f"""**Current Query:** "{query}"

**Conversation History:**
{conversation_context}

**Current Speakers:**
{speaker_context}

Analyze the request and determine the appropriate action. The response must be valid JSON and contain all required fields. Do not include any additional text or explanations outside the JSON format or before it. Just focus on returning the JSON object as specified.
"""

    try:
        response = await generate_llm_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
        )

        if response.success and response.content:
            try:
                result = json.loads(response.content.strip())
                # Add original data for context
                result["original_speakers"] = current_speakers
                result["original_query"] = query
                return result
            except json.JSONDecodeError:
                logger.warning(
                    f"Failed to parse refinement response: {response.content}"
                )
                return {"action_type": "error", "reasoning": "Could not parse response"}
        else:
            return {"action_type": "error", "reasoning": "LLM call failed"}

    except Exception as e:
        logger.error(f"Error in refinement processing: {e}")
        return {"action_type": "error", "reasoning": str(e)}


async def _generate_recommendation_with_llm(speakers: List[Dict[str, Any]], original_query: str) -> str:
    """Use LLM to generate a natural recommendation from the current speaker list"""

    # Create speaker context for LLM
    speaker_context = "\n".join([
        f"- **{speaker.get('name', 'Unknown')}** ({speaker.get('job_title', '')}) from {speaker.get('company', '')}: "
        f"Specializes in {speaker.get('specializations', 'N/A')}. "
        f"Biography: {speaker.get('bio', 'N/A')}. "
        f"Target audiences: {speaker.get('audiences', 'N/A')}. "
        f"Speaking topics: {', '.join(speaker.get('speaking_topics', [])) if speaker.get('speaking_topics') else 'Not specified'}. "
        f"Centers: {speaker.get('centers', 'N/A')}"
        for speaker in speakers  # Limit to top 5 for context
    ])

    system_prompt = """Detailed thinking off. You are a professional recommendation generator for speaker selection. Based on the user's original query and the current list of speakers, provide a natural, personalized recommendation.

**Instructions:**
1. Analyze the speakers in the context of the original query
2. Recommend the BEST speaker from the list who most closely matches the original request
3. Explain briefly (1-2 sentences) why this speaker is the top choice
4. Be natural and conversational, as if speaking directly to the event organizer
5. Focus on the speaker's relevant expertise and credentials

**Output Format:**
Provide only the recommendation text, no additional formatting or labels."""

    user_prompt = f"""**Original Query:** "{original_query}"

**Current Speakers Available:**
{speaker_context}
Based on the original query and these available speakers, who would you recommend and why? Just focus on providing the recommendation text without any additional formatting or labels. There is no need to mention that 'based on available speaker' or 'from the current list'. Just provide the recommendation text directly."""

    try:
        response = await generate_llm_response([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ])

        if response.success and response.content:
            return response.content.strip()
        else:
            # Fallback to simple recommendation
            return f"**{speakers[0].get('name', 'Unknown')}** - {speakers[0].get('job_title', '')}"

    except Exception as e:
        logger.error(f"Error generating LLM recommendation: {e}")
        return f"**{speakers[0].get('name', 'Unknown')}** remains the top choice from the available speakers."


async def _handle_ui_modification(
    action_result: Dict[str, Any], start_time: datetime
) -> SearchResponse:
    """Handle UI modifications like removing speakers with LLM-generated recommendations"""

    details = action_result.get("details", {})
    speakers_to_keep_ids = set(details.get("speakers_to_keep", []))
    original_speakers = action_result.get("original_speakers", [])

    # Filter speakers based on LLM decision
    filtered_speaker_dicts = [
        speaker
        for speaker in original_speakers
        if str(speaker.get("speaker_id")) in speakers_to_keep_ids
    ]

    # Convert to SpeakerResult objects
    updated_speakers = []
    for speaker_dict in filtered_speaker_dicts:
        try:
            speaker_result = SpeakerResult(
                speaker_id=str(speaker_dict.get("speaker_id", "unknown")),
                name=speaker_dict.get("name", "Unknown"),
                job_title=speaker_dict.get("job_title", ""),
                company=speaker_dict.get("company"),
                speaking_topics=speaker_dict.get("speaking_topics", []),
                bio=speaker_dict.get("bio", ""),
                specializations=speaker_dict.get("specializations", ""),
                audiences=speaker_dict.get("audiences", ""),
                centers=speaker_dict.get("centers", ""),
                similarity_score=speaker_dict.get("similarity_score", 0.0),
                rerank_score=speaker_dict.get("rerank_score"),
            )
            updated_speakers.append(speaker_result)
        except Exception as e:
            logger.warning(f"Could not convert speaker to SpeakerResult: {e}")
            continue

    # Generate new recommendation using LLM if we have speakers
    recommendation = ""
    explanation = f"Updated the speaker list as requested. {len(updated_speakers)} speakers remaining."

    if updated_speakers:
        # Use LLM to generate a natural recommendation
        recommendation = await _generate_recommendation_with_llm(
            [speaker.__dict__ for speaker in updated_speakers],
            action_result.get("original_query", "speaker search"),
        )
    else:
        recommendation = "No speakers remaining in the list."
        explanation = "All speakers have been removed from the list."

    search_time = int((datetime.now() - start_time).total_seconds() * 1000)

    return SearchResponse(
        speakers=updated_speakers,
        explanation=explanation,
        recommendation=recommendation,
        query_analysis={"intent": "list_modification"},
        total_results=len(updated_speakers),
        search_time_ms=search_time,
    )


async def _handle_refined_search(
    action_result: Dict[str, Any], search_query: SearchQuery, start_time: datetime
) -> SearchResponse:
    """Handle refined searches that require new database queries with enhanced query"""

    # Use enhance_query_with_llm to synthesize conversation context + current refinement
    enhanced_query = await query_processor.enhance_query_with_llm(
        query=search_query.query, conversation_history=search_query.conversation_history
    )

    # Generate query embedding using the enhanced query
    query_embedding = await query_processor._generate_query_embedding(enhanced_query)

    if not query_embedding:
        return ErrorResponse(
            message="I'm having trouble processing your refined search.",
            suggestion="Please try rephrasing your request.",
        )

    # Perform new search WITHOUT filters (as you requested)
    search_results = await vector_db.hybrid_search(
        query_embedding=query_embedding,
        query_text=enhanced_query,
        # filters=None,  # No filters as requested
        limit=search_query.max_results,
    )

    if not search_results["success"]:
        return ErrorResponse(
            message="The refined search encountered an issue.",
            suggestion="Try a different refinement or start a new search.",
        )

    candidates = search_results["candidates"]

    # Convert to speaker results
    speaker_results = []
    found_speaker_ids = set()

    for candidate in candidates:
        topics = candidate.get("speaking_topics", "")
        if isinstance(topics, str):
            topic_list = [t.strip() for t in topics.split(",") if t.strip()]
        else:
            topic_list = topics if isinstance(topics, list) else []

        speaker_id = (
            candidate.get("speaker_id")
            or candidate.get("id")
            or candidate.get("metadata", {}).get("speaker_id")
            or "unknown"
        )

        found_speaker_ids.add(str(speaker_id))

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
            centers=candidate.get(
                "centers", candidate.get("metadata", {}).get("centers", "")
            ),
            similarity_score=candidate.get("similarity_score", 0.0),
            rerank_score=candidate.get("rerank_score"),
        )
        speaker_results.append(speaker_result)

    # Simple speaker ID preservation - add existing speakers that aren't already in new results
    details = action_result.get("details", {})
    if details.get("preserve_relevant_speakers") and search_query.current_speakers:
        for existing_speaker in search_query.current_speakers:
            existing_speaker_id = str(existing_speaker.get("speaker_id", ""))

            # Only preserve if not already in new results
            if (
                existing_speaker_id not in found_speaker_ids
                and existing_speaker_id != "unknown"
            ):
                try:
                    preserved_speaker = SpeakerResult(
                        speaker_id=existing_speaker_id,
                        name=existing_speaker.get("name", "Unknown"),
                        job_title=existing_speaker.get("job_title", ""),
                        company=existing_speaker.get("company"),
                        speaking_topics=existing_speaker.get("speaking_topics", []),
                        bio=existing_speaker.get("bio", ""),
                        specializations=existing_speaker.get("specializations", ""),
                        audiences=existing_speaker.get("audiences", ""),
                        centers=existing_speaker.get("centers", ""),
                        similarity_score=existing_speaker.get(
                            "similarity_score", 0.5
                        ),  # Default score
                        rerank_score=existing_speaker.get("rerank_score"),
                    )
                    speaker_results.append(preserved_speaker)
                    logger.info(
                        f"Preserved existing speaker: {existing_speaker.get('name')} (ID: {existing_speaker_id})"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not preserve speaker {existing_speaker_id}: {e}"
                    )

    # Generate explanation and recommendation using LLM (same format as requested)
    if speaker_results:
        explanation = (
            f"Found {len(speaker_results)} speakers matching your refined criteria."
        )
        recommendation = await _generate_recommendation_with_llm(
            [speaker.__dict__ for speaker in speaker_results], search_query.query
        )
    else:
        explanation = "No speakers found matching your refined criteria."
        recommendation = "No speakers available for recommendation."

    search_time = int((datetime.now() - start_time).total_seconds() * 1000)

    return SearchResponse(
        speakers=speaker_results,
        explanation=explanation,
        recommendation=recommendation,
        query_analysis={"intent": "refined_search", "enhanced_query": enhanced_query},
        total_results=len(speaker_results),
        search_time_ms=search_time,
    )


def _speaker_meets_criteria(speaker: Dict[str, Any], criteria: Dict[str, Any]) -> bool:
    """Check if an existing speaker meets new mandatory criteria"""

    # Check job title criteria
    if criteria.get("job_title_contains"):
        job_title = speaker.get("job_title", "").lower()
        if not any(
            keyword.lower() in job_title for keyword in criteria["job_title_contains"]
        ):
            return False

    # Check topic criteria
    if criteria.get("topics_must_include"):
        all_topics = f"{speaker.get('speaking_topics', '')} {speaker.get('specializations', '')}".lower()
        if not any(
            topic.lower() in all_topics for topic in criteria["topics_must_include"]
        ):
            return False

    # Check center criteria
    if criteria.get("centers_must_include"):
        centers = speaker.get("centers", "").lower()
        if not any(
            center.lower() in centers for center in criteria["centers_must_include"]
        ):
            return False

    return True


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
        query_params = await query_processor.process_query(search_query.query,
                                                           conversation_history = search_query.conversation_history)

        query_analysis = query_params.get("llm_analysis", {})
        query_embedding = query_params.get("query_embedding")

        # Extract the mandatory filters from the analysis
        mandatory_filters = query_analysis.get("mandatory_criteria", {})
        logger.info(f"Extracted mandatory filters: {mandatory_filters}")

        if not query_embedding:
            return ErrorResponse(
                message="I'm having trouble understanding your query right now. Please try rephrasing your speaker search request.",
                suggestion="Be specific about the topic, industry, or expertise you're looking for.",
            )

        logger.info(
            f"Query processed - Intent: {query_analysis.get('intent', 'unknown')}"
        )

        # Phase 2: Perform hybrid search with mandatory filters
        logger.info(
            f"🔍 DEBUG: Calling hybrid_search with limit={search_query.max_results}"
        )
        search_results = await vector_db.hybrid_search(
            query_embedding=query_embedding,
            query_text=query_params.get("enhanced_query", search_query.query),
            filters=mandatory_filters,  # Pass the extracted filters here
            limit=search_query.max_results,
        )

        candidates = search_results["candidates"]
        logger.info(f"DEBUG - Raw first candidate structure: {candidates[0] if candidates else 'No candidates'}")

        # Check if centers exists in the raw data
        if candidates:
            first_candidate = candidates[0]
            logger.info(f"DEBUG - All keys in first candidate: {list(first_candidate.keys())}")
            logger.info(f"DEBUG - Metadata keys: {list(first_candidate.get('metadata', {}).keys()) if 'metadata' in first_candidate else 'No metadata'}")

        logger.info(
            f"🔍 DEBUG: Hybrid search returned {len(search_results.get('candidates', []))} candidates"
        )

        # Check if we need to implement fallback logic
        search_had_filters = (
            any(mandatory_filters.get(key) for key in mandatory_filters.keys())
            if mandatory_filters
            else False
        )
        search_was_successful = search_results["success"]
        results_found = len(search_results.get("candidates", [])) > 0

        explanation_prefix = ""

        if search_was_successful and not results_found and search_had_filters:
            logger.warning(
                f"Strict search for '{search_query.query}' yielded no results. Retrying with semantic search only."
            )

            # Re-run the search WITHOUT filters
            search_results = await vector_db.hybrid_search(
                query_embedding=query_embedding,
                query_text=query_params.get("enhanced_query", search_query.query),
                filters=None,  # No filters this time
                limit=search_query.max_results,
            )

            # Add a note for the user in the explanation
            # explanation_prefix = "Your search included specific criteria that returned no exact matches. The results below are the closest semantic matches based on your query. "
            logger.info(
                f"🔍 DEBUG: Fallback search returned {len(search_results.get('candidates', []))} candidates"
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

            centers_value = candidate.get("centers", candidate.get("metadata", {}).get("centers", ""))
            logger.info(f"DEBUG - Centers value for {candidate.get('name', 'Unknown')}: '{centers_value}'")

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
                centers=centers_value,
                similarity_score=candidate.get("similarity_score", 0.0),
                rerank_score=candidate.get("rerank_score"),
            )
            speaker_results.append(speaker_result)

        # NEW PHASE: LLM Shortlisting
        final_speaker_results = []
        if speaker_results:
            # Create a concise context of the candidates for the LLM
            shortlist_context = "\n".join(
                [
                    f"ID: {s.speaker_id}, Name: {s.name}, Title: {s.job_title}, Company: {s.company or 'N/A'}, Bio: {(s.bio + '...') if s.bio else s.bio or 'N/A'}, Topics: {', '.join(s.speaking_topics) if s.speaking_topics else 'N/A'}, Specializations: {s.specializations or 'N/A'}, Audiences: {s.audiences or 'N/A'}, Centers: {s.centers or 'N/A'}"
                    for s in speaker_results
                ]
            )

            shortlisting_system_prompt = """Detailed thinking off. You are an expert talent scout and event organizer's assistant. Your task is to review a list of potential speakers and shortlist the absolute best candidates based on the user's original query.

**Instructions:**
1. **Review the Query:** Carefully consider the user's original request.
2. **Analyze the Candidates:** Examine the provided list of up to 15 speaker profiles.
3. **Select the Best:** Choose a variable number of speakers who are the strongest match. Do not feel obligated to select all of them. Quality is more important than quantity. If only 3 are a great fit, select only 3. If only 1 is a great fit, select only 1.
4. **Must match certain criteria:** Ensure that the selected speakers meet the following:
    - There must be mention of the specific topic in their profile as mentioned in the query.
    - They should have relevant expertise or experience in the area.
    - There center location should match the user's query. i.e if the query mentions that they need speakers based in Bangalore then the speakers' center must be based in Bangalore. same for Santa Clara etc etc. There is a chance that the user might mention some place
    like New york, so based on gerographical proximity you can shrotlist people based in Santa Clara.
    - If there is any mention of a specific audience (e.g., executives, technical teams), they should be suitable for that audience.
    - if there is any mention of specialization, certification, or specific skills, they should have those qualifications.
    - if there is any mention of a specific role (e.g., technical speaker, executive presenter), they should fit that role.
    - if there is any mention of a certain experience level or years of experience, they should meet that requirement.
    **MUST**: The above criteria MUST be met for each speaker you select. Specifically the one on location/centers.
4. **Provide Justification:** For each speaker you select, provide a brief, one-sentence justification for why they are a good fit.
5. **Return JSON:** Your output MUST be a single, valid JSON object containing a list named "shortlist". Each item in the list should be an object with "speaker_id" and "justification".

YOU MUST ABIDE BY THE FOLLOWING FORMAT, There is no need to add any additional text or explanation outside of the JSON object or before it.

**Example Output Format:**
{
  "shortlist": [
    {
      "speaker_id": "12345",
      "justification": "This speaker's deep experience in cloud security along with their talks on AI in cybersecurity and being based in Santa Clara make them a perfect fit for the user's request."
    },
    {
      "speaker_id": "67890",
      "justification": "Offers a high-level, strategic perspective on the topic, which is perfect for a business audience or c-suite executives. They also have extensive experience in digital transformation and have spoken on multiple topics on cloud computing."
    }
  ]
}"""

            user_prompt_for_shortlisting = f"""
**User Query:** "{search_query.query}"

**Candidate Speakers:**
{shortlist_context}

Please analyze these candidates and return the JSON shortlist of the best fits that must match the criterias mentioned.There is no need to add any additional text or explanation outside of the JSON object or before it."""

            shortlisting_response = await generate_llm_response(
                [
                    {"role": "system", "content": shortlisting_system_prompt},
                    {"role": "user", "content": user_prompt_for_shortlisting},
                ]
            )

            if shortlisting_response.success:
                try:
                    shortlist_data = json.loads(shortlisting_response.content)
                    selected_ids = {
                        item["speaker_id"]
                        for item in shortlist_data.get("shortlist", [])
                    }

                    # Filter the original list to keep only the selected speakers
                    final_speaker_results = [
                        s for s in speaker_results if s.speaker_id in selected_ids
                    ]

                    logger.info(
                        f"LLM shortlisted {len(final_speaker_results)} speakers from the initial {len(speaker_results)}."
                    )

                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse LLM shortlist response. Falling back to original list."
                    )
                    final_speaker_results = speaker_results[:5]  # Fallback to top 5
            else:
                logger.warning(
                    "LLM shortlisting call failed. Falling back to original list."
                )
                final_speaker_results = speaker_results[:5]  # Fallback to top 5
        else:
            final_speaker_results = []

        # Phase 4: Generate explanation and recommendation using LLM (like in test)
        # Use final_speaker_results instead of speaker_results from this point forward
        system_prompt = """Detailed thinking off. You are an AI Assistant for Speaker Selection. Your goal is to provide a concise, professional, and helpful summary for event organizers.

**Instructions:**
1.  **Analyze the Results:** Review the user's query and the list of speakers found.
2.  **Provide a High-Level Analysis:** In the "Analysis" section, your goal is to provide a strategic overview of the search results *as a group*.
    -   Synthesize information from all returned profiles to identify common themes, shared expertise, or different categories of speakers found.
    -   Explain why this *group* of candidates is a strong starting point for the user's search.
    -   Keep this section to 2-3 sentences. **Do not discuss individual speakers here.**
3.  **Make a Top Recommendation:** In the "Top Recommendation" section, now focus on a single individual.
    -   Identify the single best speaker from the list who most closely matches the user's query.
    -   Justify your choice in 1-2 sentences, explaining what makes them stand out from the rest of the group.
    -   Do not write anything about any field being "missing" or "not specified". Focus on the strengths of the selected speaker.
4.  **Guardrail:** Base your analysis STRICTLY on the provided speaker information. Do not invent or infer details not present in the context.
5.  **Tone:** Be concise, professional, and direct.
6. **Must match certain criteria:** Ensure that the selected speakers meet the following:
    - There must be mention of the specific topic in their profile as mentioned in the query.
    - They should have relevant expertise or experience in the area.
    - There center location should match the user's query. i.e if the query mentions that they need speakers based in Bangalore then the speakers' center must be based in Bangalore. same for Santa Clara etc etc. There is a chance that the user might mention some place
    like New york, so based on gerographical proximity you can shrotlist people based in Santa Clara.
    - If there is any mention of a specific audience (e.g., executives, technical teams), they should be suitable for that audience.
    - if there is any mention of specialization, certification, or specific skills, they should have those qualifications.
    - if there is any mention of a specific role (e.g., technical speaker, executive presenter), they should fit that role.
    - if there is any mention of a certain experience level or years of experience, they should meet that requirement.
    **MUST**: The above criteria MUST be met for each speaker you select. Specifically the one on location/centers.


IF THERE IS NO SPEAKER FOUND, you MUST return a message like this:
"Unfortunately, I could not find any speakers matching your criteria. Please try broadening your search or consider different topics or expertise areas."

**Output Format:**
- You MUST use the following Markdown structure. Do not add any other text.
### Analysis
(Your 2-3 sentence high-level analysis of the group here)

### Top Recommendation
(Your 1-2 sentence specific recommendation here)

DO NOT ADD A SINGLE TEXT LINE BEYOND THE ### Analysis and ### Top Recommendation sections. NO MATTER WHAT. EVEN IF THERE IS SOME UNNECESSARY CONTENT IN THE QUERY THAT REQUIRES A FOLLOW UP
RESPONSE. THE RESPONSE MUST STRICTLY FOLLOW THIS FORMAT SINCE I NEED TO DO FURTHER PROCESSING ON IT TO DISPLAY IT IN THE UI.

If there is conversation history, you can use it to provide context, but do not mention it in the output. Just focus on the speakers and the user's query.
"""

        # Create a concise context string for the LLM using final_speaker_results
        speaker_context = "\n".join(
            [
                f"ID: {s.speaker_id}, Name: {s.name}, Title: {s.job_title}, Company: {s.company or 'N/A'}, Bio: {(s.bio + '...') if s.bio else s.bio or 'N/A'}, Topics: {', '.join(s.speaking_topics) if s.speaking_topics else 'N/A'}, Specializations: {s.specializations or 'N/A'}, Audiences: {s.audiences or 'N/A'}, Centers: {s.centers or 'N/A'}"
                for s in final_speaker_results
            ]
        )

        user_prompt = f"""
**User Query:** "{search_query.query}"

**Search Results:**
{speaker_context}

Please generate the analysis and recommendation based on these results.

DO NOT ADD A SINGLE TEXT LINE BEYOND THE ### Analysis and ### Top Recommendation sections. NO MATTER WHAT. EVEN IF THERE IS SOME UNNECESSARY CONTENT IN THE QUERY THAT REQUIRES A FOLLOW UP
RESPONSE. THE RESPONSE MUST STRICTLY FOLLOW THIS FORMAT SINCE I NEED TO DO FURTHER PROCESSING ON IT TO DISPLAY IT IN THE UI.

If there is conversation history, you can use it to provide context, but do not mention it in the output. Just focus on the speakers and the user's query.

"""

        llm_response = await generate_llm_response(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )

        explanation = f"Found {len(final_speaker_results)} speakers matching your criteria for '{query_params.get('enhanced_query', search_query.query)}'."
        recommendation = (
            f"Top recommendation: {final_speaker_results[0].name}"
            if final_speaker_results
            else "No recommendations available."
        )

        print("----------------------------------------------------------------------------------")
        print(f"LLM response: {llm_response.content}")
        if llm_response.success:
            llm_content = llm_response.content
            # Split explanation and recommendation using the new, robust structure
            if "### Top Recommendation" in llm_content:
                parts = llm_content.split("### Top Recommendation")
                exp_part = parts[0].replace("### Analysis", "").strip()
                rec_part = parts[1].strip()

                if exp_part:
                    explanation = (
                        explanation_prefix + exp_part
                    )  # Add prefix if fallback was used
                if rec_part:
                    recommendation = rec_part
            else:
                # Fallback if the model doesn't follow instructions perfectly
                explanation = explanation_prefix + llm_content.strip()

        # Calculate search time
        search_time = int((datetime.now() - start_time).total_seconds() * 1000)

        return SearchResponse(
            speakers=final_speaker_results,  # Use the shortlisted results
            explanation=explanation,
            recommendation=recommendation,
            query_analysis=query_analysis,
            total_results=len(final_speaker_results),  # Use the final count
            search_time_ms=search_time,
        )

    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return ErrorResponse(
            message="Something went wrong while processing your speaker search. Please try again.",
            suggestion="Try rephrasing your query or search for a different topic.",
        )


@router.post("/refine")
async def refine_speakers(search_query: SearchQuery):
    """Refine existing speaker results based on conversational input"""
    if not speakers_loaded:
        return ErrorResponse(
            message="Our speaker database is currently loading. Please try again in a few moments.",
            suggestion="Check the system status and try your search again shortly.",
        )

    start_time = datetime.now()

    try:
        logger.info(f"🔍 Processing refinement query: {search_query.query}")

        # Use LLM to determine the action type and execute it
        action_result = await _process_refinement_action(
            query=search_query.query,
            conversation_history=search_query.conversation_history or [],
            current_speakers=search_query.current_speakers or [],
        )

        if action_result["action_type"] == "ui_modification":
            # Handle UI modifications (remove, reorder, etc.)
            return await _handle_ui_modification(action_result, start_time)

        elif action_result["action_type"] == "new_search":
            # Handle new search with refined criteria
            return await _handle_refined_search(action_result, search_query, start_time)

        else:
            # Handle errors or unrecognized actions
            return ErrorResponse(
                message="I couldn't understand that refinement request.",
                suggestion="Try being more specific, like 'remove speaker 2' or 'only show speakers from Santa Clara'.",
            )

    except Exception as e:
        logger.error(f"❌ Refinement error: {e}")
        return ErrorResponse(
            message="Something went wrong while processing your refinement. Please try again.",
            suggestion="Try rephrasing your request or starting a new search.",
        )

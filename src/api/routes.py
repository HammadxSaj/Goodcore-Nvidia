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
from core.bm25_service import bm25_service

logger = logging.getLogger(__name__)


# Pydantic models for API
class SearchQuery(BaseModel):
    query: str
    max_results: int = 10
    conversation_history: Optional[List[Dict[str, str]]] = None
    current_speakers: Optional[List[Dict[str, Any]]] = None
    enhanced_query: Optional[str] = None


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
    system_prompt = """/no_think detailed thinking off. You are a security and relevance guard for a speaker search system. 

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
        {"role": "user", "content": f'/no_think Classify this query: "{query}"'},
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

        # Phase 2.5: Initialize BM25 index (NEW)
        logger.info("🔍 Creating BM25 index...")
        bm25_success = bm25_service.create_index(speaker_profiles)
        if not bm25_success:
            logger.warning("BM25 index creation failed, continuing without BM25")
        else:
            logger.info("✅ BM25 index created successfully")

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

    speaker_context = "\n".join(
        [
            f"Speaker {i+1}: ID={speaker.get('speaker_id', 'unknown')}, Name={speaker.get('name', 'Unknown')}"
            for i, speaker in enumerate(current_speakers)
        ]
    )

    conversation_context = "\n".join(
        [
            f"{msg['role'].title()}: {msg['content']}"
            for msg in conversation_history[-1:]
        ]
    )

    system_prompt = """/no_think Detailed thinking off. You are a conversational refinement processor for a speaker search system.

**IMPORTANT: Focus ONLY on the current user query.**

**Action Types:**
1.  **ui_modification**: For requests to remove speakers from the current list.
2.  **new_search**: For requests that add new criteria and require a new search.

**Instructions:**
1.  **Analyze ONLY the CURRENT user request.**
2.  **For removal requests**:
    -   Identify the speaker(s) to remove by name (case-insensitive, partial matching allowed).
    -   Your JSON output's `details` object MUST contain a `speakers_to_remove` list with the exact `speaker_id`(s) of the identified
speaker(s).
    -   If the requested speaker is not in the current list, return an empty `speakers_to_remove` list and explain why in the `reasoning`.
3.  **For new searches**: Return `{"action_type": "new_search"}`.

**Output JSON Format:**
{
  "action_type": "ui_modification" | "new_search",
  "reasoning": "Explain the action taken in 1-2 sentences.",
  "details": {
    "speakers_to_remove": ["speaker_id_to_remove_1", "speaker_id_to_remove_2"]
  }
}

**CRITICAL**: Your primary job for UI modifications is to identify the IDs of speakers to be removed. Do not add any other fields to the
details object.
"""

    user_prompt = f"""/no_think **CURRENT REQUEST ONLY:** "{query}"

**Current Speakers Available to Modify:**
{speaker_context}

**Previous Context (for reference only, do NOT act on these):**
{conversation_context}

Analyze ONLY the current request "{query}" and determine the appropriate action. The response must be valid JSON.
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
        for speaker in speakers 
    ])

    system_prompt = """ /no_think Detailed thinking off. You are a professional recommendation generator for speaker selection. Based on the user's original query and the current list of speakers, provide a natural, personalized recommendation.

**Instructions:**
1. Analyze the speakers in the context of the original query
2. Recommend the BEST speaker from the list who most closely matches the original request
3. Explain briefly (1-2 sentences) why this speaker is the top choice
4. Be natural and conversational, as if speaking directly to the event organizer
5. Focus on the speaker's relevant expertise and credentials
6. If there are no speakers available, just say that there are no speakers available.

**Output Format:**
Provide only the recommendation text, no additional formatting or labels."""

    user_prompt = f"""/no_think **Original Query:** "{original_query}"

**Current Speakers Available:**
{speaker_context}
Based on the original query and these available speakers, who would you recommend and why? Just focus on providing the recommendation text without any additional formatting or labels. There is no need to mention that 'based on available speaker' or 'from the current list'. Just provide the recommendation text directly."""

    try:
        response = await generate_llm_response([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ], temperature=0.7)

        if response.success and response.content:
            return response.content.strip()
        else:
            # Fallback to simple recommendation
            return f"**{speakers[0].get('name', 'Unknown')}** - {speakers[0].get('job_title', '')}"

    except Exception as e:
        logger.error(f"Error generating LLM recommendation: {e}")
        return f"**{speakers[0].get('name', 'Unknown')}** remains the top choice from the available speakers."

# Add this helper function after the imports and before the router definition:


def _convert_list_to_string(value) -> str:
    """Convert a list to a comma-separated string, or return the string as-is"""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value if item)
    return str(value) if value else ""


async def _handle_ui_modification(
    action_result: Dict[str, Any],
    start_time: datetime,
    enhanced_query: Optional[str] = None,
) -> SearchResponse:
    """Handle UI modifications like removing speakers with LLM-generated recommendations"""

    details = action_result.get("details", {})
    speakers_to_remove_ids = set(details.get("speakers_to_remove", []))
    original_speakers = action_result.get("original_speakers", [])

    # Filter the list by EXCLUDING the speakers to remove
    filtered_speaker_dicts = [
        speaker
        for speaker in original_speakers
        if str(speaker.get("speaker_id")) not in speakers_to_remove_ids
    ]

    # Convert to SpeakerResult objects
    updated_speakers = []
    for speaker_dict in filtered_speaker_dicts:
        try:
            speaker_result = SpeakerResult(**speaker_dict)
            updated_speakers.append(speaker_result)
        except Exception as e:
            logger.warning(f"Could not convert speaker to SpeakerResult: {e}")
            continue

    # Generate new recommendation using LLM if we have speakers
    recommendation = ""
    explanation = action_result.get(
        "reasoning",
        f"Updated the speaker list as requested. {len(updated_speakers)} speakers remaining.",
    )

    if updated_speakers:
        # recommendation = await _generate_recommendation_with_llm(
        #     [speaker.model_dump() for speaker in updated_speakers],
        #     enhanced_query or action_result.get("original_query", "speaker search"),
        # )

        recommendation = ""
        
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

    # Add BM25 stats
    bm25_stats = bm25_service.get_index_stats()

    return {
        "total_speakers": db_stats.get("total_speakers", 0),
        "collection_name": db_stats.get("collection_name", "unknown"),
        "database_path": db_stats.get("db_path", "unknown"),
        "data_statistics": data_stats,
        "bm25_statistics": bm25_stats,
    }


@router.post("/search")
async def search_speakers(search_query: SearchQuery):
    """Search for speakers based on query - Following the exact flow from test"""
    import time

    start_time = time.time()


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
            suggestion="Try asking about speakers for specific topics like 'Healthcare experts that have knowledge of NVIDIA Clara and can speak on the topic of AI and LLMs'",
        )

    start_time = datetime.now()

    try:
        logger.info(f"🔍 Processing search query: {search_query.query}")

        query_params = await query_processor.process_query(search_query.query,
                                                           conversation_history = search_query.conversation_history)

        query_analysis = query_params.get("llm_analysis", {})
        query_embedding = query_params.get("query_embedding")

        if not query_embedding:
            return ErrorResponse(
                message="I'm having trouble understanding your query right now. Please try rephrasing your speaker search request.",
                suggestion="Be specific about the topic, industry, or expertise you're looking for.",
            )

        # Phase 2: Perform hybrid search with mandatory filters
        logger.info(
            f"🔍 DEBUG: Calling hybrid_search with limit={search_query.max_results}"
        )
        search_results = await vector_db.hybrid_search(
            query_embedding=query_embedding,
            query_text=query_params.get("enhanced_query", search_query.query),
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

        explanation_prefix = ""

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
                or candidate.get("metadata", {}).get("id")
                or "unknown"
            )

            centers_value = candidate.get("centers", candidate.get("metadata", {}).get("centers", ""))
            # logger.info(f"DEBUG - Centers value for {candidate.get('name', 'Unknown')}: '{centers_value}'")

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
                audiences=_convert_list_to_string(
                    candidate.get(
                        "audiences", candidate.get("metadata", {}).get("audiences", "")
                    )
                ),
                centers=centers_value,
                similarity_score=candidate.get("similarity_score", 0.0),
                rerank_score=candidate.get("rerank_score"),
            )
            speaker_results.append(speaker_result)

        top_15_speakers = speaker_results[:15]  # Take only top 15 for LLM shortlisting

        # NEW PHASE: LLM Shortlisting
        final_speaker_results = []
        if top_15_speakers:
            # Create a concise context of the candidates for the LLM
            shortlist_context = "\n".join(
                [
                    f"ID: {s.speaker_id}, Name: {s.name}, Title: {s.job_title}, Company: {s.company or 'N/A'}, Bio: {(s.bio + '...') if s.bio else s.bio or 'N/A'}, Topics: {', '.join(s.speaking_topics) if s.speaking_topics else 'N/A'}, Specializations: {s.specializations or 'N/A'}, Audiences: {s.audiences or 'N/A'}, Centers: {s.centers or 'N/A'}"
                    for s in top_15_speakers
                ]
            )

            previous_speakers_context = ""
            if search_query.current_speakers:
                previous_speakers_context = "\n".join(
                    [
                        f"ID: {s.get('speaker_id', 'unknown')}, Name: {s.get('name', 'Unknown')}, Title: {s.get('job_title', '')}, Company: {s.get('company', '') or 'N/A'}, Bio: {(s.get('bio', '') + '...') if s.get('bio') else 'N/A'}, Topics: {', '.join(s.get('speaking_topics', [])) if s.get('speaking_topics') else 'N/A'}, Specializations: {s.get('specializations', '') or 'N/A'}, Audiences: {s.get('audiences', '') or 'N/A'}, Centers: {s.get('centers', '') or 'N/A'}"
                        for s in search_query.current_speakers
                    ]
                )

            shortlisting_system_prompt = """/no_think Detailed thinking off. You are an expert talent scout and event organizer's assistant. Your task is to review a list of potential speakers and shortlist the absolute best candidates based on the user's original query.

**Instructions:**
1. **Review the Query:** Carefully consider the user's original request.
2. **Analyze the Candidates:** Examine the provided list of speaker profiles.
3. **Select the Best:** Choose a variable number of speakers who are the strongest match. Do not feel obligated to select all of them. Quality is more important than quantity. If only 6 are a great fit, select only 6. If only 1 is a great fit, select only 1. If all 15 are a great fit, select all 15.
4. **Consider Previous Speakers:** You can also consider speakers from previous searches if they match the new criteria. But if the the past speakers are totally different, you can ignore the previous speakers.
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
5. **Return JSON:** Your output MUST be a single, valid JSON object containing a list named "shortlist". Each item in the list should be an object with "speaker_id" and "justification".
6. The criteria for location/centers is very important, so make sure to check that the speakers' center matches the user's query. If the query mentions a specific location like "Bangalore", then the speakers' center must be based in Bangalore or a nearby area. 
7. Even if the profile is a perfect match, if the center/location does not match the user's query, you must not include that speaker in the shortlist. If no speaker matches the location criteria, you can still provide a recommendation based on the best available speaker, but make sure to mention that in your recommendation that no speaker matched the location criteria and that the recommendation is based on the best available speaker.
8. There can be an exception in case of center/location in the case when the job title specifically says that the person is maybe the "Head of Healthcare - EMEA" or "Head of Healthcare - APAC" etc. In that case you can shortlist the speaker even if the center/location does not match the user's query, but make sure to mention that in your recommendation that the speaker is based in a different location but is a great fit for the role.
YOU MUST ABIDE BY THE FOLLOWING FORMAT, There is no need to add any additional text or explanation outside of the JSON object or before it.

**Example Output Format:**
{
  "shortlist": [
    {
      "speaker_id": "12345",
    },
    {
      "speaker_id": "67890"
    }
  ]
}

You MUST return a single JSON object with a list named "shortlist". Each item in the list should be an object with "speaker_id" having a string value.
"""

###########################################BELOW IS THE PROMPT PROVIDED BY GEMINI WHICH DOES NOT WORK, SO WE ARE USING THE ABOVE PROMPT INSTEAD###########################################

#             shortlisting_system_prompt = """/no_think Detailed thinking off. You are an elite AI Talent Scout. Your purpose is to perform a rigorous, criteria-driven analysis of speaker candidates and identify the absolute best matches for a user's request. Your judgment is precise, and you adhere strictly to the provided constraints.

# WORKFLOW  
# Deconstruct the Query: Dissect the user's request into a clear set of requirements: topic, location, audience, experience, specific skills, etc.  

# Analyze the Candidate Pool: Scrutinize the profile of each provided candidate. You may also consider speakers from previous searches if they are provided and align with the current query's requirements.  

# Apply Mandatory Filters: Each candidate must pass ALL the mandatory filters defined below. A single failure means disqualification.  

# Generate Output: Produce a single, valid JSON object containing the speaker_id of every candidate who passed the filtration process.  

# MANDATORY FILTERING CRITERIA  
# You must apply these rules with zero exceptions. If a candidate fails even one, they are not to be included in the shortlist.  

# - Topic Match: The speaker's profile must contain an explicit mention of the specific topic requested in the query.  
# - Location/Center Match (CRITICAL):  
#   - Primary Rule: The speaker's center/location must match the location specified in the query.  
#   - Proximity Rule: If a broader region is mentioned (e.g., "Bay Area"), candidates in geographically close and relevant cities are acceptable (e.g., a query for "San Francisco" allows for a candidate from "San Jose" or "Palo Alto").  
#   - Regional Role Exception: A candidate with a regional title (e.g., "Head of Healthcare - EMEA", "President - APAC") can be shortlisted for a query within that region, even if their listed center is in a different city within that region.  
# - Demonstrable Expertise: The profile must show relevant experience, projects, or credentials in the requested topic area.  
# - Audience Suitability: If the query specifies an audience type (e.g., C-level executives, technical developers, sales teams), the speaker's background must be appropriate for that audience.  
# - Specific Qualifications: If the query demands specific certifications (e.g., PMP, AWS Certified), skills, or roles (e.g., "technical evangelist"), the candidate must possess them.  

# OUTPUT SPECIFICATION  
# Format: Your entire output MUST be a single, valid JSON object. Do not include any explanatory text, apologies, or summaries before or after the JSON block.  

# Content: The JSON object will contain a single key, "shortlist", which holds a list of objects. Each object in the list will contain a single key, "speaker_id".  

# No-Match Protocol: If NO candidates satisfy all the mandatory criteria (especially location), you MUST return an empty list.  

# Example of a valid output with matches:  

# JSON  
# ```json
# {
#   "shortlist": [
#     {
#       "speaker_id": "spk_1a2b3c"
#     },
#     {
#       "speaker_id": "spk_4d5e6f"
#     }
#   ]
# }

# Example of a valid output with NO matches:

# ```json
# {
#   "shortlist": []
# }
# """

            user_prompt_parts = [
                f'/no_think **User Query:** "{query_params.get("enhanced_query", search_query.query)}"'
            ]

            user_prompt_parts.append(f"**NEW Search Results:**\n{shortlist_context}")

            if previous_speakers_context:
                ####################currently commented out since we are not using previous speakers context due to context length issues
                # user_prompt_parts.append(f"**PREVIOUS Speakers (from earlier searches):**\n{previous_speakers_context}")
                user_prompt_parts.append(
                    "**PREVIOUS Speakers (from earlier searches):**None"
                )
            else:
                user_prompt_parts.append("**PREVIOUS Speakers:** None")

            user_prompt_parts.append(
                "Please analyze these candidates and return the JSON shortlist of the best fits that must match the criterias mentioned. Consider speakers from BOTH new and previous results since the previous candidates may also match the new requirement. There is no need to add any additional text or explanation outside of the JSON object or before it. YOU MUST ABIDE BY THE FOLLOWING FORMAT, that is a single JSON object with a list named 'shortlist'. Each item in the list should be an object with 'speaker_id' having a string value"
            )

            user_prompt_for_shortlisting = "\n\n".join(user_prompt_parts)

            shortlisting_response = await generate_llm_response(
                [
                    {"role": "system", "content": shortlisting_system_prompt},
                    {"role": "user", "content": user_prompt_for_shortlisting},
                ],
                temperature=0.1,
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
                        s for s in top_15_speakers if s.speaker_id in selected_ids
                    ]

                    logger.info(
                        f"LLM shortlisted {len(final_speaker_results)} speakers from the initial {len(top_15_speakers)}."
                    )

                except json.JSONDecodeError:
                    logger.warning(
                        "Failed to parse LLM shortlist response. Falling back to original list."
                    )
                    final_speaker_results = top_15_speakers[:5]  # Fallback to top 5
            else:
                logger.warning(
                    "LLM shortlisting call failed. Falling back to original list."
                )
                final_speaker_results = top_15_speakers[:5]  # Fallback to top 5
        else:
            final_speaker_results = []

        # Phase 4: Generate explanation and recommendation using LLM (like in test)
        # Use final_speaker_results instead of speaker_results from this point forward
        system_prompt = """/no_think Detailed thinking off. You are an AI Assistant for Speaker Selection. Your goal is to provide a concise, professional, and helpful summary for event organizers.

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
6.  **Tone:** Be concise, professional, and direct.
7. **If search results are empty, You must follow this format:**
   ### Analysis
   No speakers found matching your criteria.
    ### Top Recommendation
    None

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

        user_prompt = f""" /no_think
**User Query:** "{query_params.get('enhanced_query', search_query.query)}"

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
            ],
            temperature=0.7,
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
        end_time = datetime.now()

        search_time = int((end_time - start_time).total_seconds() * 1000)

        logger.info(f"🔍 Search completed in {search_time} ms")

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
            return await _handle_ui_modification(action_result, start_time, search_query.enhanced_query)

        elif action_result["action_type"] == "new_search":
            # Handle new search with refined criteria
            return await search_speakers(search_query)
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

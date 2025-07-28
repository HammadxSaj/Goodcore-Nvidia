"""
Query processor for handling natural language queries - Simplified with LLM-based processing
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
import json
from datetime import datetime

from .nvidia_services import nvidia_client
from .config import config

logger = logging.getLogger(__name__)

class QueryProcessor:
    """Processes natural language queries for speaker search - Simplified approach"""

    def __init__(self):
        pass

    async def process_query(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Process a natural language query and extract search parameters"""

        logger.info(f"Processing query: {query}")

        # Basic cleaning - just trim whitespace
        cleaned_query = query.strip()

        # STEP 1: Enhance the query with LLM for better search results (with conversation context)
        enhanced_query = await self.enhance_query_with_llm(cleaned_query, conversation_history)
        logger.info(f"Enhanced query: {enhanced_query}")

        # STEP 2: Use enhanced query for LLM analysis (with conversation context)
        # llm_analysis = await self._analyze_query_with_llm(enhanced_query, conversation_history)

        # STEP 3: Generate query embedding using enhanced query
        query_embedding = await self._generate_query_embedding(enhanced_query)

        # Create search parameters
        search_params = {
            'original_query': query,
            'cleaned_query': cleaned_query,
            'enhanced_query': enhanced_query,
            #'llm_analysis': llm_analysis,
            'query_embedding': query_embedding,
            #'boost_factors': self._create_boost_factors(llm_analysis),
            'timestamp': datetime.now().isoformat()
        }

        # logger.info(f"Query processing complete - Primary intent: {llm_analysis.get('intent', 'unknown')}")

        return search_params

    async def _analyze_query_with_llm(self, current_query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Use LLM to analyze the current query in the context of a conversation history."""

        system_prompt = """/no_think Detailed thinking off. You are a highly intelligent query analysis engine for a speaker search system. Your purpose is to analyze a user's latest query by synthesizing it with the entire preceding conversation history to produce a single, consolidated set of search criteria.

    **Core Task:**
    Based on the full `conversation_history` and the `current_query`, determine the user's complete and final intent. A follow-up query like "only the ones from Bangalore" MUST be combined with the previous context (e.g., "find me GPU experts") to form a new, complete search for "GPU experts from Bangalore".

    **Instructions:**
    1. **Synthesize, Don't Just Analyze:** Do not just analyze the `current_query`. You MUST interpret it based on the `conversation_history`.
    2. **Consolidate Criteria:** If the user refines their search, merge the new criteria with the old. For example, if they first ask for "AI experts" and then say "who are also VPs", the new `job_title_contains` criteria should be `["VP"]` and the `expertise_keywords` should still contain `["AI"]`.
    3. **Identify Mandatory Criteria:** Extract strict requirements (specific job titles, locations, required topics) into the `mandatory_criteria` object.
    4. **Handle Refinements:** If the current query is a refinement like "remove John Smith" or "only show the first 3", this is a UI action and should still preserve the original search intent.
    5. **Location Keywords:** Look for location indicators like "North America", "Santa Clara", "HQ", city names, country names, regions.
    6. **Job Title Keywords:** Look for specific roles like "architect", "director", "VP", "manager", "lead", etc.
    7. **Topic Requirements:** Identify specific technologies, domains, or subjects that are explicitly requested.
    8. **Output JSON:** Your output MUST be ONLY a single, valid JSON object with the structure below. Do not add any explanations or any follow-up questions or comments or further analysis.

    **JSON Structure:**
    {
    "intent": "A concise summary of the user's complete, synthesized goal.",
    "expertise_keywords": ["List of general subjects for semantic search, consolidated from the conversation."],
    "mandatory_criteria": {
        "job_title_contains": ["List of keywords that MUST be in the job title."],
        "topics_must_include": ["List of topics that the speaker MUST cover."],
        "centers_must_include": ["List of required locations/centers."]
    }
    }"""

        # Construct the messages payload, including history if it exists
        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            # Add the conversation history (excluding the current query which will be added separately)
            for message in conversation_history:
                messages.append(message)

        # Add the current user query
        messages.append(
            {"role": "user", "content": f"/no_think Current Query: {current_query}"}
        )

        try:
            response = await nvidia_client.generate_llm_response(
                messages
            )

            if response.success and response.content:
                try:
                    analysis = json.loads(response.content.strip())
                    logger.info(f"LLM analysis successful: {analysis}")
                    return analysis
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse LLM response as JSON: {response.content}")
                    return self._fallback_analysis(current_query)
            else:
                logger.warning(f"LLM analysis failed: {response.error}")
                return self._fallback_analysis(current_query)

        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return self._fallback_analysis(current_query)

    def _fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback analysis when LLM fails"""
        return {
            "intent": "find relevant speakers",
            "expertise_keywords": [query],
            "mandatory_criteria": {
                "job_title_contains": [],
                "topics_must_include": [],
                "centers_must_include": []
            }
        }

    async def _generate_query_embedding(self, query: str) -> Optional[List[float]]:
        """Generate embedding for the query"""

        try:
            response = await nvidia_client.generate_embeddings([query], input_type="query")  # Use "query" for search queries

            if response.success and response.embeddings:
                return response.embeddings[0]
            else:
                logger.warning(f"Failed to generate query embedding: {response.error}")
                return None

        except Exception as e:
            logger.error(f"Error generating query embedding: {e}")
            return None

    def _create_boost_factors(self, llm_analysis: Dict[str, Any]) -> Dict[str, float]:
        """Create boost factors based on LLM analysis"""

        # Base boost factors
        boost_factors = {
            'bio': 1.0,
            'speaking_topics': 1.5,
            'specializations': 1.8,
            'job_title': 1.2,
            'primary_topics': 2.0,
            'audiences': 1.0,
            'work_experience': 1.1,
            'certifications': 1.3
        }

        # Adjust based on what was found in the query
        if llm_analysis.get('technologies'):
            boost_factors['specializations'] *= 1.4
            boost_factors['work_experience'] *= 1.3

        if llm_analysis.get('audiences'):
            boost_factors['audiences'] *= 2.0
            boost_factors['job_title'] *= 1.3

        if llm_analysis.get('expertise'):
            boost_factors['speaking_topics'] *= 1.4
            boost_factors['primary_topics'] *= 1.3
            boost_factors['specializations'] *= 1.2

        if llm_analysis.get('industries'):
            boost_factors['work_experience'] *= 1.4
            boost_factors['bio'] *= 1.2

        if llm_analysis.get('roles'):
            boost_factors['job_title'] *= 1.5

        return boost_factors

    async def enhance_query_with_llm(self, query: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Use LLM to enhance and expand the query for better speaker search results considering conversation context"""

        system_prompt = """/no_think Detailed thinking off. You are a query enhancement specialist for a professional speaker search system. Your task is to expand and enrich user queries to maximize search effectiveness by clarifying and professionalizing the user's specific request. You must strictly maintain the original intent and avoid introducing unrelated concepts.

    **Core Task:**
    Your primary role is to refine, not to reinvent. Enhance the user's keywords with professional language and direct synonyms, but do not add new topics, products, or specializations that the user did not mention.

    **Enhancement Strategy:**
    1. **Preserve Original Intent**: Keep the core meaning and requirements intact
    2. **Consider Conversation Context**: If there's conversation history, synthesize the current query with previous context
    3. **Add Technical Synonyms**: Include related technical terms, frameworks, and technologies
    4. **Professional Language**: Use formal language that matches speaker bios and professional profiles
    5. **Broaden Scope Intelligently**: Add closely related topics that speakers might cover
    6. **Include Presentation Context**: Add terms related to speaking, presenting, and knowledge sharing
    7. **Handle Refinements**: If the current query is a refinement (like "only from Bangalore"), combine it with the original search intent

    **Guidelines:**
    - Transform casual language into professional terminology
    - Add industry-standard terms and acronyms
    - Include related technologies and methodologies
    - Mention presentation and communication skills when relevant
    - Focus on terms likely to appear in speaker profiles and bios
    - If there's conversation history, create a complete enhanced query that incorporates both the history and current request
    - If the new query is a refinement, and has no relation to the previous queries in the conversation history, there is no need to consider the conversation history, just enhance the query based on the current query since this implies that it is a new search.
    - Keep the enhanced query comprehensive yet focused
    - Make sure that if there is a mention of a location, specialization, topic, work experience, job title, or any other criteria, it is included in the enhanced query and is emphasized as a MUST requirement.

    **Example:**
    Input: "GPU experts proficient in AI and ML" (first query)
    Follow-up: "only from North America" (current query with history)
    Output: "Technical speakers, engineers, and researchers with expertise in Graphics Processing Units (GPUs) and their application in accelerating Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning workloads. Seeking presenters proficient in GPU-centric frameworks like CUDA, ROCm, and parallel computing paradigms MUST be based in North America."

    
    Input: "Find speakers with expertise in cloud computing and AI" (first query)
    Follow-up: "Healthcare experts based in Bangalore with experience of telemedicine" (current query with history but this follow up query has no relation to the previous queries in the conversation history)
    Output: "Healthcare speakers and experts in telemedicine and digital health MUST be based in Bangalore with experience of presenting healthcare technologies."

    Above queries are examples of how to enhance the query. Do not use them as a template for your output.

    If the case is of a follow up query, there is no need to mention the process of how you are refining the query rather just refine it and return it under 150 words.
    Return only the enhanced query text and nothing else. There is no need to add any additional text or explanation outside of the enhanced query. Even if there is a conversation history, you can use it to provide context, but do not mention it in the output. Just focus on enhancing the speaker search query."""

        # Construct messages with conversation context
        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            # Add conversation history for context
            for message in conversation_history:
                messages.append(message)

        # Add the current enhancement request
        messages.append(
            {
                "role": "user",
                "content": f"/no_think Enhance this speaker search query considering the full conversation context under 150 words: {query}. Return only the enhanced query text and nothing else. There is no need to add any additional text or explanation outside of the enhanced query. Even if there is a conversation history, you can use it to provide context, but do not mention it in the output. Just focus on enhancing the speaker search query.",
            }
        )

        try:
            response = await nvidia_client.generate_llm_response(messages, 
                max_tokens=150,  # Limit to 150 words
                temperature=0.7
            )

            if response.success and response.content:
                enhanced_query = response.content.strip()
                logger.info(f"Query enhanced from '{query}' to '{enhanced_query}'")
                return enhanced_query
            else:
                logger.warning(f"Failed to enhance query: {response.error}")
                return query

        except Exception as e:
            logger.error(f"Error enhancing query: {e}")
            return query

# Global query processor instance
query_processor = QueryProcessor()

# Utility functions
async def process_query(query: str) -> Dict[str, Any]:
    """Utility function to process a query"""
    return await query_processor.process_query(query)

async def enhance_query(query: str) -> str:
    """Utility function to enhance a query"""
    return await query_processor.enhance_query_with_llm(query)

if __name__ == "__main__":
    # Test the query processor
    async def test_query_processor():
        test_queries = [
            "Find cloud computing experts",
            "Who can speak about AI to executives?",
            "Technical speakers with DevOps experience",
            "Find speakers from healthcare industry"
        ]
        
        for query in test_queries:
            print(f"\n🔍 Testing query: {query}")
            result = await query_processor.process_query(query)
            print(f"LLM Analysis: {result['llm_analysis']}")
            print(f"Boost factors: {result['boost_factors']}")
            print(f"Has embedding: {'Yes' if result['query_embedding'] else 'No'}")
    
    asyncio.run(test_query_processor())

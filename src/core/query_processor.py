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
    
    async def process_query(self, query: str) -> Dict[str, Any]:
        """Process a natural language query and extract search parameters"""
        
        logger.info(f"Processing query: {query}")
        
        # Basic cleaning - just trim whitespace
        cleaned_query = query.strip()
        
        # STEP 1: Enhance the query with LLM for better search results
        enhanced_query = await self.enhance_query_with_llm(cleaned_query)
        logger.info(f"Enhanced query: {enhanced_query}")
        
        # STEP 2: Use enhanced query for LLM analysis
        llm_analysis = await self._analyze_query_with_llm(enhanced_query)
        
        # STEP 3: Generate query embedding using enhanced query
        query_embedding = await self._generate_query_embedding(enhanced_query)
        
        # Create search parameters
        search_params = {
            'original_query': query,
            'cleaned_query': cleaned_query,
            'enhanced_query': enhanced_query,  # Add enhanced query to results
            'llm_analysis': llm_analysis,
            'query_embedding': query_embedding,
            'boost_factors': self._create_boost_factors(llm_analysis),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Query processing complete - Primary intent: {llm_analysis.get('intent', 'unknown')}")
        
        return search_params
    
    async def _analyze_query_with_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM to analyze query and extract entities/intent"""
        
        system_prompt = """Detailed thinking off. You are a highly intelligent query analysis engine for a speaker search system. Your sole purpose is to analyze a user's query and convert it into a structured JSON object.
            **Instructions:**
            1.  **Analyze the query:** Carefully examine the user's request to understand their needs.
            2.  **Extract entities:** Populate the fields in the JSON structure below based on the query.
            3.  **Be precise:** If a specific entity (like a technology or industry) is not mentioned, leave the corresponding list empty.
            4.  **Guardrail:** Do NOT invent or infer any information not explicitly present in the query.
            5.  **Intent:** The 'intent' field should be a concise summary of the user's goal.
            6.  **Expertise:** The 'expertise' field should contain the key subjects or skills the user is looking for. If the query is broad, use the most relevant nouns and concepts.
    
            **Output Format:**
            - You MUST return ONLY a valid JSON object.
            - Do not include any explanations, markdown formatting, or any text outside of the JSON structure.
    
            **JSON Structure:**
            {
              "intent": "A concise summary of the user's goal.",
              "technologies": ["List of specific technologies, frameworks, or products mentioned."],
              "industries": ["List of industries mentioned (e.g., 'healthcare', 'finance')."],
              "audiences": ["List of audience types mentioned (e.g., 'executives', 'technical', 'developers')."],
              "expertise": ["List of key subjects, skills, or topics requested."],
              "roles": ["List of job titles or roles mentioned (e.g., 'CTO', 'engineer')."]
            }"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Query to analyze: {query}"}
        ]
        
        try:
            response = await nvidia_client.generate_llm_response(messages, max_tokens=300)
            
            if response.success and response.content:
                # Try to parse JSON response
                try:
                    analysis = json.loads(response.content.strip())
                    logger.info(f"LLM analysis successful: {analysis}")
                    return analysis
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse LLM response as JSON: {response.content}")
                    return self._fallback_analysis(query)
            else:
                logger.warning(f"LLM analysis failed: {response.error}")
                return self._fallback_analysis(query)
                
        except Exception as e:
            logger.error(f"Error in LLM analysis: {e}")
            return self._fallback_analysis(query)
    
    def _fallback_analysis(self, query: str) -> Dict[str, Any]:
        """Fallback analysis when LLM fails"""
        return {
            "intent": "find relevant speakers",
            "technologies": [],
            "industries": [],
            "audiences": [],
            "expertise": [query],  # Use the whole query as expertise
            "roles": []
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
    
    async def enhance_query_with_llm(self, query: str) -> str:
        """Use LLM to enhance and expand the query for better speaker search results"""
        
        system_prompt = """<no_think>. Detailed thinking off. You are a query enhancement specialist for a professional speaker search system. Your task is to expand and enrich user queries to maximize search effectiveness while maintaining the original intent.

    **Enhancement Strategy:**
    1. **Preserve Original Intent**: Keep the core meaning and requirements intact
    2. **Add Technical Synonyms**: Include related technical terms, frameworks, and technologies
    3. **Professional Language**: Use formal language that matches speaker bios and professional profiles
    4. **Broaden Scope Intelligently**: Add closely related topics that speakers might cover
    5. **Include Presentation Context**: Add terms related to speaking, presenting, and knowledge sharing

    **Guidelines:**
    - Transform casual language into professional terminology
    - Add industry-standard terms and acronyms
    - Include related technologies and methodologies
    - Mention presentation and communication skills when relevant
    - Keep the enhanced query under 80 words
    - Focus on terms likely to appear in speaker profiles and bios

    Keep in mind that the purpose of the entire system is to find the best speakers based on user queries. 
    The enhanced query should be comprehensive yet concise, ensuring it captures all relevant aspects of the user's request.
    Keep in mind that if there is a mention of a specific topic or speaking topic, do not modify or paraphrase it.

    **Example:**
    Input: "GPU experts with experience delivering briefings on AI topics"
    Output: "Technical speakers and experts in GPUs, CUDA, high-performance computing, and parallel processing with experience presenting on artificial intelligence, machine learning, deep learning, and neural networks to professional and technical audiences"

    Return only the enhanced query text and nothing else. There is no need to return any explanations or additional information or any follow-up."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Enhance this speaker search query: <no_think> {query}"}
        ]
        
        try:
            response = await nvidia_client.generate_llm_response(messages, max_tokens=150)
            
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
    
    def get_query_suggestions(self, partial_query: str) -> List[str]:
        """Get simple query suggestions based on partial input"""
        
        suggestions = []
        
        # Simple predefined suggestions
        common_patterns = [
            "Find speakers with expertise in",
            "Who can speak about",
            "Find experts in",
            "Speakers for executive audience",
            "Technical speakers with experience in",
            "Find speakers from industry",
            "Who has experience with"
        ]
        
        # Add relevant suggestions based on partial query
        partial_lower = partial_query.lower()
        
        if 'cloud' in partial_lower:
            suggestions.extend([
                "cloud computing experts",
                "cloud architecture specialists",
                "AWS cloud experts"
            ])
        elif 'ai' in partial_lower or 'artificial' in partial_lower:
            suggestions.extend([
                "artificial intelligence experts",
                "machine learning specialists", 
                "AI strategy speakers"
            ])
        elif 'security' in partial_lower:
            suggestions.extend([
                "cybersecurity experts",
                "security architecture specialists",
                "data security professionals"
            ])
        else:
            suggestions.extend(common_patterns[:5])
        
        return suggestions[:8]  # Return top 8 suggestions

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
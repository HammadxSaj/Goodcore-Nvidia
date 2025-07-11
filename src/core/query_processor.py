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
        
        # Use LLM to extract entities and intent
        llm_analysis = await self._analyze_query_with_llm(cleaned_query)
        
        # Generate query embedding
        query_embedding = await self._generate_query_embedding(cleaned_query)
        
        # Create search parameters
        search_params = {
            'original_query': query,
            'cleaned_query': cleaned_query,
            'llm_analysis': llm_analysis,
            'query_embedding': query_embedding,
            'boost_factors': self._create_boost_factors(llm_analysis),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Query processing complete - Primary intent: {llm_analysis.get('intent', 'unknown')}")
        
        return search_params
    
    async def _analyze_query_with_llm(self, query: str) -> Dict[str, Any]:
        """Use LLM to analyze query and extract entities/intent"""
        
        system_prompt = """You are a query analysis assistant for a speaker search system. 
        Analyze the user's query and extract relevant information.
        
        Extract:
        1. Intent (what the user is looking for)
        2. Technologies mentioned
        3. Industries mentioned  
        4. Audience types mentioned
        5. Expertise areas
        6. Job titles or roles
        
        IMPORTANT: Return ONLY a valid JSON object, no explanations, no markdown, no additional text.
        
        Use this exact structure:
        {
          "intent": "brief description of what user wants",
          "technologies": ["list", "of", "technologies"],
          "industries": ["list", "of", "industries"],
          "audiences": ["list", "of", "audience", "types"],
          "expertise": ["list", "of", "expertise", "areas"],
          "roles": ["list", "of", "job", "titles"]
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
        """Use LLM to enhance and expand the query"""
        
        system_prompt = """You are a query enhancement assistant for a speaker search system. 
        Enhance the user's query by adding relevant synonyms, related terms, and technical keywords.
        
        Rules:
        1. Keep the original meaning and intent
        2. Add relevant synonyms and related terms
        3. Include technical keywords that would help find speakers
        4. Make it more comprehensive for search
        5. Keep response under 100 words
        
        Return only the enhanced query text, nothing else."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Enhance this query: {query}"}
        ]
        
        try:
            response = await nvidia_client.generate_llm_response(messages, max_tokens=200)
            
            if response.success and response.content:
                enhanced_query = response.content.strip()
                logger.info(f"Enhanced query: {enhanced_query}")
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
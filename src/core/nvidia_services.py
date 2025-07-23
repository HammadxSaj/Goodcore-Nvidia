"""
NVIDIA AI Services Integration
Handles embedding generation, semantic search, and LLM-powered ranking
"""

import asyncio
import aiohttp
import requests
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import time
from dataclasses import dataclass
import numpy as np

from .config import config

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingResponse:
    """Response from embedding service"""
    embeddings: List[List[float]]
    usage: Dict[str, Any]
    model: str
    success: bool = True
    error: Optional[str] = None

@dataclass
class LLMResponse:
    """Response from LLM service"""
    content: str
    usage: Dict[str, Any]
    model: str
    success: bool = True
    error: Optional[str] = None

@dataclass
class RerankResponse:
    """Response from reranking service"""
    rankings: List[Dict[str, Any]]
    usage: Dict[str, Any]
    model: str
    success: bool = True
    error: Optional[str] = None

class RateLimiter:
    """Rate limiter for API requests"""
    
    def __init__(self, requests_per_minute: int = 40):
        self.requests_per_minute = requests_per_minute
        self.request_times = []
        self.lock = asyncio.Lock()
    
    async def wait_if_needed(self):
        """Wait if we're approaching rate limit"""
        async with self.lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            self.request_times = [req_time for req_time in self.request_times if now - req_time < 60]
            
            if len(self.request_times) >= self.requests_per_minute:
                # Calculate wait time
                oldest_request = min(self.request_times)
                wait_time = 60 - (now - oldest_request) + 1  # Add 1 second buffer
                
                if wait_time > 0:
                    logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                    await asyncio.sleep(wait_time)
                    # Remove the oldest request after waiting
                    self.request_times = self.request_times[1:]
            
            # Record this request
            self.request_times.append(now)

class NVIDIAServicesClient:
    """Client for NVIDIA AI services with rate limiting and service-specific ports"""

    def __init__(self):
        # Service-specific URLs
        self.base_host = config.nvidia.base_url.replace('http://', '').replace('https://', '').split(':')[0]
        self.protocol = 'http://' if 'http://' in config.nvidia.base_url else 'https://'

        # Service-specific ports
        self.embedding_url = f"{self.protocol}{self.base_host}:8090"  # Embedding service
        self.llm_url = f"{self.protocol}{self.base_host}:8070"        # LLM service  
        self.reranker_url = f"{self.protocol}{self.base_host}:8060"   # Reranker service

        self.embedding_model = config.nvidia.embedding_model
        self.llm_model = config.nvidia.llm_model
        self.reranker_model = config.nvidia.reranker_model
        self.timeout = config.nvidia.timeout
        self.max_retries = config.nvidia.max_retries

        # Rate limiter for developer account (40 requests/minute)
        self.rate_limiter = RateLimiter(requests_per_minute=35)  # Conservative limit

        self.headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        }

        # Add API key if provided
        if config.nvidia.api_key:
            self.headers['Authorization'] = f'Bearer {config.nvidia.api_key}'

        logger.info(f"Initialized NVIDIA client with service-specific ports:")
        logger.info(f"  Embedding: {self.embedding_url}")
        logger.info(f"  LLM: {self.llm_url}")  
        logger.info(f"  Reranker: {self.reranker_url}")

    async def health_check(self) -> Dict[str, Any]:
        """Check if NVIDIA services are available"""
        services_status = {}

        # Check each service individually
        services = {
            'embedding': self.embedding_url,
            'llm': self.llm_url,
            'reranker': self.reranker_url
        }

        for service_name, service_url in services.items():
            try:
                async with aiohttp.ClientSession() as session:
                    # Try a simple GET request to the base URL
                    async with session.get(
                        f"{service_url}/health",
                        headers=self.headers,
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        services_status[service_name] = {
                            'status': 'healthy' if response.status == 200 else 'unhealthy',
                            'status_code': response.status
                        }
            except Exception as e:
                services_status[service_name] = {
                    'status': 'unhealthy',
                    'error': str(e)
                }

        # Overall status
        all_healthy = all(service['status'] == 'healthy' for service in services_status.values())

        return {
            'status': 'healthy' if all_healthy else 'partial',
            'services': services_status
        }

    async def generate_embeddings(
        self, 
        texts: List[str],
        input_type: str = "passage",  # Added input_type parameter
        batch_size: int = 5  # Smaller batches for rate limiting
    ) -> EmbeddingResponse:
        """Generate embeddings for texts using NVIDIA embedding model"""

        if not texts:
            return EmbeddingResponse(
                embeddings=[],
                usage={},
                model=self.embedding_model,
                success=False,
                error="No texts provided"
            )

        logger.info(f"Generating embeddings for {len(texts)} texts with input_type: {input_type}")

        try:
            # Process in smaller batches with rate limiting
            all_embeddings = []
            total_usage = {"prompt_tokens": 0, "total_tokens": 0}

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.debug(f"Processing embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")

                # Wait for rate limit
                await self.rate_limiter.wait_if_needed()

                batch_response = await self._generate_embeddings_batch(batch, input_type)

                if not batch_response.success:
                    return batch_response

                all_embeddings.extend(batch_response.embeddings)

                # Accumulate usage stats
                for key in total_usage:
                    total_usage[key] += batch_response.usage.get(key, 0)

                # Small delay between batches
                await asyncio.sleep(0.5)

            return EmbeddingResponse(
                embeddings=all_embeddings,
                usage=total_usage,
                model=self.embedding_model,
                success=True
            )

        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return EmbeddingResponse(
                embeddings=[],
                usage={},
                model=self.embedding_model,
                success=False,
                error=str(e)
            )

    async def _generate_embeddings_batch(self, texts: List[str], input_type: str) -> EmbeddingResponse:
        """Generate embeddings for a batch of texts"""

        payload = {
            "model": self.embedding_model,
            "input": texts,
            "encoding_format": "float",
            "input_type": input_type  # Added input_type parameter
        }

        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{self.embedding_url}/v1/embeddings",
                        json=payload,
                        headers=self.headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:

                        if response.status == 200:
                            result = await response.json()

                            # Extract embeddings from response
                            embeddings = [item['embedding'] for item in result['data']]

                            return EmbeddingResponse(
                                embeddings=embeddings,
                                usage=result.get('usage', {}),
                                model=self.embedding_model,
                                success=True
                            )
                        else:
                            error_text = await response.text()
                            logger.warning(f"Embedding request failed (attempt {attempt + 1}): {response.status} - {error_text}")

                            if attempt == self.max_retries - 1:
                                return EmbeddingResponse(
                                    embeddings=[],
                                    usage={},
                                    model=self.embedding_model,
                                    success=False,
                                    error=f"HTTP {response.status}: {error_text}"
                                )

                            # Exponential backoff
                            await asyncio.sleep(2 ** attempt)

            except Exception as e:
                logger.warning(f"Embedding request exception (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return EmbeddingResponse(
                        embeddings=[],
                        usage={},
                        model=self.embedding_model,
                        success=False,
                        error=str(e)
                    )
                await asyncio.sleep(2 ** attempt)

        return EmbeddingResponse(
            embeddings=[],
            usage={},
            model=self.embedding_model,
            success=False,
            error="Max retries exceeded"
        )

    async def generate_llm_response(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> LLMResponse:
        """Generate LLM response using NVIDIA LLM model or Ollama model"""

        # Check if we should use Ollama
        if config.nvidia.use_ollama_for_llm:
            endpoint = f"{config.nvidia.ollama_base_url}/api/chat"
            logger.info(f"Routing LLM request to Ollama: {endpoint}")
        elif config.nvidia.use_vllm_for_llm:
            endpoint = f"{config.nvidia.vllm_base_url}/v1/chat/completions"
            logger.info(f"Routing LLM request to vLLM: {endpoint}")
        else:
            endpoint = f"{self.llm_url}/v1/chat/completions"
            logger.info(f"Routing LLM request to NVIDIA NIM: {endpoint}")

        # Wait for rate limit
        await self.rate_limiter.wait_if_needed()

        # Prepare payload based on the service
        if config.nvidia.use_ollama_for_llm:
            # Ollama API format
            payload = {
                "model": "qwen3:32b",
                "messages": messages,
                "stream": False,
                "options": {
                    "temperature": temperature if temperature is not None else 0.7,
                }
            }

            #print(f"🔍 DEBUG: Ollama LLM Payload: {json.dumps(payload, indent=2)}")
            # Note: Ollama doesn't use max_tokens in the same way as OpenAI
            if max_tokens is not None:
                payload["options"]["num_predict"] = max_tokens

        elif config.nvidia.use_vllm_for_llm:
            # vLLM API format
            payload = {
                "model": "Qwen/Qwen3-14B-AWQ",
                "messages": messages,
                "temperature": temperature if temperature is not None else 0.7,
                "stream": False
            }

            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
                
            #print(f"🔍 DEBUG: vLLM LLM Payload: {json.dumps(payload, indent=2)}")
        else:
            # NVIDIA NIM API format
            payload = {
                "model": self.llm_model,
                "messages": messages,
                "stream": False
            }
            # Only add parameters if they're provided
            if max_tokens is not None:
                payload["max_tokens"] = max_tokens
            if temperature is not None:
                payload["temperature"] = temperature

        logger.info(f"Generating LLM response with {len(messages)} messages using model: {payload['model']}")

        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        endpoint,
                        json=payload,
                        headers=self.headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:

                        if response.status == 200:
                            result = await response.json()

                            # Handle different response structures
                            if config.nvidia.use_ollama_for_llm:
                                # Ollama's response structure
                                content = result.get("message", {}).get("content", "")
                                usage = {
                                    "prompt_tokens": result.get("prompt_eval_count", 0),
                                    "completion_tokens": result.get("eval_count", 0),
                                    "total_tokens": result.get("prompt_eval_count", 0) + result.get("eval_count", 0)
                                }
                            else:
                                # NVIDIA's response structure
                                content = result['choices'][0]['message']['content']
                                usage = result.get('usage', {})

                            #make sure to remove <think> </think> tags
                            content = content.replace("<think>", "").replace("</think>", "").strip()

                            return LLMResponse(
                                content=content,
                                usage=usage,
                                model=payload['model'],
                                success=True
                            )
                        else:
                            error_text = await response.text()
                            logger.warning(f"LLM request failed (attempt {attempt + 1}): {response.status} - {error_text}")

                            if attempt == self.max_retries - 1:
                                return LLMResponse(
                                    content="",
                                    usage={},
                                    model=payload['model'],
                                    success=False,
                                    error=f"HTTP {response.status}: {error_text}"
                                )

                            await asyncio.sleep(2 ** attempt)

            except Exception as e:
                logger.warning(f"LLM request exception (attempt {attempt + 1}): {e}")
                if attempt == self.max_retries - 1:
                    return LLMResponse(
                        content="",
                        usage={},
                        model=payload['model'],
                        success=False,
                        error=str(e)
                    )
                await asyncio.sleep(2 ** attempt)

        return LLMResponse(
            content="",
            usage={},
            model=payload['model'],
            success=False,
            error="Max retries exceeded"
        )

    # async def rerank_results(
    #     self,
    #     query: str,
    #     candidates: List[Dict[str, Any]],
    #     top_k: int = 10
    # ) -> RerankResponse:
    #     """Rerank search results using NVIDIA reranking model"""

    #     if not candidates:
    #         return RerankResponse(
    #             rankings=[],
    #             usage={},
    #             model=self.reranker_model,
    #             success=False,
    #             error="No candidates provided"
    #         )

    #     # Wait for rate limit
    #     await self.rate_limiter.wait_if_needed()

    #     # Prepare candidates for reranking - using passages format as per NVIDIA docs
    #     passages = []
    #     for candidate in candidates:
    #         # Use searchable_text or create a text representation
    #         doc_text = candidate.get('searchable_text', '')
    #         if not doc_text:
    #             doc_text = f"{candidate.get('name', '')} {candidate.get('job_title', '')} {candidate.get('bio', '')}"
    #         # passages.append({"text": doc_text})
    #         passages.append(doc_text)

    #     # Format payload according to NVIDIA NIM documentation
    #     # payload = {
    #     #     "model": self.reranker_model,
    #     #     "query": {"text": query},  # Query must be an object with "text" field
    #     #     "passages": passages,      # Use "passages" instead of "documents"                                                                                            
    #     #     "top_k": min(top_k, len(passages)),
    #     #     "truncate": "END"  # Add truncate parameter as shown in docs
    #     # }

    #     payload = {
    #         "model": "Qwen/Qwen3-Reranker-0.6B",  # Use the specific reranker model
    #         "query": query,
    #         "documents": passages,
    #         "top_k": min(top_k, len(passages)),
    #     }                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           

    #     logger.info(f"Reranking {len(candidates)} candidates")
    #     logger.debug(f"Rerank payload: {json.dumps(payload, indent=2)}")

    #     for attempt in range(self.max_retries):
    #         try:
    #             async with aiohttp.ClientSession() as session:
    #                 async with session.post(
    #                     # f"{self.reranker_url}/v1/ranking",
    #                     f"http://{self.base_host}:8060/v2/rerank",
    #                     json=payload,
    #                     headers=self.headers,
    #                     timeout=aiohttp.ClientTimeout(total=self.timeout)
    #                 ) as response:

    #                     if response.status == 200:
    #                         result = await response.json()
    #                         # DEBUG: Print detailed reranking response
    #                         print(f"\n🔍 DEBUG: Reranking API Response:")
    #                         print(f"Response status: {response.status}")
    #                         print(f"Response keys: {list(result.keys())}")
    #                         print(f"Full response: {json.dumps(result, indent=2)}")

    #                         logger.debug(f"Rerank response: {json.dumps(result, indent=2)}")

    #                         ranked_candidates = []

    #                         for item in result.get('results', []):
    #                             original_index = item['index']
    #                             candidate = candidates[original_index].copy()
    #                             candidate['rerank_score'] = item['relevance_score']
    #                             ranked_candidates.append(candidate)

    #                         # Check response structure and handle different formats
    #                         # rankings = []

    #                         # # Try different possible response structures
    #                         # if 'rankings' in result:
    #                         #     # Handle NVIDIA NIM reranker response format with logits
    #                         #     for i, item in enumerate(result['rankings']):
    #                         #         original_idx = item['index']
    #                         #         candidate = candidates[original_idx].copy()

    #                         #         # Convert logit to normalized score (0-1 range)
    #                         #         logit = item.get('logit', 0.0)
    #                         #         # Use sigmoid function to convert logit to probability
    #                         #         import math
    #                         #         rerank_score = 1.0 / (1.0 + math.exp(-logit))

    #                         #         candidate['rerank_score'] = rerank_score
    #                         #         candidate['rerank_position'] = i + 1
    #                         #         rankings.append(candidate)
    #                         # elif 'data' in result:
    #                         #     # Another possible format
    #                         #     for i, item in enumerate(result['data']):
    #                         #         original_idx = item.get('index', i)
    #                         #         if original_idx < len(candidates):
    #                         #             candidate = candidates[original_idx].copy()
    #                         #             candidate['rerank_score'] = item.get('relevance_score', item.get('score', 0.0))
    #                         #             candidate['rerank_position'] = len(rankings) + 1
    #                         #             rankings.append(candidate)
    #                         # else:
    #                         #     # Fallback: return original order with dummy scores
    #                         #     logger.warning(f"Unexpected rerank response format. Keys: {list(result.keys())}")
    #                         #     for i, candidate in enumerate(candidates[:top_k]):
    #                         #         candidate_copy = candidate.copy()
    #                         #         candidate_copy['rerank_score'] = 1.0 - (i * 0.1)  # Dummy decreasing scores
    #                         #         candidate_copy['rerank_position'] = i + 1
    #                         #         rankings.append(candidate_copy)

    #                         #TEI part here

    #                         # ranked_candidates = []
    #                         # for item in result:
    #                         #     original_index = item['index']
    #                         #     candidate = candidates[original_index].copy()
    #                         #     candidate['rerank_score'] = item['score']
    #                         #     ranked_candidates.append(candidate)

    #                         #     # Sort by the new score, as TEI might not guarantee order
    #                         #     ranked_candidates.sort(key=lambda x: x['rerank_score'], reverse=True)


    #                         return RerankResponse(
    #                             rankings=ranked_candidates[:top_k],
    #                             usage=result.get('usage', {}),
    #                             model=self.reranker_model,
    #                             success=True
    #                         )
    #                     else:
    #                         error_text = await response.text()
    #                         logger.warning(f"Rerank request failed (attempt {attempt + 1}): {response.status} - {error_text}")

    #                         if attempt == self.max_retries - 1:
    #                             # Return original candidates without reranking
    #                             # return RerankResponse(
    #                             #     rankings=candidates[:top_k],
    #                             #     usage={},
    #                             #     model=self.reranker_model,
    #                             #     success=False,
    #                             #     error=f"HTTP {response.status}: {error_text}"
    #                             # )

    #                             return RerankResponse(rankings=[], usage={}, model=self.reranker_model, success=False, error=f"HTTP {response.status}: {error_text}")
    #                         await asyncio.sleep(2 ** attempt)

    #         except Exception as e:
    #             logger.warning(f"Rerank request exception (attempt {attempt + 1}): {e}")
    #             logger.debug(f"Exception details: {type(e).__name__}: {str(e)}")

    #             if attempt == self.max_retries - 1:
    #                 # return RerankResponse(
    #                 #     rankings=candidates[:top_k],
    #                 #     usage={},
    #                 #     model=self.reranker_model,
    #                 #     success=False,
    #                 #     error=str(e)
    #                 # )

    #                 return RerankResponse(rankings=[], usage={}, model=self.reranker_model, success=False, error=str(e))
    #             await asyncio.sleep(2 ** attempt)

    #     # return RerankResponse(
    #     #     rankings=candidates[:top_k],
    #     #     usage={},
    #     #     model=self.reranker_model,
    #     #     success=False,
    #     #     error="Max retries exceeded"
    #     # )

    #     return RerankResponse(rankings=[], usage={}, model=self.reranker_model, success=False, error="Max retries exceeded")

    async def rerank_results(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
        top_k: int = 10
    ) -> RerankResponse:
        """Reranks search results using a vLLM-hosted reranking model."""
        if not candidates:
            return RerankResponse(
                rankings=[],
                usage={},
                model=self.reranker_model,
                success=False,
                error="No candidates provided"
            )

        await self.rate_limiter.wait_if_needed()

        documents = [
            candidate.get('searchable_text') or f"{candidate.get('name', '')} {candidate.get('job_title', '')} {candidate.get('bio', '')}".strip()
            for candidate in candidates
        ]

        # Payload for the vLLM /rerank endpoint (Cohere-compatible)
        payload = {
            "model": self.reranker_model,
            "query": query,
            "documents": documents,
            "top_n": min(top_k, len(documents)),  # Use top_n for Cohere compatibility
            "return_documents": False  # More efficient as we already have the docs
        }

        logger.info(f"Reranking {len(candidates)} candidates for query: '{query}'")
        logger.debug(f"Rerank payload: {json.dumps(payload, indent=2)}")

        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    # Using the /v1/rerank endpoint, which is designed for this task
                    async with session.post(
                        f"http://{self.base_host}:8060/v1/rerank",
                        json=payload,
                        headers=self.headers,
                        timeout=aiohttp.ClientTimeout(total=self.timeout)
                    ) as response:
                        response_json = await response.json()

                        if response.status == 200:
                            logger.debug(f"Rerank response: {json.dumps(response_json, indent=2)}")

                            ranked_candidates = []
                            # The response contains a 'results' list with 'index' and 'relevance_score'
                            results = response_json.get('results', [])
                            for item in results:
                                original_index = item['index']
                                candidate = candidates[original_index].copy()
                                candidate['rerank_score'] = item['relevance_score']
                                ranked_candidates.append(candidate)

                            # The API returns the results already sorted by relevance
                            return RerankResponse(
                                rankings=ranked_candidates,
                                usage=response_json.get('usage', {}),
                                model=response_json.get('model', self.reranker_model),
                                success=True
                            )
                        else:
                            error_message = response_json.get('detail', await response.text())
                            logger.warning(f"Rerank request failed (attempt {attempt + 1}/{self.max_retries}): {response.status} - {error_message}")
                            return RerankResponse(rankings=[], usage={}, model=self.reranker_model, success=False, error=f"HTTP {response.status}: {error_message}")

            except aiohttp.ClientConnectorError as e:
                logger.error(f"Rerank connection error (attempt {attempt + 1}/{self.max_retries}): {e}")
            except Exception as e:
                logger.warning(f"Rerank request exception (attempt {attempt + 1}/{self.max_retries}): {e}")

            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)

        logger.error("Reranking failed after max retries.")
        return RerankResponse(rankings=[], usage={}, model=self.reranker_model, success=False, error="Max retries exceeded")

    def calculate_similarity(
        self,
        embedding1: List[float],
        embedding2: List[float]
    ) -> float:
        """Calculate cosine similarity between two embeddings"""

        try:
            # Convert to numpy arrays
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            # Calculate cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return float(similarity)

        except Exception as e:
            logger.error(f"Error calculating similarity: {e}")
            return 0.0

    async def test_all_services(self) -> Dict[str, Any]:
        """Test all NVIDIA services"""

        logger.info("Testing NVIDIA services...")

        results = {
            'health_check': await self.health_check(),
            'embedding_test': None,
            'llm_test': None,
            'rerank_test': None
        }

        # Test embedding service
        try:
            embedding_response = await self.generate_embeddings(
                ["This is a test sentence for embedding generation."],
                input_type="passage"  # Added input_type parameter
            )
            results['embedding_test'] = {
                'success': embedding_response.success,
                'model': embedding_response.model,
                'embedding_dimension': len(embedding_response.embeddings[0]) if embedding_response.embeddings else 0,
                'error': embedding_response.error
            }
        except Exception as e:
            results['embedding_test'] = {
                'success': False,
                'error': str(e)
            }

        # Test LLM service
        try:
            llm_response = await self.generate_llm_response([
                {"role": "user", "content": "Hello, please respond with 'Service test successful'."}
            ])
            results['llm_test'] = {
                'success': llm_response.success,
                'model': llm_response.model,
                'response_length': len(llm_response.content),
                'error': llm_response.error
            }
        except Exception as e:
            results['llm_test'] = {
                'success': False,
                'error': str(e)
            }

        # Test reranking service
        try:
            test_candidates = [
                {'name': 'John Doe', 'job_title': 'Engineer', 'searchable_text': 'John is a software engineer'},
                {'name': 'Jane Smith', 'job_title': 'Manager', 'searchable_text': 'Jane is a project manager'}
            ]

            rerank_response = await self.rerank_results(
                "software engineer",
                test_candidates,
                top_k=2
            )
            results['rerank_test'] = {
                'success': rerank_response.success,
                'model': rerank_response.model,
                'ranked_count': len(rerank_response.rankings),
                'error': rerank_response.error
            }
        except Exception as e:
            results['rerank_test'] = {
                'success': False,
                'error': str(e)
            }

        return results

# Global client instance
nvidia_client = NVIDIAServicesClient()

# Updated utility functions with input_type parameter
async def generate_embeddings(texts: List[str], input_type: str = "passage") -> EmbeddingResponse:
    """Utility function to generate embeddings"""
    return await nvidia_client.generate_embeddings(texts, input_type)


async def generate_llm_response(
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
) -> LLMResponse:
    """Utility function to generate LLM response"""
    return await nvidia_client.generate_llm_response(messages, max_tokens, temperature)


async def rerank_results(query: str, candidates: List[Dict[str, Any]]) -> RerankResponse:
    """Utility function to rerank results"""
    return await nvidia_client.rerank_results(query, candidates)

if __name__ == "__main__":
    # Test the NVIDIA services
    async def test_services():
        results = await nvidia_client.test_all_services()
        print("🧪 NVIDIA Services Test Results:")
        print(json.dumps(results, indent=2))
    
    asyncio.run(test_services())

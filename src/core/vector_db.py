"""
Vector Database Management using ChromaDB
Handles embedding storage, similarity search, and metadata filtering
"""

import asyncio
import chromadb
from chromadb.config import Settings
import json
import logging
from typing import List, Dict, Any, Optional, Tuple
import uuid
from datetime import datetime
import numpy as np
from pathlib import Path

from .config import config
from .nvidia_services import nvidia_client

logger = logging.getLogger(__name__)

class VectorDatabase:
    """Vector database manager using ChromaDB for speaker embeddings"""

    def __init__(self):
        # Initialize ChromaDB client
        self.db_path = Path(config.vectordb.persist_directory)
        self.db_path.mkdir(parents=True, exist_ok=True)

        # Create ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        # Collection name for speakers
        self.collection_name = "speakers"
        self.collection = None

        # Embedding dimension (NVIDIA NV-Embed-QA typically uses 1024)
        self.embedding_dimension = 1024

        logger.info(f"Initialized Vector DB at: {self.db_path}")

    async def initialize_collection(self) -> bool:
        """Initialize or get the speakers collection"""

        try:
            # Try to get existing collection
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=None  # We'll handle embeddings manually
                )
                logger.info(f"Using existing collection: {self.collection_name}")

            except Exception:
                # Create new collection if it doesn't exist
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=None,  # We'll handle embeddings manually
                    metadata={"description": "Speaker embeddings for semantic search"}
                )
                logger.info(f"Created new collection: {self.collection_name}")

            # Get collection info
            count = self.collection.count()
            logger.info(f"Collection contains {count} speaker embeddings")

            return True

        except Exception as e:
            logger.error(f"Failed to initialize collection: {e}")
            return False

    async def add_speakers(self, speakers_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Add speakers to the vector database with embeddings"""

        if not self.collection:
            await self.initialize_collection()

        if not speakers_data:
            return {"success": False, "error": "No speakers data provided"}

        logger.info(f"Adding {len(speakers_data)} speakers to vector database...")

        # Prepare data for batch processing
        ids = []
        embeddings = []
        documents = []
        metadatas = []

        # Generate embeddings for all speakers
        texts_to_embed = []
        speaker_indices = []

        for i, speaker in enumerate(speakers_data):
            # Use searchable_text or create it
            searchable_text = speaker.get('searchable_text', '')
            if not searchable_text:
                # Create searchable text from speaker data
                searchable_text = self._create_searchable_text(speaker)

            # Truncate text to fit within token limits
            truncated_text = self._truncate_text_for_embedding(searchable_text)

            texts_to_embed.append(truncated_text)
            speaker_indices.append(i)

        # Generate embeddings in batches
        try:
            embedding_response = await nvidia_client.generate_embeddings(
                texts_to_embed, 
                input_type="passage"
            )

            if not embedding_response.success:
                return {
                    "success": False,
                    "error": f"Failed to generate embeddings: {embedding_response.error}"
                }

            # Process each speaker
            for i, speaker in enumerate(speakers_data):
                speaker_id = speaker.get('id') or str(uuid.uuid4())

                # Prepare embedding
                speaker_embedding = embedding_response.embeddings[i]

                # Prepare document text (what gets returned in search)
                doc_text = texts_to_embed[i]

                # Prepare metadata (filterable fields)
                metadata = self._prepare_metadata(speaker)

                ids.append(speaker_id)
                embeddings.append(speaker_embedding)
                documents.append(doc_text)
                metadatas.append(metadata)

            # Add to ChromaDB in batch
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )

            logger.info(f"Successfully added {len(speakers_data)} speakers to vector database")

            return {
                "success": True,
                "added_count": len(speakers_data),
                "embedding_usage": embedding_response.usage
            }

        except Exception as e:
            logger.error(f"Error adding speakers to vector database: {e}")
            return {"success": False, "error": str(e)}

    def _truncate_text_for_embedding(self, text: str, max_tokens: int = 450) -> str:
        """Truncate text to fit within token limits"""
        # Rough estimation: 1 token ≈ 4 characters (conservative)
        max_chars = max_tokens * 4

        if len(text) <= max_chars:
            return text

        # Truncate at word boundary
        truncated = text[:max_chars]
        last_space = truncated.rfind(' ')
        if last_space > max_chars * 0.8:  # If we found a space in the last 20%
            truncated = truncated[:last_space]

        return truncated + "..."

    async def search_speakers(
        self,
        query_embedding: List[float],
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 20
    ) -> Dict[str, Any]:
        """Search for speakers using vector similarity"""

        if not self.collection:
            await self.initialize_collection()

        try:
            # Prepare where clause for filtering
            where_clause = None
            if filters:
                where_clause = self._build_where_clause(filters)

            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=limit,
                where=where_clause,
                include=["embeddings", "documents", "metadatas", "distances"]
            )

            # Process results
            candidates = []
            print(f"\n🔍 DEBUG: Raw ChromaDB results:")
            print(f"Distances: {results.get('distances', [])}")
            print(f"IDs: {results.get('ids', [])}")
            if results['distances'] and results['distances'][0]:
                print(f"First 3 distances: {results['distances'][0][:3]}")
                print(f"Distance type: {type(results['distances'][0][0]) if results['distances'][0] else 'None'}") 
            if results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    candidate = {
                        'id': results['ids'][0][i],
                        'similarity_score': max(0.0, 1.0 / (1.0 +(results['distances'][0][i]))),
                        'document': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'searchable_text': results['documents'][0][i]
                    }

                    # Add speaker fields from metadata
                    for key, value in results['metadatas'][0][i].items():
                        if key not in candidate:
                            candidate[key] = value

                    candidates.append(candidate)

            logger.info(f"Vector search found {len(candidates)} candidates")

            return {
                "success": True,
                "candidates": candidates,
                "total_found": len(candidates)
            }

        except Exception as e:
            logger.error(f"Error searching speakers: {e}")
            return {"success": False, "error": str(e), "candidates": []}

    async def hybrid_search(
        self,
        query_embedding: List[float],
        query_text: str,
        filters: Optional[Dict[str, Any]] = None,
        vector_weight: float = 0.7,
        limit: int = 20
    ) -> Dict[str, Any]:
        """Perform hybrid vector + rerank search with post-filtering for mandatory criteria"""
        
        try:
            # Separate mandatory criteria from basic filters
            mandatory_criteria = {}
            basic_filters = {}
            
            if filters:
                mandatory_criteria = {
                    "job_title_contains": filters.get("job_title_contains", []),
                    "topics_must_include": filters.get("topics_must_include", []),
                    "centers_must_include": filters.get("centers_must_include", [])
                }
                
                # Keep only basic filters that ChromaDB supports
                for key, value in filters.items():
                    if key not in ["job_title_contains", "topics_must_include", "centers_must_include"]:
                        basic_filters[key] = value

            # Build ChromaDB-compatible where clause (only basic filters)
            where_clause = self._build_basic_where_clause(basic_filters)
            
            logger.info(f"Vector search with basic filters: {where_clause}")
            logger.info(f"Post-filtering with mandatory criteria: {mandatory_criteria}")

            # Step 1: Vector search with basic filters only
            collection = self.client.get_collection(name=self.collection_name)
            
            # Get more results to account for post-filtering
            search_limit = limit * 3  # Get 3x more results for post-filtering
            
            results = collection.query(
                query_embeddings=[query_embedding],
                where=where_clause,
                n_results=search_limit,
                include=["documents", "metadatas", "distances"]
            )

            if not results['documents'] or not results['documents'][0]:
                logger.info("No documents found in vector search")
                return {
                    "success": True,
                    "candidates": [],
                    "total_found": 0,
                    "search_type": "no_results"
                }

            # Convert to candidates with similarity scores
            documents = results['documents'][0]
            metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
            distances = results['distances'][0] if results['distances'] else [0.0] * len(documents)
            ids = results['ids'][0] if results['ids'] else [str(i) for i in range(len(documents))]

            candidates = []
            for i, doc in enumerate(documents):
                metadata = metadatas[i] if i < len(metadatas) else {}
                distance = distances[i] if i < len(distances) else 0.0
                doc_id = ids[i] if i < len(ids) else str(i)
                
                # Convert distance to similarity (assuming cosine distance)
                similarity = 1.0 - distance
                
                candidate = {
                    'id': doc_id,
                    'document': doc,
                    'metadata': metadata,
                    'similarity_score': similarity,
                    **metadata  # Flatten metadata into candidate
                }
                candidates.append(candidate)

            logger.info(f"Vector search found {len(candidates)} candidates before post-filtering")

            # Step 2: Apply mandatory criteria post-filtering
            if any(mandatory_criteria.get(key) for key in mandatory_criteria.keys()):
                candidates = self._apply_mandatory_filters(candidates, mandatory_criteria)
                logger.info(f"After mandatory filtering: {len(candidates)} candidates remain")

            # Step 3: Rerank the filtered candidates
            if candidates:
                try:
                    rerank_response = await nvidia_client.rerank_results(
                        query=query_text,
                        candidates=candidates,
                        top_k=limit
                    )

                    if rerank_response.success:
                        # Combine vector similarity and rerank scores
                        for candidate in rerank_response.rankings:
                            vector_score = candidate.get('similarity_score', 0.0)
                            rerank_score = candidate.get('rerank_score', 0.0)

                            # Weighted combination
                            candidate['hybrid_score'] = (
                                vector_weight * vector_score + 
                                (1 - vector_weight) * rerank_score
                            )

                        # Sort by hybrid score
                        rerank_response.rankings.sort(
                            key=lambda x: x.get('hybrid_score', 0.0), 
                            reverse=True
                        )

                        return {
                            "success": True,
                            "candidates": rerank_response.rankings[:limit],
                            "total_found": len(rerank_response.rankings),
                            "search_type": "hybrid_with_mandatory_filtering"
                        }
                    else:
                        logger.warning(f"Reranking failed, using filtered vector results: {rerank_response.error}")
                        return {
                            "success": True,
                            "candidates": candidates[:limit],
                            "total_found": len(candidates),
                            "search_type": "vector_with_mandatory_filtering"
                        }

                except Exception as e:
                    logger.error(f"Error in reranking: {e}")
                    return {
                        "success": True,
                        "candidates": candidates[:limit],
                        "total_found": len(candidates),
                        "search_type": "vector_with_mandatory_filtering"
                    }

            return {
                "success": True,
                "candidates": [],
                "total_found": 0,
                "search_type": "no_results_after_filtering"
            }

        except Exception as e:
            logger.error(f"Error in hybrid search: {e}")
            return {
                "success": False,
                "candidates": [],
                "total_found": 0,
                "search_type": "error",
                "error": str(e)
            }

    def _build_basic_where_clause(self, filters: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Build ChromaDB where clause for basic filters only (no mandatory criteria)"""
        if not filters:
            return None

        and_conditions = []

        # Handle text filters (exact match only)
        for field in ['location', 'company']:
            if filters.get(field):
                and_conditions.append({field: {"$eq": filters[field]}})

        # Handle numeric filters
        if filters.get('min_experience'):
            and_conditions.append({'years_experience': {"$gte": float(filters['min_experience'])}})

        if filters.get('max_experience'):
            and_conditions.append({'years_experience': {"$lte": float(filters['max_experience'])}})

        # Handle list filters (exact match only)
        for field in ['speaking_topics', 'specializations', 'audiences']:
            if filters.get(field):
                and_conditions.append({field: {"$eq": filters[field]}})

        if not and_conditions:
            return None

        if len(and_conditions) == 1:
            return and_conditions[0]

        return {"$and": and_conditions}

    def _apply_mandatory_filters(self, candidates: List[Dict], mandatory_criteria: Dict[str, Any]) -> List[Dict]:
        """Apply mandatory criteria filtering in Python"""
        filtered_candidates = []
        
        for candidate in candidates:
            # Check job title requirements
            if mandatory_criteria.get("job_title_contains"):
                job_title = candidate.get("job_title", "").lower()
                job_title_match = any(
                    keyword.lower() in job_title 
                    for keyword in mandatory_criteria["job_title_contains"]
                )
                if not job_title_match:
                    continue
            
            # Check topic requirements
            if mandatory_criteria.get("topics_must_include"):
                # Collect all topic-related fields
                speaking_topics = candidate.get("speaking_topics", "").lower()
                specializations = candidate.get("specializations", "").lower()
                topics_general = candidate.get("topics_general", "").lower()
                all_topics = f"{speaking_topics} {specializations} {topics_general}"
                
                # Check if ANY of the required topics are present
                topic_match = any(
                    topic.lower() in all_topics 
                    for topic in mandatory_criteria["topics_must_include"]
                )
                if not topic_match:
                    continue
            
            # Check center/location requirements
            if mandatory_criteria.get("centers_must_include"):
                centers = candidate.get("centers", "").lower()
                location = candidate.get("location", "").lower()
                all_locations = f"{centers} {location}"
                
                location_match = any(
                    center.lower() in all_locations 
                    for center in mandatory_criteria["centers_must_include"]
                )
                if not location_match:
                    continue
            
            # If we reach here, candidate passed all mandatory criteria
            filtered_candidates.append(candidate)
        
        return filtered_candidates

    def _create_searchable_text(self, speaker: Dict[str, Any]) -> str:
        """Create searchable text from speaker data"""

        text_parts = []

        # Add basic info
        if speaker.get('name'):
            text_parts.append(f"Name: {speaker['name']}")

        if speaker.get('job_title'):
            text_parts.append(f"Job Title: {speaker['job_title']}")

        if speaker.get('bio'):
            text_parts.append(f"Bio: {speaker['bio']}")

        # Add speaking topics
        if speaker.get('speaking_topics'):
            topics = speaker['speaking_topics']
            if isinstance(topics, list):
                text_parts.append(f"Speaking Topics: {', '.join(topics)}")
            else:
                text_parts.append(f"Speaking Topics: {topics}")

        # Add specializations
        if speaker.get('specializations'):
            specs = speaker['specializations']
            if isinstance(specs, list):
                text_parts.append(f"Specializations: {', '.join(specs)}")
            else:
                text_parts.append(f"Specializations: {specs}")

        # Add work experience
        if speaker.get('work_experience'):
            text_parts.append(f"Experience: {speaker['work_experience']}")

        # Add certifications
        if speaker.get('certifications'):
            certs = speaker['certifications']
            if isinstance(certs, list):
                text_parts.append(f"Certifications: {', '.join(certs)}")
            else:
                text_parts.append(f"Certifications: {certs}")

        return " | ".join(text_parts)

    def _prepare_metadata(self, speaker: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB (must be simple types)"""

        metadata = {}

        # Add simple fields
        for field in ['name', 'job_title', 'company', 'location', 'bio']:
            if speaker.get(field):
                metadata[field] = str(speaker[field])[:500]  # Limit length

        # Add numeric fields
        for field in ['years_experience']:
            if speaker.get(field) is not None:
                metadata[field] = float(speaker[field])

        # Convert lists to comma-separated strings
        for field in ['speaking_topics', 'specializations', 'certifications', 'audiences']:
            if speaker.get(field):
                if isinstance(speaker[field], list):
                    metadata[field] = ', '.join(str(x) for x in speaker[field])
                else:
                    metadata[field] = str(speaker[field])

        # Add timestamp
        metadata['added_at'] = datetime.now().isoformat()

        return metadata

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector database"""

        if not self.collection:
            await self.initialize_collection()

        try:
            count = self.collection.count()

            # Get some sample data to understand the structure
            sample = self.collection.peek(limit=5)

            return {
                "success": True,
                "total_speakers": count,
                "collection_name": self.collection_name,
                "embedding_dimension": self.embedding_dimension,
                "sample_ids": sample.get('ids', [])[:3] if sample else [],
                "db_path": str(self.db_path)
            }

        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"success": False, "error": str(e)}

    async def delete_speaker(self, speaker_id: str) -> bool:
        """Delete a speaker from the vector database"""

        if not self.collection:
            await self.initialize_collection()

        try:
            self.collection.delete(ids=[speaker_id])
            logger.info(f"Deleted speaker {speaker_id} from vector database")
            return True

        except Exception as e:
            logger.error(f"Error deleting speaker {speaker_id}: {e}")
            return False

    async def update_speaker(self, speaker_id: str, speaker_data: Dict[str, Any]) -> bool:
        """Update a speaker in the vector database"""

        # For ChromaDB, we need to delete and re-add
        deleted = await self.delete_speaker(speaker_id)
        if not deleted:
            return False

        # Add the updated speaker
        result = await self.add_speakers([speaker_data])
        return result["success"]

    async def clear_collection(self) -> bool:
        """Clear all data from the collection"""

        try:
            if self.collection:
                # Delete the collection
                self.client.delete_collection(self.collection_name)
                logger.info(f"Cleared collection: {self.collection_name}")

            # Reinitialize
            await self.initialize_collection()
            return True

        except Exception as e:
            logger.error(f"Error clearing collection: {e}")
            return False

# Global vector database instance
vector_db = VectorDatabase()

# Utility functions
async def initialize_vector_db() -> bool:
    """Initialize the vector database"""
    return await vector_db.initialize_collection()

async def add_speakers_to_db(speakers: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Add speakers to the vector database"""
    return await vector_db.add_speakers(speakers)

async def search_speakers_by_embedding(
    query_embedding: List[float],
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """Search speakers by embedding"""
    return await vector_db.search_speakers(query_embedding, filters, limit)

async def hybrid_search_speakers(
    query_embedding: List[float],
    query_text: str,
    filters: Optional[Dict[str, Any]] = None,
    limit: int = 20
) -> Dict[str, Any]:
    """Perform hybrid search"""
    return await vector_db.hybrid_search(query_embedding, query_text, filters, limit)

if __name__ == "__main__":
    # Test the vector database
    async def test_vector_db():
        print("🧪 Testing Vector Database...")
        
        # Initialize
        success = await vector_db.initialize_collection()
        print(f"Initialization: {'✅' if success else '❌'}")
        
        # Get stats
        stats = await vector_db.get_collection_stats()
        print(f"Stats: {stats}")
    
    asyncio.run(test_vector_db())

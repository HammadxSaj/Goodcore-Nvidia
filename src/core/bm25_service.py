"""
BM25 service for exact keyword matching in speaker search
"""

import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from rank_bm25 import BM25Okapi
import numpy as np

logger = logging.getLogger(__name__)


class BM25Service:
    """BM25 service for keyword-based speaker search"""

    def __init__(self):
        self.bm25_index: Optional[BM25Okapi] = None
        self.speaker_corpus: List[List[str]] = []
        self.speaker_ids: List[str] = []
        self.speaker_data: List[Dict[str, Any]] = []

    def create_index(self, speaker_profiles: List[Dict[str, Any]]) -> bool:
        """Create BM25 index from speaker profiles"""
        try:
            logger.info(f"Creating BM25 index for {len(speaker_profiles)} speakers...")

            self.speaker_corpus = []
            self.speaker_ids = []
            self.speaker_data = []

            for speaker in speaker_profiles:
                # Get or create searchable text
                searchable_text = speaker.get("searchable_text", "")
                if not searchable_text:
                    searchable_text = self._create_searchable_text(speaker)

                # Tokenize the text
                tokens = self._tokenize_text(searchable_text)

                self.speaker_corpus.append(tokens)
                self.speaker_ids.append(
                    str(speaker.get("speaker_id", speaker.get("id", "")))
                )
                self.speaker_data.append(speaker)

            # Create BM25 index
            self.bm25_index = BM25Okapi(self.speaker_corpus)

            logger.info(
                f"✅ BM25 index created successfully with {len(self.speaker_corpus)} documents"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Failed to create BM25 index: {e}")
            return False

    def _tokenize_text(self, text: str) -> List[str]:
        """Tokenize text for BM25 indexing"""
        # Convert to lowercase and extract words
        tokens = re.findall(r"\b\w+\b", text.lower())
        return tokens

    def _create_searchable_text(self, speaker: Dict[str, Any]) -> str:
        """Create searchable text from speaker data"""
        text_parts = []

        # Basic info
        if speaker.get("name"):
            text_parts.append(speaker["name"])
        if speaker.get("job_title"):
            text_parts.append(speaker["job_title"])
        if speaker.get("company"):
            text_parts.append(speaker["company"])
        if speaker.get("bio"):
            text_parts.append(speaker["bio"])

        # Speaking topics
        if speaker.get("speaking_topics"):
            topics = speaker["speaking_topics"]
            if isinstance(topics, list):
                text_parts.extend(topics)
            else:
                text_parts.append(str(topics))

        # Specializations
        if speaker.get("specializations"):
            text_parts.append(speaker["specializations"])

        # Centers/location
        if speaker.get("centers"):
            text_parts.append(speaker["centers"])

        # Work experience
        if speaker.get("work_experience"):
            text_parts.append(speaker["work_experience"])

        # Audiences
        if speaker.get("audiences"):
            audiences = speaker["audiences"]
            if isinstance(audiences, list):
                text_parts.extend(audiences)
            else:
                text_parts.append(str(audiences))

        if speaker.get("industry"):
            text_parts.append(speaker["industry"])

        if speaker.get("certifications"):
            certifications = speaker["certifications"]
            if isinstance(certifications, list):
                text_parts.extend(certifications)
            else:
                text_parts.append(str(certifications))

        return " ".join(str(part) for part in text_parts if part)

    def search(
        self, query: str, limit: int = 50
    ) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Search using BM25"""
        if not self.bm25_index:
            logger.warning("BM25 index not initialized")
            return []

        try:
            # Tokenize query
            query_tokens = self._tokenize_text(query)

            if not query_tokens:
                return []

            # Get BM25 scores
            scores = self.bm25_index.get_scores(query_tokens)

            # Get top candidates with scores > 0
            candidates = []
            for idx, score in enumerate(scores):
                if score > 0:
                    candidates.append((idx, score, self.speaker_data[idx]))

            # Sort by score (descending) and limit
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:limit]

        except Exception as e:
            logger.error(f"Error in BM25 search: {e}")
            return []

    def get_index_stats(self) -> Dict[str, Any]:
        """Get BM25 index statistics"""
        return {
            "indexed_documents": len(self.speaker_corpus),
            "total_tokens": sum(len(doc) for doc in self.speaker_corpus),
            "average_doc_length": (
                np.mean([len(doc) for doc in self.speaker_corpus])
                if self.speaker_corpus
                else 0
            ),
            "index_ready": self.bm25_index is not None,
        }


# Global BM25 service instance
bm25_service = BM25Service()

"""
Result fusion utilities for combining vector search and BM25 results
"""

import logging
from typing import List, Dict, Any, Set
import math

logger = logging.getLogger(__name__)


def reciprocal_rank_fusion(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int = 60,
    vector_weight: float = 0.6,
    bm25_weight: float = 0.4,
) -> List[Dict[str, Any]]:
    """
    Combine vector search and BM25 results using Reciprocal Rank Fusion (RRF)

    Args:
        vector_results: Results from vector search (with similarity_score)
        bm25_results: Results from BM25 search (with bm25_score)
        k: RRF constant (typical value: 60)
        vector_weight: Weight for vector search component
        bm25_weight: Weight for BM25 component

    Returns:
        Fused and ranked results
    """
    try:
        # Create lookup maps for easy access
        vector_lookup = {}
        bm25_lookup = {}

        # Process vector results
        for rank, result in enumerate(vector_results, 1):
            speaker_id = str(result.get("speaker_id", result.get("id", "")))
            vector_lookup[speaker_id] = {
                "rank": rank,
                "score": result.get("similarity_score", 0.0),
                "data": result,
            }

        # Process BM25 results
        for rank, (idx, score, speaker_data) in enumerate(bm25_results, 1):
            speaker_id = str(speaker_data.get("speaker_id", speaker_data.get("id", "")))
            bm25_lookup[speaker_id] = {
                "rank": rank,
                "score": score,
                "data": speaker_data,
            }

        # Get all unique speaker IDs
        all_speaker_ids: Set[str] = set(vector_lookup.keys()) | set(bm25_lookup.keys())

        # Calculate RRF scores
        fused_results = []

        for speaker_id in all_speaker_ids:
            # Get data (prefer vector data if available, otherwise BM25)
            speaker_data = vector_lookup.get(speaker_id, {}).get(
                "data"
            ) or bm25_lookup.get(speaker_id, {}).get("data", {})

            if not speaker_data:
                continue

            # Calculate RRF score
            vector_rrf = 0.0
            bm25_rrf = 0.0

            if speaker_id in vector_lookup:
                vector_rank = vector_lookup[speaker_id]["rank"]
                vector_rrf = 1.0 / (k + vector_rank)

            if speaker_id in bm25_lookup:
                bm25_rank = bm25_lookup[speaker_id]["rank"]
                bm25_rrf = 1.0 / (k + bm25_rank)

            # Weighted RRF score
            rrf_score = (vector_weight * vector_rrf) + (bm25_weight * bm25_rrf)

            # Create fused result
            fused_result = speaker_data.copy()
            fused_result.update(
                {
                    "rrf_score": rrf_score,
                    "vector_rank": vector_lookup.get(speaker_id, {}).get("rank", None),
                    "bm25_rank": bm25_lookup.get(speaker_id, {}).get("rank", None),
                    "vector_score": vector_lookup.get(speaker_id, {}).get("score", 0.0),
                    "bm25_score": bm25_lookup.get(speaker_id, {}).get("score", 0.0),
                    "fusion_type": "rrf",
                }
            )

            fused_results.append(fused_result)

        # Sort by RRF score (descending)
        fused_results.sort(key=lambda x: x.get("rrf_score", 0.0), reverse=True)

        # print("-------------------------------------------------------------------------")
        # print("Fused results:", fused_results)

        logger.info(
            f"RRF fusion: {len(vector_results)} vector + {len(bm25_results)} BM25 → {len(fused_results)} fused results"
        )

        return fused_results

    except Exception as e:
        logger.error(f"Error in reciprocal rank fusion: {e}")
        # Fallback to vector results
        return vector_results

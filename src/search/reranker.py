"""
Hybrid Reranker — Phase 2

Merges results from Elasticsearch (keyword) and Pinecone (vector) searches
using Reciprocal Rank Fusion (RRF) to produce a unified ranked list.

RRF Formula:
    RRF_score(d) = Σ  1 / (k + rank_i(d))

Where:
    k = constant (default 60, from the original RRF paper)
    rank_i(d) = rank of document d in result list i (1-indexed)
"""

from __future__ import annotations

import logging
from collections import defaultdict

from src.search.es_search import SearchResult

logger = logging.getLogger(__name__)

# RRF constant from the original paper (Cormack, Clarke & Buettcher, 2009)
RRF_K = 60


def reciprocal_rank_fusion(
    *result_lists: list[SearchResult],
    k: int = RRF_K,
    top_k: int = 10,
) -> list[SearchResult]:
    """
    Merge multiple ranked result lists using Reciprocal Rank Fusion.

    Args:
        *result_lists: Variable number of ranked SearchResult lists
        k: RRF constant (default: 60)
        top_k: Number of results to return after fusion

    Returns:
        Unified ranked list of SearchResult objects with updated scores
    """
    # chunk_id → accumulated RRF score
    rrf_scores: dict[str, float] = defaultdict(float)
    # chunk_id → best SearchResult object (we keep the one with highest individual score)
    result_map: dict[str, SearchResult] = {}

    for list_idx, results in enumerate(result_lists):
        source_name = f"list_{list_idx}"
        if results:
            source_name = results[0].source

        for rank, result in enumerate(results, start=1):
            rrf_score = 1.0 / (k + rank)
            rrf_scores[result.chunk_id] += rrf_score

            # Keep the result with richer metadata
            if result.chunk_id not in result_map:
                result_map[result.chunk_id] = result
            elif len(result.content) > len(result_map[result.chunk_id].content):
                result_map[result.chunk_id] = result

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    # Build final results with fused scores
    fused_results: list[SearchResult] = []
    for chunk_id in sorted_ids[:top_k]:
        result = result_map[chunk_id]
        fused_results.append(SearchResult(
            chunk_id=result.chunk_id,
            document_name=result.document_name,
            car_brand=result.car_brand,
            car_model=result.car_model,
            content=result.content,
            section=result.section,
            page_number=result.page_number,
            score=rrf_scores[chunk_id],
            source="hybrid",
            section_title=result.section_title,
            subsection_title=result.subsection_title,
            hierarchy_path=result.hierarchy_path,
            block_type=result.block_type,
            metadata={
                "rrf_score": rrf_scores[chunk_id],
                "original_source": result.source,
            },
        ))

    logger.info(
        f"RRF fusion: {sum(len(rl) for rl in result_lists)} total results "
        f"→ {len(fused_results)} fused results"
    )
    return fused_results


def deduplicate_results(
    results: list[SearchResult],
    similarity_threshold: float = 0.85,
) -> list[SearchResult]:
    """
    Remove near-duplicate results based on content overlap.

    Uses a simple Jaccard-like similarity on word sets.
    """
    if not results:
        return []

    unique: list[SearchResult] = [results[0]]

    for result in results[1:]:
        is_duplicate = False
        result_words = set(result.content.lower().split())

        for existing in unique:
            existing_words = set(existing.content.lower().split())
            if not result_words or not existing_words:
                continue

            intersection = len(result_words & existing_words)
            union = len(result_words | existing_words)
            similarity = intersection / union if union > 0 else 0

            if similarity > similarity_threshold:
                is_duplicate = True
                break

        if not is_duplicate:
            unique.append(result)

    if len(results) != len(unique):
        logger.info(f"Deduplication: {len(results)} → {len(unique)} results")

    return unique

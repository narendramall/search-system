"""
Pinecone Vector Search — Phase 2

Performs semantic similarity search against the Pinecone vector index
using Gemini-generated query embeddings.
"""

from __future__ import annotations

import logging

from pinecone import Pinecone

from config import get_settings
from src.llm.gemini_client import GeminiClient
from src.search.es_search import SearchResult

logger = logging.getLogger(__name__)


class VectorSearch:
    """Pinecone semantic search engine for car brochures."""

    def __init__(self) -> None:
        settings = get_settings()

        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required for vector search.")

        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index = self._pc.Index(settings.pinecone_index_name)
        self._gemini = GeminiClient()
        self._top_k = settings.top_k

    def search(
        self,
        query: str,
        top_k: int | None = None,
        brand_filter: str | None = None,
        model_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        Perform semantic search using Pinecone.

        Args:
            query: User's natural language query
            top_k: Number of results to return
            brand_filter: Filter by car brand
            model_filter: Filter by car model

        Returns:
            Ranked list of SearchResult objects
        """
        k = top_k or self._top_k

        # Generate query embedding
        query_embedding = self._gemini.generate_single_embedding(query)
        if not query_embedding:
            logger.error("Failed to generate query embedding")
            return []

        # Build metadata filter
        filter_dict = {}
        if brand_filter:
            filter_dict["car_brand"] = {"$eq": brand_filter}
        if model_filter:
            filter_dict["car_model"] = {"$eq": model_filter}

        # Query Pinecone
        try:
            response = self._index.query(
                vector=query_embedding,
                top_k=k,
                include_metadata=True,
                filter=filter_dict if filter_dict else None,
            )
        except Exception as e:
            logger.error(f"Pinecone search failed: {e}")
            return []

        results: list[SearchResult] = []
        for match in response.matches:
            meta = match.metadata or {}
            results.append(SearchResult(
                chunk_id=match.id,
                document_name=meta.get("document_name", ""),
                car_brand=meta.get("car_brand", ""),
                car_model=meta.get("car_model", ""),
                content=meta.get("content", ""),
                section=meta.get("section", ""),
                page_number=meta.get("page_number", 0),
                score=match.score,
                source="pinecone",
                section_title=meta.get("section_title", ""),
                subsection_title=meta.get("subsection_title", ""),
                hierarchy_path=meta.get("hierarchy_path", ""),
                block_type=meta.get("block_type", "paragraph"),
            ))

        logger.info(f"Pinecone search for '{query}' → {len(results)} results")
        return results

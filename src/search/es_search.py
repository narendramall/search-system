"""
Elasticsearch Search — Phase 1

Performs keyword-based search against the car brochures index
using BM25 scoring with field boosting.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from elasticsearch import Elasticsearch

from config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result with hierarchy metadata."""
    chunk_id: str
    document_name: str
    car_brand: str
    car_model: str
    content: str
    section: str
    page_number: int
    score: float
    source: str = "elasticsearch"
    section_title: str = ""
    subsection_title: str = ""
    hierarchy_path: str = ""
    block_type: str = "paragraph"
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "document_name": self.document_name,
            "car_brand": self.car_brand,
            "car_model": self.car_model,
            "content": self.content,
            "section": self.section,
            "section_title": self.section_title,
            "subsection_title": self.subsection_title,
            "hierarchy_path": self.hierarchy_path,
            "block_type": self.block_type,
            "page_number": self.page_number,
            "score": self.score,
            "source": self.source,
        }


class ElasticsearchSearch:
    """
    Elasticsearch keyword search engine.

    Search strategy:
    - Multi-match across content, car_brand, car_model, section
    - Field boosting: car_model^3, car_brand^2, section^2, content^1
    - Optional filters by brand, model, section
    """

    def __init__(self) -> None:
        settings = get_settings()
        es_kwargs: dict = {"hosts": [settings.elasticsearch_url]}
        if settings.elasticsearch_api_key:
            es_kwargs["api_key"] = settings.elasticsearch_api_key

        self._client = Elasticsearch(**es_kwargs)
        self._index = settings.es_index_name
        self._top_k = settings.top_k

    def search(
        self,
        query: str,
        top_k: int | None = None,
        brand_filter: str | None = None,
        model_filter: str | None = None,
        section_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        Search for car brochure chunks matching the query.

        Args:
            query: User's search query
            top_k: Override for number of results
            brand_filter: Filter by car brand
            model_filter: Filter by car model
            section_filter: Filter by section type

        Returns:
            Ranked list of SearchResult objects
        """
        k = top_k or self._top_k

        # Build the multi-match query with field boosting (includes hierarchy fields)
        must_clause = {
            "multi_match": {
                "query": query,
                "fields": [
                    "content",
                    "car_model^3",
                    "car_brand^2",
                    "section^2",
                    "section_title^2",
                    "subsection_title^2.5",
                    "hierarchy_path",
                ],
                "type": "best_fields",
                "fuzziness": "AUTO",
                "prefix_length": 2,
            }
        }

        # Build optional filters
        filter_clauses = []
        if brand_filter:
            filter_clauses.append({"term": {"car_brand.keyword": brand_filter}})
        if model_filter:
            filter_clauses.append({"match": {"car_model": model_filter}})
        if section_filter:
            filter_clauses.append({"term": {"section.keyword": section_filter}})

        # Assemble the full query
        es_query = {
            "bool": {
                "must": [must_clause],
                "filter": filter_clauses,
            }
        }

        # Add highlight for content
        highlight = {
            "fields": {
                "content": {
                    "fragment_size": 300,
                    "number_of_fragments": 2,
                    "pre_tags": ["**"],
                    "post_tags": ["**"],
                }
            }
        }

        try:
            response = self._client.search(
                index=self._index,
                query=es_query,
                highlight=highlight,
                size=k,
                _source=True,
            )
        except Exception as e:
            logger.error(f"Elasticsearch search failed: {e}")
            return []

        results: list[SearchResult] = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            results.append(SearchResult(
                chunk_id=src.get("chunk_id", hit["_id"]),
                document_name=src.get("document_name", ""),
                car_brand=src.get("car_brand", ""),
                car_model=src.get("car_model", ""),
                content=src.get("content", ""),
                section=src.get("section", ""),
                page_number=src.get("page_number", 0),
                score=hit["_score"],
                source="elasticsearch",
                section_title=src.get("section_title", ""),
                subsection_title=src.get("subsection_title", ""),
                hierarchy_path=src.get("hierarchy_path", ""),
                block_type=src.get("block_type", "paragraph"),
                metadata={
                    "highlights": hit.get("highlight", {}).get("content", []),
                },
            ))

        logger.info(f"ES search for '{query}' → {len(results)} results")
        return results

    def search_by_specs(self, spec_query: str, top_k: int | None = None) -> list[SearchResult]:
        """
        Specialized search for technical specifications.
        Boosts the 'specifications' and 'engine' sections.
        """
        k = top_k or self._top_k

        es_query = {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": spec_query,
                            "fields": ["content^2", "car_model^3"],
                            "type": "best_fields",
                        }
                    }
                ],
                "should": [
                    {"term": {"section.keyword": {"value": "specifications", "boost": 3.0}}},
                    {"term": {"section.keyword": {"value": "engine", "boost": 2.0}}},
                    {"term": {"section.keyword": {"value": "performance", "boost": 2.0}}},
                    {"term": {"section.keyword": {"value": "dimensions", "boost": 2.0}}},
                    {"term": {"block_type": {"value": "table", "boost": 2.5}}},
                ],
            }
        }

        try:
            response = self._client.search(
                index=self._index,
                query=es_query,
                size=k,
                _source=True,
            )
        except Exception as e:
            logger.error(f"ES spec search failed: {e}")
            return []

        results: list[SearchResult] = []
        for hit in response["hits"]["hits"]:
            src = hit["_source"]
            results.append(SearchResult(
                chunk_id=src.get("chunk_id", hit["_id"]),
                document_name=src.get("document_name", ""),
                car_brand=src.get("car_brand", ""),
                car_model=src.get("car_model", ""),
                content=src.get("content", ""),
                section=src.get("section", ""),
                page_number=src.get("page_number", 0),
                score=hit["_score"],
                source="elasticsearch",
                section_title=src.get("section_title", ""),
                subsection_title=src.get("subsection_title", ""),
                hierarchy_path=src.get("hierarchy_path", ""),
                block_type=src.get("block_type", "paragraph"),
            ))

        return results

    def close(self) -> None:
        """Close the Elasticsearch client."""
        self._client.close()

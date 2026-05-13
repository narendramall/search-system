"""
Elasticsearch Indexer — Phase 1

Indexes chunked car brochure content into Elasticsearch
with an optimized mapping for car specification searches.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from elasticsearch import Elasticsearch, helpers

from config import get_settings
from src.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# Index settings — analysis config (shard/replica settings omitted for serverless compat)
INDEX_SETTINGS = {
    "analysis": {
        "analyzer": {
            "car_analyzer": {
                "type": "custom",
                "tokenizer": "standard",
                "filter": ["lowercase", "stop", "snowball"],
            }
        }
    },
}

# Index mapping optimized for car brochure search
INDEX_MAPPINGS = {
    "properties": {
        "chunk_id": {"type": "keyword"},
        "document_name": {"type": "keyword"},
        "car_brand": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword"}},
        },
        "car_model": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword"}},
        },
        "content": {
            "type": "text",
            "analyzer": "car_analyzer",
        },
        "chunk_index": {"type": "integer"},
        "page_number": {"type": "integer"},
        "section": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword"}},
        },
        "section_title": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword"}},
        },
        "subsection_title": {
            "type": "text",
            "fields": {"keyword": {"type": "keyword"}},
        },
        "hierarchy_path": {
            "type": "text",
            "analyzer": "car_analyzer",
        },
        "block_type": {"type": "keyword"},
        "token_count": {"type": "integer"},
        "metadata": {"type": "object", "enabled": False},
        "ingested_at": {"type": "date"},
    }
}


class ElasticsearchIndexer:
    """Handles indexing chunks into Elasticsearch."""

    def __init__(self) -> None:
        settings = get_settings()
        es_kwargs: dict = {"hosts": [settings.elasticsearch_url]}

        if settings.elasticsearch_api_key:
            es_kwargs["api_key"] = settings.elasticsearch_api_key

        self._client = Elasticsearch(**es_kwargs)
        self._index = settings.es_index_name

    def ping(self) -> bool:
        """Check if Elasticsearch is reachable."""
        try:
            return self._client.ping()
        except Exception as e:
            logger.error(f"Elasticsearch ping failed: {e}")
            return False

    def create_index(self, delete_existing: bool = False) -> None:
        """Create the Elasticsearch index with the car brochure mapping."""
        if self._client.indices.exists(index=self._index):
            if delete_existing:
                logger.warning(f"Deleting existing index: {self._index}")
                self._client.indices.delete(index=self._index)
            else:
                logger.info(f"Index '{self._index}' already exists, skipping creation.")
                return

        self._client.indices.create(
            index=self._index,
            settings=INDEX_SETTINGS,
            mappings=INDEX_MAPPINGS,
        )
        logger.info(f"✓ Created index: {self._index}")

    def index_chunks(self, chunks: list[Chunk], batch_size: int = 100) -> int:
        """
        Bulk-index chunks into Elasticsearch.

        Returns:
            Number of successfully indexed documents.
        """
        if not chunks:
            logger.warning("No chunks to index.")
            return 0

        now = datetime.now(timezone.utc).isoformat()

        def _generate_actions():
            for chunk in chunks:
                doc = chunk.to_dict()
                doc["ingested_at"] = now
                yield {
                    "_index": self._index,
                    "_id": chunk.chunk_id,
                    "_source": doc,
                }

        success_count, errors = helpers.bulk(
            self._client,
            _generate_actions(),
            chunk_size=batch_size,
            raise_on_error=False,
        )

        if errors:
            logger.error(f"Bulk indexing errors: {len(errors)} failures")
            for err in errors[:5]:  # Log first 5 errors
                logger.error(f"  {err}")

        logger.info(f"✓ Indexed {success_count}/{len(chunks)} chunks into '{self._index}'")
        return success_count

    def get_doc_count(self) -> int:
        """Get the number of documents in the index."""
        try:
            result = self._client.count(index=self._index)
            return result["count"]
        except Exception:
            return 0

    def delete_index(self) -> None:
        """Delete the index."""
        if self._client.indices.exists(index=self._index):
            self._client.indices.delete(index=self._index)
            logger.info(f"Deleted index: {self._index}")

    def close(self) -> None:
        """Close the Elasticsearch client."""
        self._client.close()

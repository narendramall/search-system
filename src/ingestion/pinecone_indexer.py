"""
Pinecone Vector Indexer — Phase 2

Generates embeddings via Gemini and indexes chunks into Pinecone
for semantic similarity search.
"""

from __future__ import annotations

import logging
import time

from pinecone import Pinecone, ServerlessSpec

from config import get_settings
from src.ingestion.chunker import Chunk
from src.llm.gemini_client import GeminiClient

logger = logging.getLogger(__name__)


class PineconeIndexer:
    """Handles embedding generation and vector indexing into Pinecone."""

    def __init__(self) -> None:
        settings = get_settings()

        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required for Phase 2. Set it in your .env file.")

        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        self._index_name = settings.pinecone_index_name
        self._gemini = GeminiClient()
        self._dimension = 768  # gemini-embedding-001 with output_dimensionality=768

    def create_index(self, delete_existing: bool = False) -> None:
        """Create the Pinecone index if it doesn't exist."""
        existing_indexes = [idx.name for idx in self._pc.list_indexes()]

        if self._index_name in existing_indexes:
            if delete_existing:
                logger.warning(f"Deleting existing Pinecone index: {self._index_name}")
                self._pc.delete_index(self._index_name)
                time.sleep(5)  # Wait for deletion to propagate
            else:
                logger.info(f"Pinecone index '{self._index_name}' already exists.")
                return

        self._pc.create_index(
            name=self._index_name,
            dimension=self._dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region=get_settings().pinecone_environment,
            ),
        )
        # Wait for index to be ready
        while not self._pc.describe_index(self._index_name).status["ready"]:
            time.sleep(1)

        logger.info(f"✓ Created Pinecone index: {self._index_name}")

    def index_chunks(self, chunks: list[Chunk], batch_size: int = 50) -> int:
        """
        Generate embeddings and upsert chunks into Pinecone.

        Returns:
            Number of successfully indexed vectors.
        """
        if not chunks:
            logger.warning("No chunks to index into Pinecone.")
            return 0

        index = self._pc.Index(self._index_name)
        total_indexed = 0

        # Process in batches to manage memory and API limits
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [chunk.content for chunk in batch]

            # Generate embeddings
            embeddings = self._gemini.generate_embeddings(texts)

            # Prepare vectors for upsert
            vectors = []
            for chunk, embedding in zip(batch, embeddings):
                vectors.append({
                    "id": chunk.chunk_id,
                    "values": embedding,
                    "metadata": {
                        "document_name": chunk.document_name,
                        "car_brand": chunk.car_brand,
                        "car_model": chunk.car_model,
                        "content": chunk.content[:1000],  # Pinecone metadata limit
                        "section": chunk.section,
                        "section_title": chunk.section_title,
                        "subsection_title": chunk.subsection_title,
                        "hierarchy_path": chunk.hierarchy_path,
                        "block_type": chunk.block_type,
                        "page_number": chunk.page_number,
                        "chunk_index": chunk.chunk_index,
                    },
                })

            # Upsert to Pinecone
            index.upsert(vectors=vectors)
            total_indexed += len(vectors)
            logger.info(f"  Upserted batch {i // batch_size + 1}: {len(vectors)} vectors")

        logger.info(f"✓ Indexed {total_indexed} vectors into Pinecone '{self._index_name}'")
        return total_indexed

    def get_stats(self) -> dict:
        """Get index statistics."""
        try:
            index = self._pc.Index(self._index_name)
            stats = index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "dimension": stats.dimension,
            }
        except Exception as e:
            logger.error(f"Failed to get Pinecone stats: {e}")
            return {}

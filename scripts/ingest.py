#!/usr/bin/env python3
"""
Ingest Script — Parse PDFs and index into search backends.

Usage:
    # Phase 1: Ingest into Elasticsearch only
    python scripts/ingest.py

    # Phase 2: Ingest into both Elasticsearch and Pinecone
    python scripts/ingest.py --mode hybrid

    # Re-index (delete existing and recreate)
    python scripts/ingest.py --fresh

    # Custom data directory
    python scripts/ingest.py --data-dir /path/to/pdfs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import get_settings, SearchMode
from src.ingestion.pdf_parser import parse_all_pdfs
from src.ingestion.chunker import chunk_all_documents

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def ingest_elasticsearch(chunks, fresh: bool = False) -> int:
    """Index chunks into Elasticsearch."""
    from src.ingestion.es_indexer import ElasticsearchIndexer

    indexer = ElasticsearchIndexer()

    if not indexer.ping():
        logger.error("Cannot connect to Elasticsearch. Is it running?")
        logger.error(f"  URL: {get_settings().elasticsearch_url}")
        return 0

    indexer.create_index(delete_existing=fresh)
    count = indexer.index_chunks(chunks)
    indexer.close()
    return count


def ingest_pinecone(chunks, fresh: bool = False) -> int:
    """Index chunks into Pinecone (Phase 2)."""
    from src.ingestion.pinecone_indexer import PineconeIndexer

    indexer = PineconeIndexer()
    indexer.create_index(delete_existing=fresh)
    count = indexer.index_chunks(chunks)
    return count


def main():
    parser = argparse.ArgumentParser(
        description="Ingest car brochure PDFs into the search system."
    )
    parser.add_argument(
        "--mode",
        choices=["elasticsearch", "pinecone", "hybrid"],
        default=None,
        help="Search mode to ingest for (default: from .env SEARCH_MODE)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Path to directory containing PDFs (default: ./data)",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Delete existing indices and re-create from scratch",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Override chunk size in tokens",
    )
    args = parser.parse_args()

    settings = get_settings()
    mode = SearchMode(args.mode) if args.mode else settings.search_mode
    data_dir = Path(args.data_dir) if args.data_dir else settings.data_dir
    chunk_size = args.chunk_size or settings.chunk_size
    chunk_overlap = settings.chunk_overlap

    # Banner
    logger.info("=" * 60)
    logger.info("  Car Brochure Search System — Ingestion Pipeline")
    logger.info("=" * 60)
    logger.info(f"  Mode         : {mode.value}")
    logger.info(f"  Data dir     : {data_dir}")
    logger.info(f"  Chunk size   : {chunk_size} tokens")
    logger.info(f"  Chunk overlap: {chunk_overlap} tokens")
    logger.info(f"  Fresh index  : {args.fresh}")
    logger.info("=" * 60)

    # Step 1: Parse PDFs
    logger.info("\n📄 Step 1: Parsing PDFs...")
    documents = parse_all_pdfs(data_dir)
    if not documents:
        logger.error(f"No PDFs found in {data_dir}. Add PDF brochures and try again.")
        sys.exit(1)

    total_pages = sum(doc.total_pages for doc in documents)
    total_chars = sum(len(doc.full_text) for doc in documents)
    logger.info(f"  Parsed {len(documents)} PDFs ({total_pages} pages, {total_chars:,} chars)")

    # Step 2: Chunk documents
    logger.info("\n✂️  Step 2: Chunking documents...")
    chunks = chunk_all_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info(f"  Created {len(chunks)} chunks")

    # Step 3: Index into backend(s)
    if mode in (SearchMode.ELASTICSEARCH, SearchMode.HYBRID):
        logger.info("\n🔍 Step 3a: Indexing into Elasticsearch...")
        es_count = ingest_elasticsearch(chunks, fresh=args.fresh)
        logger.info(f"  Elasticsearch: {es_count} chunks indexed")

    if mode in (SearchMode.PINECONE, SearchMode.HYBRID):
        logger.info("\n🧠 Step 3b: Indexing into Pinecone (generating embeddings)...")
        pc_count = ingest_pinecone(chunks, fresh=args.fresh)
        logger.info(f"  Pinecone: {pc_count} vectors indexed")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("  ✅ Ingestion Complete!")
    logger.info(f"  Documents: {len(documents)}")
    logger.info(f"  Chunks   : {len(chunks)}")
    logger.info(f"  Mode     : {mode.value}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

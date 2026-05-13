# Car Brochure Search System — Architecture

## Overview

A hybrid search system that ingests car brochure PDFs and provides an intelligent chatbot interface for car recommendations and specifications. The system combines **keyword search** (Elasticsearch) with **semantic vector search** (Pinecone) and uses **Gemini LLM** to generate natural language responses.

---

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     USER INTERFACE                       │
│              (CLI Chat / FastAPI REST API)               │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                    CHAT ENGINE                           │
│  • Query understanding                                  │
│  • Conversation context management                      │
│  • Search strategy routing                              │
└──────────────────────┬──────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼                         ▼
┌──────────────────┐    ┌──────────────────────┐
│  ELASTICSEARCH   │    │    PINECONE           │
│  (Phase 1)       │    │    (Phase 2)          │
│                  │    │                       │
│  • Exact keyword │    │  • Semantic similarity│
│  • BM25 scoring  │    │  • Dense embeddings   │
│  • Spec lookups  │    │  • Feeling/summary    │
└────────┬─────────┘    └──────────┬────────────┘
         │                         │
         └────────────┬────────────┘
                      ▼
         ┌────────────────────────┐
         │      RERANKER          │
         │      (Phase 2)         │
         │                        │
         │  • Reciprocal Rank     │
         │    Fusion (RRF)        │
         │  • Score normalization │
         │  • Deduplication       │
         └────────────┬───────────┘
                      ▼
         ┌────────────────────────┐
         │      GEMINI LLM        │
         │                        │
         │  • Context synthesis   │
         │  • Answer generation   │
         │  • Car recommendations │
         │  • Spec formatting     │
         └────────────┬───────────┘
                      ▼
              Final Response
```

---

## Phase 1: Elasticsearch Only

### Components
1. **PDF Ingestion Pipeline** — Extract text from PDFs, chunk intelligently, index into Elasticsearch
2. **Elasticsearch Service** — BM25 keyword search with field boosting
3. **Gemini LLM** — Generate natural language answers from search results
4. **Chat Engine** — Orchestrate search → LLM flow
5. **API / CLI** — User interaction layer

### Data Flow
```
PDFs → PDF Parser → Chunker → ES Indexer → Elasticsearch
User Query → ES Search → Context Chunks → Gemini → Response
```

### Elasticsearch Index Schema
```json
{
  "car_brochures": {
    "mappings": {
      "properties": {
        "chunk_id":     { "type": "keyword" },
        "document_name":{ "type": "keyword" },
        "car_brand":    { "type": "keyword" },
        "car_model":    { "type": "text", "fields": { "keyword": { "type": "keyword" }}},
        "content":      { "type": "text", "analyzer": "standard" },
        "chunk_index":  { "type": "integer" },
        "metadata":     { "type": "object" },
        "page_number":  { "type": "integer" },
        "section":      { "type": "text" },
        "ingested_at":  { "type": "date" }
      }
    }
  }
}
```

---

## Phase 2: Hybrid Search (Elasticsearch + Pinecone)

### Additional Components
1. **Pinecone Vector Indexer** — Generate embeddings via Gemini, store in Pinecone
2. **Vector Search Service** — Semantic similarity search
3. **Hybrid Reranker** — Reciprocal Rank Fusion to merge ES + Pinecone results

### Data Flow
```
PDFs → PDF Parser → Chunker → ES Indexer + Pinecone Indexer

User Query → ┬─ ES Keyword Search ────┐
             └─ Pinecone Vector Search─┤
                                       ▼
                                   Reranker (RRF)
                                       ▼
                                   Gemini LLM
                                       ▼
                                   Response
```

### Reranking Strategy: Reciprocal Rank Fusion (RRF)
```
RRF_score(d) = Σ 1 / (k + rank_i(d))
```
Where `k = 60` (constant), `rank_i(d)` is the rank of document `d` in result list `i`.

---

## Chunking Strategy

- **Chunk size**: 512 tokens with 50-token overlap
- **Respect boundaries**: Split on paragraphs/sections, never mid-sentence
- **Metadata preservation**: Each chunk carries source document, page number, section heading
- **Special handling**: Tables and spec sheets kept as atomic chunks

---

## Directory Structure

```
search-system/
├── app.py                      # FastAPI entry point
├── requirements.txt
├── .env                        # API keys & config
├── config/
│   └── settings.py             # Centralized configuration
├── data/                       # Drop PDF brochures here
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── pdf_parser.py       # PDF text extraction
│   │   ├── chunker.py          # Intelligent text chunking
│   │   ├── es_indexer.py       # Phase 1: ES indexing
│   │   └── pinecone_indexer.py # Phase 2: Pinecone indexing
│   ├── search/
│   │   ├── __init__.py
│   │   ├── es_search.py        # Phase 1: ES keyword search
│   │   ├── vector_search.py    # Phase 2: Pinecone vector search
│   │   └── reranker.py         # Phase 2: Hybrid reranking
│   ├── llm/
│   │   ├── __init__.py
│   │   └── gemini_client.py    # Gemini LLM integration
│   └── chat/
│       ├── __init__.py
│       └── engine.py           # Chat orchestration engine
├── scripts/
│   ├── ingest.py               # CLI: Ingest all PDFs
│   └── chat_cli.py             # CLI: Interactive chatbot
└── docs/
    └── architecture.md         # This file
```

---

## Configuration

All secrets and tunable parameters live in `.env` and `config/settings.py`:

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key |
| `ELASTICSEARCH_URL` | Elasticsearch cluster URL |
| `ELASTICSEARCH_API_KEY` | ES authentication key |
| `PINECONE_API_KEY` | Pinecone API key (Phase 2) |
| `PINECONE_INDEX_NAME` | Pinecone index name (Phase 2) |
| `SEARCH_MODE` | `elasticsearch` / `pinecone` / `hybrid` |
| `CHUNK_SIZE` | Tokens per chunk (default: 512) |
| `CHUNK_OVERLAP` | Overlap tokens (default: 50) |
| `ES_INDEX_NAME` | Elasticsearch index name |
| `TOP_K` | Number of results to retrieve |

---

## Key Design Decisions

1. **Strategy Pattern for Search**: `SearchProvider` interface allows swapping/combining search backends
2. **Feature Flags**: `SEARCH_MODE` env var controls Phase 1 vs Phase 2 behavior at runtime
3. **Chunking with Overlap**: Prevents losing context at boundaries
4. **RRF Reranking**: Simple, effective, no ML model needed — proven in hybrid search literature
5. **Gemini for Embeddings + LLM**: Single ecosystem reduces complexity (Phase 2)
6. **Separation of Concerns**: Ingestion, search, LLM, and chat are independent modules

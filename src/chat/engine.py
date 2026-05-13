"""
Chat Engine — Orchestrates search, retrieval, and LLM generation.

This is the core brain of the chatbot. It:
1. Understands the user's query intent
2. Routes to the appropriate search backend(s)
3. Retrieves relevant context
4. Generates an answer via Gemini

Supports Phase 1 (ES-only) and Phase 2 (hybrid) modes via SEARCH_MODE config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from config import get_settings, SearchMode
from src.llm.gemini_client import GeminiClient
from src.search.es_search import ElasticsearchSearch, SearchResult

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """A single message in the conversation."""
    role: str  # "user" or "assistant"
    content: str


@dataclass
class ChatResponse:
    """Response from the chat engine."""
    answer: str
    sources: list[dict] = field(default_factory=list)
    search_mode: str = "elasticsearch"
    query_intent: dict = field(default_factory=dict)


class ChatEngine:
    """
    Main chat orchestration engine.

    Manages conversation history and routes queries through
    the search → context → LLM pipeline.
    """

    def __init__(self) -> None:
        self._settings = get_settings()
        self._gemini = GeminiClient()
        self._conversation_history: list[ChatMessage] = []

        # Initialize search backends based on mode
        self._es_search: ElasticsearchSearch | None = None
        self._vector_search = None
        self._search_mode = self._settings.search_mode

        self._init_search_backends()

    def _init_search_backends(self) -> None:
        """Initialize search backends based on configured mode."""
        mode = self._search_mode

        if mode in (SearchMode.ELASTICSEARCH, SearchMode.HYBRID):
            try:
                self._es_search = ElasticsearchSearch()
                logger.info("✓ Elasticsearch search backend initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Elasticsearch: {e}")

        if mode in (SearchMode.PINECONE, SearchMode.HYBRID):
            try:
                from src.search.vector_search import VectorSearch
                self._vector_search = VectorSearch()
                logger.info("✓ Pinecone vector search backend initialized")
            except Exception as e:
                logger.warning(f"Pinecone not available (Phase 2): {e}")

    def chat(self, user_query: str) -> ChatResponse:
        """
        Process a user query and return a response.

        Pipeline:
        1. Classify query intent
        2. Search relevant context
        3. Generate LLM answer
        4. Update conversation history
        """
        logger.info(f"Processing query: '{user_query}'")

        # Step 1: Classify intent to optimize search
        intent = self._gemini.classify_query_intent(user_query)
        logger.info(f"Query intent: {intent}")

        # Step 2: Retrieve relevant context
        search_results = self._retrieve_context(user_query, intent)

        # Step 3: Generate answer
        context_dicts = [r.to_dict() for r in search_results]
        history_dicts = [
            {"role": m.role, "content": m.content}
            for m in self._conversation_history
        ]

        answer = self._gemini.generate_answer(
            query=user_query,
            context_chunks=context_dicts,
            conversation_history=history_dicts,
        )

        # Step 4: Update conversation history
        self._conversation_history.append(ChatMessage(role="user", content=user_query))
        self._conversation_history.append(ChatMessage(role="assistant", content=answer))

        # Keep history bounded
        if len(self._conversation_history) > 20:
            self._conversation_history = self._conversation_history[-20:]

        # Build source references
        sources = []
        seen_docs = set()
        for r in search_results:
            doc_key = f"{r.car_brand} {r.car_model} ({r.document_name})"
            if doc_key not in seen_docs:
                sources.append({
                    "document": r.document_name,
                    "car": f"{r.car_brand} {r.car_model}",
                    "section": r.section,
                    "page": r.page_number,
                })
                seen_docs.add(doc_key)

        return ChatResponse(
            answer=answer,
            sources=sources,
            search_mode=self._search_mode.value,
            query_intent=intent,
        )

    def _retrieve_context(
        self, query: str, intent: dict
    ) -> list[SearchResult]:
        """
        Retrieve relevant context based on search mode and query intent.

        Phase 1: Elasticsearch only
        Phase 2: Hybrid (ES + Pinecone → RRF reranking)
        """
        mode = self._search_mode
        all_results: list[SearchResult] = []

        # --- Elasticsearch search ---
        if self._es_search and mode in (SearchMode.ELASTICSEARCH, SearchMode.HYBRID):
            is_spec_query = intent.get("intent") in ("specification", "comparison")
            if is_spec_query:
                es_results = self._es_search.search_by_specs(query)
            else:
                es_results = self._es_search.search(query)

            if mode == SearchMode.ELASTICSEARCH:
                # Phase 1: return ES results directly
                return self._dedupe(es_results)
            else:
                all_results.append(es_results)

        # --- Pinecone vector search ---
        if self._vector_search and mode in (SearchMode.PINECONE, SearchMode.HYBRID):
            vector_results = self._vector_search.search(query)

            if mode == SearchMode.PINECONE:
                return self._dedupe(vector_results)
            else:
                all_results.append(vector_results)

        # --- Hybrid: Rerank with RRF ---
        if mode == SearchMode.HYBRID and len(all_results) >= 2:
            from src.search.reranker import reciprocal_rank_fusion, deduplicate_results
            fused = reciprocal_rank_fusion(
                *all_results,
                top_k=self._settings.top_k,
            )
            return deduplicate_results(fused)

        # Fallback: return whatever we have
        flat = []
        for result_list in all_results:
            flat.extend(result_list)
        return self._dedupe(flat)

    def _dedupe(self, results: list[SearchResult]) -> list[SearchResult]:
        """Simple deduplication by chunk_id."""
        seen = set()
        unique = []
        for r in results:
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                unique.append(r)
        return unique[: self._settings.top_k]

    def reset_conversation(self) -> None:
        """Clear conversation history."""
        self._conversation_history.clear()
        logger.info("Conversation history cleared")

    @property
    def search_mode(self) -> str:
        return self._search_mode.value

    @property
    def history_length(self) -> int:
        return len(self._conversation_history)

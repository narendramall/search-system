"""
Gemini LLM Client — Generates natural language responses and embeddings.

Uses the Google Generative AI SDK (google-genai) to:
- Generate conversational answers from search context (Phase 1 + 2)
- Create text embeddings for vector search (Phase 2)
"""

from __future__ import annotations

import logging
from typing import Optional

from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

from config import get_settings

logger = logging.getLogger(__name__)

# System prompt for the car recommendation chatbot
CAR_ADVISOR_SYSTEM_PROMPT = """\
You are an expert car advisor and consultant with deep knowledge of automobiles.
You help users find the right car based on their needs and provide detailed specifications.

RULES:
1. Answer ONLY based on the provided context from car brochures. Do not make up specifications.
2. If the context doesn't contain enough information, say so honestly.
3. When comparing cars, create structured comparisons with key specs side by side.
4. For specification queries (top speed, ground clearance, mileage, etc.), provide exact numbers from the brochure.
5. When recommending cars, explain WHY based on the user's stated needs.
6. Format numbers and specs clearly (e.g., "185 km/h", "200mm ground clearance").
7. If multiple variants exist, mention which variant the spec applies to.
8. Be conversational but precise — users trust you for accurate data.

RESPONSE FORMAT:
- Use bullet points for specifications
- Use tables when comparing multiple cars
- Bold the key numbers and model names
- Keep responses concise but complete
"""


class GeminiClient:
    """Google Gemini LLM client for answer generation and embeddings."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = "gemini-2.5-flash"
        self._embedding_model = "gemini-embedding-2"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate_answer(
        self,
        query: str,
        context_chunks: list[dict],
        conversation_history: list[dict] | None = None,
    ) -> str:
        """
        Generate an answer based on search context and conversation history.

        Args:
            query: User's current question
            context_chunks: List of relevant chunks from search
            conversation_history: Previous chat messages [{"role": "user"|"assistant", "content": "..."}]

        Returns:
            Generated answer string
        """
        # Build context string from chunks
        context_parts = []
        for i, chunk in enumerate(context_chunks, 1):
            source = chunk.get("document_name", "Unknown")
            brand = chunk.get("car_brand", "")
            model = chunk.get("car_model", "")
            section = chunk.get("section", "")
            content = chunk.get("content", "")
            context_parts.append(
                f"[Source {i}: {brand} {model} — {source} (Section: {section})]\n{content}"
            )

        context_str = "\n\n---\n\n".join(context_parts) if context_parts else "No relevant information found."

        # Build the user message with context
        user_message = f"""Based on the following car brochure excerpts, answer the user's question.

CONTEXT FROM CAR BROCHURES:
{context_str}

USER QUESTION: {query}

Provide a helpful, accurate answer based on the context above."""

        # Build conversation contents
        contents: list[types.Content] = []

        # Add conversation history if present
        if conversation_history:
            for msg in conversation_history[-10:]:  # Keep last 10 turns
                role = "user" if msg["role"] == "user" else "model"
                contents.append(
                    types.Content(
                        role=role,
                        parts=[types.Part.from_text(text=msg["content"])],
                    )
                )

        # Add current query with context
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_message)],
            )
        )

        # Generate response
        response = self._client.models.generate_content(
            model=self._model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=CAR_ADVISOR_SYSTEM_PROMPT,
                temperature=0.3,
                top_p=0.9,
                max_output_tokens=2048,
            ),
        )

        answer = response.text or "I couldn't generate a response. Please try rephrasing your question."
        logger.info(f"Gemini generated answer ({len(answer)} chars)")
        return answer

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a list of texts using Gemini.
        Used in Phase 2 for Pinecone vector indexing.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        embeddings = []
        # Process in batches of 100 (API limit)
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self._client.models.embed_content(
                model=self._embedding_model,
                contents=batch,
                config={"output_dimensionality": 768},
            )
            for embedding in response.embeddings:
                embeddings.append(embedding.values)

        logger.info(f"Generated {len(embeddings)} embeddings")
        return embeddings

    def generate_single_embedding(self, text: str) -> list[float]:
        """Generate embedding for a single text. Used for query embedding in Phase 2."""
        result = self.generate_embeddings([text])
        return result[0] if result else []

    def classify_query_intent(self, query: str) -> dict:
        """
        Classify the user's query intent to optimize search strategy.

        Returns a dict with:
        - intent: "specification" | "comparison" | "recommendation" | "general"
        - keywords: extracted key terms
        - car_mentions: any specific cars mentioned
        """
        prompt = f"""Classify this car-related query and extract information.
Return ONLY a JSON object with these fields:
- "intent": one of "specification", "comparison", "recommendation", "general"
- "keywords": list of important search keywords
- "car_mentions": list of any specific car brands/models mentioned

Query: "{query}"

JSON:"""

        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=256,
            ),
        )

        # Parse the response — fallback to general if parsing fails
        import json
        try:
            text = response.text.strip()
            # Remove markdown code block if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            return json.loads(text)
        except Exception:
            return {
                "intent": "general",
                "keywords": query.split(),
                "car_mentions": [],
            }

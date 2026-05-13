"""
Car Brochure Search System — Centralized Configuration

All settings are loaded from environment variables (.env file)
and validated using Pydantic Settings.
"""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SearchMode(str, Enum):
    """Available search strategies."""
    ELASTICSEARCH = "elasticsearch"
    PINECONE = "pinecone"
    HYBRID = "hybrid"


class Settings(BaseSettings):
    """Application settings loaded from .env file."""

    # --- Gemini ---
    gemini_api_key: str = Field(..., description="Google Gemini API key")

    # --- LlamaParse ---
    llama_cloud_api_key: str = Field(
        default="",
        description="LlamaCloud / LlamaParse API key for PDF extraction",
    )

    # --- Elasticsearch ---
    elasticsearch_url: str = Field(
        default="http://localhost:9200",
        description="Elasticsearch cluster URL",
    )
    elasticsearch_api_key: str = Field(
        default="",
        description="Elasticsearch API key for authentication",
    )
    es_index_name: str = Field(
        default="car_brochures",
        description="Elasticsearch index name",
    )

    # --- Pinecone (Phase 2) ---
    pinecone_api_key: str = Field(
        default="",
        description="Pinecone API key",
    )
    pinecone_index_name: str = Field(
        default="car-brochures",
        description="Pinecone index name",
    )
    pinecone_environment: str = Field(
        default="us-east-1",
        description="Pinecone environment/region",
    )

    # --- Search ---
    search_mode: SearchMode = Field(
        default=SearchMode.ELASTICSEARCH,
        description="Active search mode: elasticsearch | pinecone | hybrid",
    )

    # --- Chunking ---
    chunk_size: int = Field(default=512, description="Tokens per chunk")
    chunk_overlap: int = Field(default=50, description="Overlap tokens between chunks")

    # --- Retrieval ---
    top_k: int = Field(default=10, description="Number of results to retrieve per search backend")

    # --- Server ---
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    # --- Paths ---
    @property
    def data_dir(self) -> Path:
        return PROJECT_ROOT / "data"

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """Singleton settings instance."""
    return Settings()

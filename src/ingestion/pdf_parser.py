"""
PDF Parser — Extracts text & structure from car brochure PDFs.

Primary extractor : **LlamaParse** (cloud API).
  – Understands tables, figures, headings, and complex layouts.
  – Returns structured Markdown with page boundaries.
  – Results are cached locally in `.cache/llamaparse/` to avoid
    redundant API calls on re-ingestion.

Fallback extractor: **PyMuPDF (fitz)** (local, offline).
  – Used when LlamaParse is unavailable or the API key is not set.

The output `ParsedDocument` keeps a page-by-page representation so the
downstream hierarchical chunker can map content back to page numbers.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF — local fallback

logger = logging.getLogger(__name__)

# Cache directory for LlamaParse outputs (relative to project root)
from config import PROJECT_ROOT

_CACHE_DIR = PROJECT_ROOT / ".cache" / "llamaparse"


# ──────────────────────────────────────────────
#  Data classes
# ──────────────────────────────────────────────

@dataclass
class PageContent:
    """Represents extracted content from a single PDF page."""
    page_number: int
    text: str
    tables: list[list[list[str]]] = field(default_factory=list)


@dataclass
class ParsedDocument:
    """Represents the full extracted content from a PDF."""
    file_name: str
    file_path: str
    total_pages: int
    pages: list[PageContent]
    metadata: dict = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LlamaParse cache helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _cache_key(pdf_path: Path) -> str:
    """
    Deterministic cache key based on file name + size + mtime.
    If the PDF changes, the cache key changes → stale cache is ignored.
    """
    stat = pdf_path.stat()
    raw = f"{pdf_path.name}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def _cache_path(pdf_path: Path) -> Path:
    """Return the path to the cached markdown file for a given PDF."""
    key = _cache_key(pdf_path)
    stem = pdf_path.stem  # e.g. Hyundai_Creta_2024
    return _CACHE_DIR / f"{stem}_{key}.md"


def _cache_meta_path(pdf_path: Path) -> Path:
    """Return the path to the cached metadata JSON file."""
    key = _cache_key(pdf_path)
    stem = pdf_path.stem
    return _CACHE_DIR / f"{stem}_{key}.meta.json"


def _load_from_cache(pdf_path: Path) -> Optional[str]:
    """Load cached LlamaParse markdown output. Returns None if not cached."""
    cache_file = _cache_path(pdf_path)
    if cache_file.exists():
        md = cache_file.read_text(encoding="utf-8")
        if md.strip():
            logger.info(f"  ✓ Loaded from cache: {cache_file.name}")
            return md
    return None


def _save_to_cache(pdf_path: Path, markdown: str, metadata: dict | None = None) -> None:
    """Save LlamaParse markdown output to local cache."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_path(pdf_path)
    cache_file.write_text(markdown, encoding="utf-8")

    # Save metadata sidecar
    meta_file = _cache_meta_path(pdf_path)
    meta = {
        "source_pdf": pdf_path.name,
        "source_size": pdf_path.stat().st_size,
        "extractor": "llamaparse",
        "markdown_chars": len(markdown),
        **(metadata or {}),
    }
    meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(f"  ✓ Cached output: {cache_file.name} ({len(markdown):,} chars)")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  LlamaParse extractor (primary)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _llamaparse_available() -> bool:
    """Check whether LlamaParse can be used (API key present + library installed)."""
    try:
        from config import get_settings
        return bool(get_settings().llama_cloud_api_key)
    except Exception:
        return False


def _extract_with_llamaparse(pdf_path: Path) -> Optional[ParsedDocument]:
    """
    Extract content using LlamaParse (with local cache).

    1. Check `.cache/llamaparse/` for a cached markdown file.
    2. If cache hit → build ParsedDocument from cached markdown.
    3. If cache miss → call LlamaParse API → cache the result → return.

    LlamaParse returns high-fidelity Markdown with:
      – Headings preserved as `#` / `##` / `###`
      – Tables rendered in Markdown table syntax
      – Figures tagged as `![image](…)`
      – `---` page separators (when using `page_separator`)
    """
    PAGE_SEP = "---PAGE_BREAK---"

    # ── 1. Try loading from cache ──
    cached_md = _load_from_cache(pdf_path)
    if cached_md is not None:
        return _markdown_to_parsed_doc(cached_md, pdf_path, PAGE_SEP)

    # ── 2. Call LlamaParse API ──
    try:
        from llama_parse import LlamaParse
        from config import get_settings

        settings = get_settings()

        parser = LlamaParse(
            api_key=settings.llama_cloud_api_key,
            result_type="markdown",          # structured Markdown output
            parsing_instruction=(
                "This is a car brochure PDF. Extract ALL text including "
                "specifications tables, feature lists, dimensions, engine "
                "details, pricing, variants, and colour options. Preserve "
                "headings, sub-headings, bullet lists, and table structures. "
                "Mark tables using Markdown table syntax. Keep numbers and "
                "units exact (e.g. '188 km/h', '200 mm')."
            ),
            page_separator=f"\n\n{PAGE_SEP}\n\n",
            verbose=False,
        )

        # LlamaParse's load_data is sync but internally uses async
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                documents = pool.submit(
                    lambda: parser.load_data(str(pdf_path))
                ).result()
        else:
            documents = parser.load_data(str(pdf_path))

        if not documents:
            logger.warning(f"LlamaParse returned no documents for {pdf_path.name}")
            return None

        # Concatenate all returned documents
        full_md = "\n\n".join(doc.text for doc in documents if doc.text)
        if not full_md.strip():
            return None

        # Extract metadata from first document
        meta = {}
        if hasattr(documents[0], "metadata"):
            meta = documents[0].metadata or {}

        # ── 3. Save to cache ──
        _save_to_cache(pdf_path, full_md, meta)

        return _markdown_to_parsed_doc(full_md, pdf_path, PAGE_SEP)

    except ImportError:
        logger.warning("llama-parse not installed — run: pip install llama-parse")
        return None
    except Exception as e:
        logger.warning(f"LlamaParse failed for {pdf_path.name}: {e}")
        return None


def _markdown_to_parsed_doc(
    full_md: str, pdf_path: Path, page_separator: str
) -> Optional[ParsedDocument]:
    """Convert raw LlamaParse markdown (or cached markdown) into a ParsedDocument."""
    raw_pages = re.split(re.escape(page_separator), full_md)
    pages: list[PageContent] = []
    for i, page_text in enumerate(raw_pages):
        cleaned = page_text.strip()
        if not cleaned:
            continue
        cleaned = _wrap_markdown_tables(cleaned)
        pages.append(PageContent(
            page_number=i + 1,
            text=cleaned,
        ))

    if not pages:
        return None

    return ParsedDocument(
        file_name=pdf_path.name,
        file_path=str(pdf_path),
        total_pages=len(pages),
        pages=pages,
        metadata={"extractor": "llamaparse"},
    )


def _wrap_markdown_tables(text: str) -> str:
    """
    Find Markdown tables (lines starting with `|`) and wrap them in
    [TABLE] … [/TABLE] tags for downstream processing.
    """
    lines = text.split("\n")
    result: list[str] = []
    in_table = False

    for line in lines:
        stripped = line.strip()
        is_table_line = stripped.startswith("|") and stripped.endswith("|")

        if is_table_line and not in_table:
            result.append("[TABLE]")
            in_table = True
        elif not is_table_line and in_table:
            result.append("[/TABLE]")
            in_table = False

        result.append(line)

    if in_table:
        result.append("[/TABLE]")

    return "\n".join(result)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  PyMuPDF extractor (offline fallback)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _extract_with_pymupdf(pdf_path: Path) -> Optional[ParsedDocument]:
    """Fallback extraction using PyMuPDF — local, no API key needed."""
    try:
        doc = fitz.open(str(pdf_path))
        pages: list[PageContent] = []
        for i, page in enumerate(doc):
            text = page.get_text("text") or ""
            pages.append(PageContent(
                page_number=i + 1,
                text=text.strip(),
            ))

        metadata = doc.metadata or {}
        doc.close()

        return ParsedDocument(
            file_name=pdf_path.name,
            file_path=str(pdf_path),
            total_pages=len(pages),
            pages=pages,
            metadata={
                "extractor": "pymupdf",
                "info": metadata,
            },
        )
    except Exception as e:
        logger.error(f"PyMuPDF failed for {pdf_path.name}: {e}")
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_pdf(pdf_path: Path) -> ParsedDocument:
    """
    Parse a PDF file and extract text content.

    Strategy (ordered by quality):
    1. **LlamaParse** — cloud API; best for complex brochures with tables,
       figures, and multi-column layouts.  Requires LLAMA_CLOUD_API_KEY.
    2. **PyMuPDF** — local fallback; no API key needed, decent quality.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Parsing PDF: {pdf_path.name}")

    # ── Try LlamaParse first ──
    if _llamaparse_available():
        logger.info(f"  Using LlamaParse for {pdf_path.name}")
        doc = _extract_with_llamaparse(pdf_path)
        if doc and doc.full_text.strip():
            logger.info(
                f"✓ Extracted {doc.total_pages} pages from {doc.file_name} "
                f"({len(doc.full_text)} chars) [llamaparse]"
            )
            return doc
        logger.warning(f"  LlamaParse returned empty for {pdf_path.name}, falling back")
    else:
        logger.info("  LlamaParse not available (no API key), using fallback")

    # ── Fallback to PyMuPDF ──
    logger.info(f"  Using PyMuPDF for {pdf_path.name}")
    doc = _extract_with_pymupdf(pdf_path)
    if doc and doc.full_text.strip():
        logger.info(
            f"✓ Extracted {doc.total_pages} pages from {doc.file_name} "
            f"({len(doc.full_text)} chars) [pymupdf]"
        )
        return doc

    raise ValueError(f"Failed to extract text from {pdf_path.name} with any extractor.")


def parse_all_pdfs(data_dir: Path) -> list[ParsedDocument]:
    """Parse all PDFs in the given directory."""
    pdf_files = sorted(data_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {data_dir}")
        return []

    logger.info(f"Found {len(pdf_files)} PDF(s) in {data_dir}")

    extractor = "LlamaParse" if _llamaparse_available() else "PyMuPDF (fallback)"
    logger.info(f"Primary extractor: {extractor}")

    documents = []
    for pdf_path in pdf_files:
        try:
            doc = parse_pdf(pdf_path)
            documents.append(doc)
        except Exception as e:
            logger.error(f"✗ Skipping {pdf_path.name}: {e}")

    logger.info(f"Successfully parsed {len(documents)}/{len(pdf_files)} PDFs")
    return documents

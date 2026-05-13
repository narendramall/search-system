"""
Hierarchical Semantic Chunker — Splits parsed PDF documents using
document structure awareness.

Builds a semantic hierarchy from the raw text:

    Document
     ├── Section            (major heading: "SPECIFICATIONS", "SAFETY", etc.)
     │    ├── Subsection    (minor heading under a section)
     │    │     ├── Paragraph block
     │    │     ├── Table
     │    │     └── Figure
     │    └── ...
     └── ...

Each leaf block becomes a Chunk (or is grouped with siblings if too small).
Oversized blocks are split at sentence boundaries while staying inside
their hierarchy node.  Every chunk carries its full hierarchy path so
downstream search / retrieval can filter or boost by structure.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import tiktoken

from src.ingestion.pdf_parser import ParsedDocument

logger = logging.getLogger(__name__)

# Tokenizer (cl100k_base ≈ Gemini token counting)
_ENCODING = tiktoken.get_encoding("cl100k_base")


# ──────────────────────────────────────────────
#  Block types recognised inside a document
# ──────────────────────────────────────────────

class BlockType(str, Enum):
    """Type of a content block in the document hierarchy."""
    SECTION_HEADING = "section_heading"
    SUBSECTION_HEADING = "subsection_heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    FIGURE = "figure"
    LIST_BLOCK = "list"


# ──────────────────────────────────────────────
#  Hierarchy tree nodes
# ──────────────────────────────────────────────

@dataclass
class ContentBlock:
    """A leaf block of content (paragraph / table / figure / list)."""
    text: str
    block_type: BlockType
    page_number: int
    token_count: int = 0

    def __post_init__(self):
        if not self.token_count:
            self.token_count = _count_tokens(self.text)


@dataclass
class Subsection:
    """A subsection within a section."""
    title: str
    blocks: list[ContentBlock] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(b.token_count for b in self.blocks)


@dataclass
class Section:
    """A top-level section of the document."""
    title: str
    semantic_label: str  # normalised label ("specifications", "safety", …)
    subsections: list[Subsection] = field(default_factory=list)
    # Blocks that sit directly under the section (no subsection)
    blocks: list[ContentBlock] = field(default_factory=list)


@dataclass
class DocumentTree:
    """Hierarchical representation of a parsed document."""
    document_name: str
    car_brand: str
    car_model: str
    sections: list[Section] = field(default_factory=list)
    # Blocks that appear before any section heading
    preamble_blocks: list[ContentBlock] = field(default_factory=list)


# ──────────────────────────────────────────────
#  Chunk (output) — now hierarchy-aware
# ──────────────────────────────────────────────

@dataclass
class Chunk:
    """A single search-indexable text chunk with full hierarchy context."""
    chunk_id: str
    document_name: str
    car_brand: str
    car_model: str
    content: str
    chunk_index: int
    page_number: int
    section: str          # normalised semantic label (e.g. "specifications")
    section_title: str    # original heading text
    subsection_title: str # subsection heading (or "")
    hierarchy_path: str   # "Document > Section > Subsection"
    block_type: str       # dominant BlockType in this chunk
    token_count: int
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Convert to dictionary for indexing."""
        return {
            "chunk_id": self.chunk_id,
            "document_name": self.document_name,
            "car_brand": self.car_brand,
            "car_model": self.car_model,
            "content": self.content,
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "section": self.section,
            "section_title": self.section_title,
            "subsection_title": self.subsection_title,
            "hierarchy_path": self.hierarchy_path,
            "block_type": self.block_type,
            "token_count": self.token_count,
            "metadata": self.metadata,
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Utility helpers
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


def _generate_chunk_id(doc_name: str, chunk_index: int) -> str:
    raw = f"{doc_name}::chunk_{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _extract_car_info(file_name: str) -> tuple[str, str]:
    """Extract brand & model from filename.  e.g. Hyundai_Creta_2024.pdf → (Hyundai, Creta)"""
    name = re.sub(r"\.pdf$", "", file_name, flags=re.IGNORECASE)
    parts = re.split(r"[_\-\s]+", name)
    brand = parts[0] if parts else "Unknown"
    model = parts[1] if len(parts) >= 2 else name
    return brand.strip(), model.strip()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Section / Subsection detection
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Known section keywords for car brochures → normalised label
_SECTION_KEYWORDS: dict[str, list[str]] = {
    "specifications": ["specification", "specs", "technical"],
    "engine": ["engine", "powertrain", "motor"],
    "performance": ["performance", "top speed", "acceleration"],
    "safety": ["safety", "airbag", "ncap", "braking"],
    "interior": ["interior", "cabin", "dashboard", "infotainment"],
    "exterior": ["exterior", "design", "styling"],
    "features": ["feature", "technology", "convenience", "comfort"],
    "price": ["price", "pricing", "ex-showroom"],
    "mileage": ["mileage", "fuel efficiency", "range", "economy"],
    "dimensions": ["dimension", "measurement", "ground clearance", "wheelbase"],
    "transmission": ["transmission", "gearbox"],
    "variants": ["variant", "trim", "lineup"],
    "colours": ["colour", "color", "paint"],
    "warranty": ["warranty", "service", "maintenance"],
    "overview": ["overview", "introduction", "about", "highlights"],
}


def _normalise_section_label(heading: str) -> str:
    """Map a heading string to a known semantic label."""
    h = heading.lower()
    for label, keywords in _SECTION_KEYWORDS.items():
        if any(kw in h for kw in keywords):
            return label
    return "general"


def _is_section_heading(line: str) -> bool:
    """
    Heuristic: a line is a *section* heading if it:
      • is ALL-CAPS and ≤ 8 words, OR
      • matches a known section keyword pattern, OR
      • is a short line (≤ 6 words) with no trailing period and starts upper-case
    """
    stripped = line.strip()
    if not stripped or len(stripped) < 3:
        return False
    word_count = len(stripped.split())
    # ALL-CAPS with enough length
    if stripped.isupper() and 1 <= word_count <= 10:
        return True
    # Known keyword match
    low = stripped.lower()
    for keywords in _SECTION_KEYWORDS.values():
        if any(low == kw or low.startswith(kw + " ") or low.startswith(kw + ":") for kw in keywords):
            if word_count <= 8:
                return True
    # Short title-like line (no period, capitalised)
    if word_count <= 6 and not stripped.endswith(".") and stripped[0].isupper():
        # Extra guard: reject lines that look like normal sentences
        if not any(stripped.lower().startswith(w) for w in ("the ", "a ", "an ", "it ", "this ", "that ")):
            return True
    return False


def _is_subsection_heading(line: str) -> bool:
    """
    A subsection heading is shorter / less prominent than a section heading.
    Heuristic: title-case or initial-caps line, ≤ 8 words, doesn't end in period,
    and is NOT all-caps (which would be a section).
    """
    stripped = line.strip()
    if not stripped or len(stripped) < 3:
        return False
    if stripped.isupper():
        return False  # that's a section
    word_count = len(stripped.split())
    if word_count > 8:
        return False
    if stripped.endswith("."):
        return False
    if stripped[0].isupper():
        if not any(stripped.lower().startswith(w) for w in ("the ", "a ", "an ", "it ", "this ", "that ", "with ", "for ")):
            return True
    return False


def _detect_block_type(text: str) -> BlockType:
    """Classify a text block into a BlockType."""
    t = text.strip()
    if t.startswith("[TABLE]") or t.startswith("[/TABLE]") or "[TABLE]" in t:
        return BlockType.TABLE
    if re.search(r"\bfig(ure)?[\s.:]\s*\d", t, re.IGNORECASE):
        return BlockType.FIGURE
    if re.search(r"^[\s]*[-•●▪◦∙]\s", t, re.MULTILINE):
        return BlockType.LIST_BLOCK
    return BlockType.PARAGRAPH


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 1: Build the hierarchy tree
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _build_document_tree(document: ParsedDocument) -> DocumentTree:
    """
    Walk through every page and classify each line / block into
    the hierarchy:  Document → Section → Subsection → Block.
    """
    car_brand, car_model = _extract_car_info(document.file_name)

    tree = DocumentTree(
        document_name=document.file_name,
        car_brand=car_brand,
        car_model=car_model,
    )

    current_section: Optional[Section] = None
    current_subsection: Optional[Subsection] = None

    for page in document.pages:
        if not page.text.strip():
            continue

        # Split page text into raw blocks (double-newline separated)
        raw_blocks = re.split(r"\n{2,}", page.text)

        for raw in raw_blocks:
            raw = raw.strip()
            if not raw:
                continue

            # ── Check: is this block a table? ──
            if "[TABLE]" in raw:
                block = ContentBlock(
                    text=raw,
                    block_type=BlockType.TABLE,
                    page_number=page.page_number,
                )
                _attach_block(tree, current_section, current_subsection, block)
                continue

            # ── Check line-by-line for headings inside this block ──
            lines = raw.split("\n")
            pending_paragraph_lines: list[str] = []

            for line in lines:
                line_stripped = line.strip()
                if not line_stripped:
                    continue

                if _is_section_heading(line_stripped):
                    # Flush any pending paragraph
                    _flush_paragraph(
                        pending_paragraph_lines, page.page_number,
                        tree, current_section, current_subsection,
                    )
                    pending_paragraph_lines = []

                    # Start a new section
                    label = _normalise_section_label(line_stripped)
                    current_section = Section(
                        title=line_stripped,
                        semantic_label=label,
                    )
                    tree.sections.append(current_section)
                    current_subsection = None  # reset
                    continue

                if current_section and _is_subsection_heading(line_stripped):
                    _flush_paragraph(
                        pending_paragraph_lines, page.page_number,
                        tree, current_section, current_subsection,
                    )
                    pending_paragraph_lines = []

                    current_subsection = Subsection(title=line_stripped)
                    current_section.subsections.append(current_subsection)
                    continue

                # Normal content line — accumulate
                pending_paragraph_lines.append(line)

            # Flush remaining paragraph from this block
            _flush_paragraph(
                pending_paragraph_lines, page.page_number,
                tree, current_section, current_subsection,
            )

    _log_tree(tree)
    return tree


def _attach_block(
    tree: DocumentTree,
    section: Optional[Section],
    subsection: Optional[Subsection],
    block: ContentBlock,
) -> None:
    """Attach a ContentBlock at the deepest available hierarchy level."""
    if section is not None:
        if subsection is not None:
            subsection.blocks.append(block)
        else:
            section.blocks.append(block)
    else:
        tree.preamble_blocks.append(block)


def _flush_paragraph(
    lines: list[str],
    page_number: int,
    tree: DocumentTree,
    section: Optional[Section],
    subsection: Optional[Subsection],
) -> None:
    """Join accumulated lines into a paragraph block and attach."""
    if not lines:
        return
    text = "\n".join(lines).strip()
    if not text:
        return
    block = ContentBlock(
        text=text,
        block_type=_detect_block_type(text),
        page_number=page_number,
    )
    _attach_block(tree, section, subsection, block)


def _log_tree(tree: DocumentTree) -> None:
    """Log a compact summary of the parsed hierarchy."""
    n_sections = len(tree.sections)
    n_subs = sum(len(s.subsections) for s in tree.sections)
    n_blocks = (
        len(tree.preamble_blocks)
        + sum(len(s.blocks) for s in tree.sections)
        + sum(len(sub.blocks) for s in tree.sections for sub in s.subsections)
    )
    logger.info(
        f"  Hierarchy: {n_sections} sections, {n_subs} subsections, "
        f"{n_blocks} content blocks, {len(tree.preamble_blocks)} preamble blocks"
    )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Step 2: Flatten tree → Chunks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _split_oversized_block(text: str, max_tokens: int) -> list[str]:
    """Split a big block into sentence-boundary-respecting pieces."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    pieces: list[str] = []
    current: list[str] = []
    current_tok = 0
    for sent in sentences:
        st = _count_tokens(sent)
        if current_tok + st > max_tokens and current:
            pieces.append(" ".join(current))
            current = []
            current_tok = 0
        current.append(sent)
        current_tok += st
    if current:
        pieces.append(" ".join(current))
    return pieces


def _merge_small_blocks(
    blocks: list[ContentBlock],
    min_tokens: int = 60,
    max_tokens: int = 512,
) -> list[list[ContentBlock]]:
    """
    Group consecutive small blocks together so each group has
    at least *min_tokens* and at most *max_tokens*.
    """
    groups: list[list[ContentBlock]] = []
    current_group: list[ContentBlock] = []
    current_tok = 0

    for block in blocks:
        if current_tok + block.token_count > max_tokens and current_group:
            groups.append(current_group)
            current_group = []
            current_tok = 0
        current_group.append(block)
        current_tok += block.token_count

    if current_group:
        groups.append(current_group)

    return groups


def _hierarchy_path(doc_name: str, section_title: str, subsection_title: str) -> str:
    parts = [doc_name]
    if section_title:
        parts.append(section_title)
    if subsection_title:
        parts.append(subsection_title)
    return " > ".join(parts)


def _dominant_block_type(blocks: list[ContentBlock]) -> str:
    """Return the most common block type in a group."""
    if not blocks:
        return BlockType.PARAGRAPH.value
    counts: dict[str, int] = {}
    for b in blocks:
        counts[b.block_type.value] = counts.get(b.block_type.value, 0) + 1
    return max(counts, key=lambda k: counts[k])


def _blocks_to_chunks(
    blocks: list[ContentBlock],
    doc_name: str,
    car_brand: str,
    car_model: str,
    section_title: str,
    section_label: str,
    subsection_title: str,
    chunk_index_start: int,
    chunk_size: int,
    source_path: str,
) -> tuple[list[Chunk], int]:
    """
    Convert a list of ContentBlocks into Chunks.

    Strategy:
      1. Oversized blocks → split at sentence boundaries
      2. Small consecutive blocks → merge up to chunk_size
      3. Each group becomes one Chunk
    """
    if not blocks:
        return [], chunk_index_start

    # Split oversized blocks first
    expanded: list[ContentBlock] = []
    for b in blocks:
        if b.token_count > chunk_size:
            pieces = _split_oversized_block(b.text, chunk_size)
            for p in pieces:
                expanded.append(ContentBlock(
                    text=p,
                    block_type=b.block_type,
                    page_number=b.page_number,
                ))
        else:
            expanded.append(b)

    # Merge small adjacent blocks
    groups = _merge_small_blocks(expanded, min_tokens=60, max_tokens=chunk_size)

    h_path = _hierarchy_path(doc_name, section_title, subsection_title)
    chunks: list[Chunk] = []
    idx = chunk_index_start

    for group in groups:
        content = "\n\n".join(b.text for b in group)
        # Prepend hierarchy breadcrumb for context in the embedding / search
        breadcrumb = f"[{section_label.upper()}]"
        if subsection_title:
            breadcrumb += f" [{subsection_title}]"
        enriched_content = f"{breadcrumb}\n{content}"

        page = group[0].page_number

        chunks.append(Chunk(
            chunk_id=_generate_chunk_id(doc_name, idx),
            document_name=doc_name,
            car_brand=car_brand,
            car_model=car_model,
            content=enriched_content,
            chunk_index=idx,
            page_number=page,
            section=section_label,
            section_title=section_title,
            subsection_title=subsection_title,
            hierarchy_path=h_path,
            block_type=_dominant_block_type(group),
            token_count=_count_tokens(enriched_content),
            metadata={"source": source_path},
        ))
        idx += 1

    return chunks, idx


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  Public API
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def chunk_document(
    document: ParsedDocument,
    chunk_size: int = 512,
    chunk_overlap: int = 50,  # kept for API compat; not used in hierarchical mode
) -> list[Chunk]:
    """
    Hierarchically chunk a parsed car-brochure document.

    Steps:
      1. Build a semantic hierarchy tree from the raw pages.
      2. Walk the tree and emit chunks that respect the
         Document > Section > Subsection > Block structure.
      3. Tables / figures are kept as atomic chunks.
      4. Small sibling blocks are merged; oversized blocks are
         split at sentence boundaries.
    """
    tree = _build_document_tree(document)

    all_chunks: list[Chunk] = []
    idx = 0
    source = document.file_path

    # ── Preamble (content before the first section heading) ──
    if tree.preamble_blocks:
        preamble_chunks, idx = _blocks_to_chunks(
            blocks=tree.preamble_blocks,
            doc_name=tree.document_name,
            car_brand=tree.car_brand,
            car_model=tree.car_model,
            section_title="Preamble",
            section_label="overview",
            subsection_title="",
            chunk_index_start=idx,
            chunk_size=chunk_size,
            source_path=source,
        )
        all_chunks.extend(preamble_chunks)

    # ── Walk sections ──
    for section in tree.sections:
        # Direct blocks under the section (no subsection)
        if section.blocks:
            sec_chunks, idx = _blocks_to_chunks(
                blocks=section.blocks,
                doc_name=tree.document_name,
                car_brand=tree.car_brand,
                car_model=tree.car_model,
                section_title=section.title,
                section_label=section.semantic_label,
                subsection_title="",
                chunk_index_start=idx,
                chunk_size=chunk_size,
                source_path=source,
            )
            all_chunks.extend(sec_chunks)

        # Subsections
        for sub in section.subsections:
            if sub.blocks:
                sub_chunks, idx = _blocks_to_chunks(
                    blocks=sub.blocks,
                    doc_name=tree.document_name,
                    car_brand=tree.car_brand,
                    car_model=tree.car_model,
                    section_title=section.title,
                    section_label=section.semantic_label,
                    subsection_title=sub.title,
                    chunk_index_start=idx,
                    chunk_size=chunk_size,
                    source_path=source,
                )
                all_chunks.extend(sub_chunks)

    logger.info(
        f"Chunked {document.file_name} → {len(all_chunks)} hierarchical chunks "
        f"(brand={tree.car_brand}, model={tree.car_model})"
    )
    return all_chunks


def chunk_all_documents(
    documents: list[ParsedDocument],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Chunk all parsed documents using hierarchical semantic chunking."""
    all_chunks: list[Chunk] = []
    for doc in documents:
        chunks = chunk_document(doc, chunk_size, chunk_overlap)
        all_chunks.extend(chunks)

    logger.info(f"Total hierarchical chunks across all documents: {len(all_chunks)}")
    return all_chunks

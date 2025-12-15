# ETL Pipeline Plan: PDF Document Processing for RAG

## Overview

This document outlines the architecture and implementation plan for an ETL pipeline that processes PDF documents for use in RAG (Retrieval-Augmented Generation) systems.

### Goals
- Extract text and images from PDF files
- Transform data into chunks suitable for LLM context and vector search
- Load processed data into databases (local for testing, cloud for production)
- **Support incremental processing** for dynamic document collections

---

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   PDFs          │────▶│   EXTRACT    │────▶│  MARKDOWN FILES │────▶│   TRANSFORM     │
│  (Dynamic count)│     │  Text+Images │     │  (Checkpoint)   │     │  Chunk+Embed    │
└─────────────────┘     └──────────────┘     └─────────────────┘     └────────┬────────┘
        ▲                       │                                              │
        │ Upload API            │ Images                                       │
        │ (Future UI)           ▼                                              │
                    ┌───────────────────┐                                      │
                    │  data/images/     │                                      │
                    │  {doc_id}/        │                                      │
                    └───────────────────┘                                      │
                    ┌──────────────────────────────────────────────────────────┴───────┐
                    │                              LOAD                                │
                    │  ┌─────────────────────┐           ┌────────────────────┐        │
                    │  │   LOCAL (Testing)   │           │ PRODUCTION (Cloud) │        │
                    │  │  - SQLite (metadata)│           │ - Supabase/pgvector│        │
                    │  │  - LanceDB (vectors)│           │ - Cloud Storage    │        │
                    │  │  - Local filesystem │           │   (images)         │        │
                    │  └─────────────────────┘           └────────────────────┘        │
                    └──────────────────────────────────────────────────────────────────┘
```

---

## Markdown File Storage

Extracted text is **always persisted as markdown files** before transformation. This serves as a checkpoint between Extract and Transform phases.

### Rationale

| Benefit | Description |
|---------|-------------|
| **Debugging** | Easy to inspect extraction quality by viewing markdown |
| **QA Workflow** | Review/correct extracted text before indexing (critical for medical content) |
| **Faster Iteration** | Re-chunk/re-embed without re-extracting from PDFs |
| **Clear Separation** | Each pipeline phase can run independently |
| **Audit Trail** | Human-readable record of what was indexed |
| **Manual Correction** | Fix OCR errors or add missing content from images |

### Storage Structure

```
data/
├── markdown/
│   ├── {document_id}.md              # Extracted text in markdown format
│   └── ...
├── images/
│   ├── {document_id}/
│   │   ├── page_001_img_001.png
│   │   ├── page_001_img_002.png
│   │   └── ...
│   └── ...
├── lancedb/                          # Vector store
└── sqlite/                           # Metadata database
```

### Markdown File Format

Each markdown file includes metadata header and extracted content:

```markdown
---
document_id: "abc-123-def"
filename: "EP001 Nutrition During Pregnancy.pdf"
title: "Nutrition During Pregnancy"
page_count: 4
extracted_at: "2025-01-15T10:30:00Z"
extraction_method: "pymupdf"
file_hash: "sha256:abc123..."
---

# Nutrition During Pregnancy

## Why Nutrition Matters

Good nutrition during pregnancy is essential for...

## Key Nutrients

### Folic Acid
...

### Iron
...

[Image: page_001_img_001.png - Caption: Food pyramid diagram]
```

### Pipeline Flow with Markdown Checkpoint

```python
class Pipeline:
    def process_document(self, pdf_path: str) -> str:
        """Full pipeline with markdown checkpoint."""
        doc_id = generate_id()

        # Phase 1: Extract → Markdown file
        markdown_path = self.extract_to_markdown(pdf_path, doc_id)

        # Phase 2: Transform (reads from markdown)
        chunks = self.transform_markdown(markdown_path, doc_id)

        # Phase 3: Load
        self.load_chunks(chunks)

        return doc_id

    def reprocess_from_markdown(self, doc_id: str):
        """Re-transform without re-extracting."""
        markdown_path = self.paths.markdown_dir / f"{doc_id}.md"
        if not markdown_path.exists():
            raise ValueError(f"Markdown not found for {doc_id}")

        # Delete old chunks
        self.delete_chunks(doc_id)

        # Re-transform and load
        chunks = self.transform_markdown(markdown_path, doc_id)
        self.load_chunks(chunks)
```

### Sync with Source PDFs

When a PDF changes (detected via file hash), the markdown is regenerated:

```python
def check_and_update(self, pdf_path: str):
    current_hash = compute_file_hash(pdf_path)
    existing = self.db.get_document_by_path(pdf_path)

    if existing and existing.file_hash != current_hash:
        # PDF changed - regenerate markdown
        self.delete_markdown(existing.id)
        self.delete_chunks(existing.id)
        self.process_document(pdf_path)  # Full reprocess
```

---

## Incremental Processing Design

The pipeline supports **dynamic document collections** where PDFs can be added/removed over time.

### Document State Management

```python
from enum import Enum

class DocumentStatus(Enum):
    PENDING = "pending"       # Newly uploaded, not processed
    PROCESSING = "processing" # Currently being processed
    COMPLETED = "completed"   # Successfully processed
    FAILED = "failed"         # Processing failed
    OUTDATED = "outdated"     # Source file changed, needs reprocessing

@dataclass
class DocumentState:
    id: str
    filename: str
    title: str               # Extracted/parsed document title
    file_hash: str           # SHA-256 for change detection
    status: DocumentStatus
    source_path: str         # Path to original PDF
    markdown_path: str       # Path to extracted markdown file
    extraction_method: str   # "pymupdf" or "google_docai"
    page_count: int
    uploaded_at: datetime
    processed_at: Optional[datetime]
    error_message: Optional[str]
```

### Incremental Pipeline Logic

```python
class IncrementalPipeline:
    def process_new_documents(self):
        """Process only new or changed documents."""
        for pdf_path in self.scan_pdf_directory():
            file_hash = self.compute_hash(pdf_path)
            existing = self.db.get_document_by_path(pdf_path)

            if existing is None:
                # New document - full pipeline
                self.process_document(pdf_path)
            elif existing.file_hash != file_hash:
                # Document changed - reprocess from scratch
                self.delete_document_data(existing.id)
                self.process_document(pdf_path)
            # else: unchanged, skip

    def process_document(self, pdf_path: str) -> str:
        """Full pipeline: Extract → Markdown → Transform → Load."""
        doc_id = generate_id()

        # Phase 1: Extract to markdown file
        markdown_path = self.extract_to_markdown(pdf_path, doc_id)

        # Phase 2: Transform markdown to chunks
        chunks = self.transform_markdown(markdown_path, doc_id)

        # Phase 3: Load to database
        self.load_chunks(chunks)

        return doc_id

    def reprocess_from_markdown(self, document_id: str):
        """Re-transform and re-load without re-extracting."""
        doc = self.db.get_document(document_id)
        if not Path(doc.markdown_path).exists():
            raise ValueError(f"Markdown not found: {doc.markdown_path}")

        # Delete old chunks only (keep markdown)
        self.vector_store.delete(where={"document_id": document_id})
        self.db.delete_chunks(document_id)

        # Re-transform and load
        chunks = self.transform_markdown(doc.markdown_path, document_id)
        self.load_chunks(chunks)

    def delete_document(self, document_id: str):
        """Remove document and ALL associated data including markdown."""
        doc = self.db.get_document(document_id)

        # Delete markdown file
        if doc.markdown_path and Path(doc.markdown_path).exists():
            Path(doc.markdown_path).unlink()

        # Delete images directory
        images_dir = self.paths.images_dir / document_id
        if images_dir.exists():
            shutil.rmtree(images_dir)

        # Delete from vector store
        self.vector_store.delete(where={"document_id": document_id})

        # Delete from SQLite
        self.db.delete_chunks(document_id)
        self.db.delete_images(document_id)
        self.db.delete_document(document_id)
```

### Future Upload API Interface

```python
# Designed for future UI integration
class DocumentAPI:
    async def upload(self, file: UploadFile) -> DocumentResponse:
        """Upload and process a single PDF."""

    async def upload_batch(self, files: list[UploadFile]) -> list[DocumentResponse]:
        """Upload and process multiple PDFs."""

    async def get_status(self, document_id: str) -> DocumentStatus:
        """Check processing status."""

    async def delete(self, document_id: str) -> bool:
        """Delete document and associated data."""

    async def list_documents(self, status: Optional[DocumentStatus] = None) -> list[DocumentSummary]:
        """List all documents with optional status filter."""
```

---

## Phase 1: Extract

### Option A: Python-Native (Recommended for Cost/Simplicity)

**Library: PyMuPDF (pymupdf4llm)**

Best balance of speed, quality, and RAG-readiness based on 2025 benchmarks.

```python
# Example extraction approach
import pymupdf4llm
import fitz  # PyMuPDF

# Text extraction with markdown formatting
md_text = pymupdf4llm.to_markdown("document.pdf")

# Image extraction
doc = fitz.open("document.pdf")
for page_num, page in enumerate(doc):
    images = page.get_images(full=True)
    for img_idx, img in enumerate(images):
        # Extract and save images
```

**Pros:**
- Free, no API costs
- Fast (0.12s per document)
- Excellent markdown output for RAG
- Good image extraction
- Works offline

**Cons:**
- May struggle with complex layouts
- OCR for scanned documents requires additional setup

### Option B: Google Cloud Document AI

**Use Case:** Scanned documents or complex layouts requiring OCR.

```python
from google.cloud import documentai_v1 as documentai

def process_document(project_id, location, processor_id, file_path):
    client = documentai.DocumentProcessorServiceClient()
    name = client.processor_path(project_id, location, processor_id)

    with open(file_path, "rb") as f:
        content = f.read()

    request = documentai.ProcessRequest(
        name=name,
        raw_document=documentai.RawDocument(content=content, mime_type="application/pdf")
    )
    result = client.process_document(request=request)
    return result.document
```

**Pros:**
- Superior OCR accuracy
- Handles complex layouts
- Entity extraction capabilities

**Cons:**
- Costs money (per page pricing)
- Requires GCP setup
- Network dependency

### Recommendation

Use a **hybrid approach**:
1. Default to PyMuPDF for all documents
2. Fall back to Google Document AI for documents with poor extraction quality

### Extraction Quality Detection (Text Density Heuristics)

To automatically detect when PyMuPDF extraction fails and Google Document AI is needed:

```python
@dataclass
class ExtractionQualityMetrics:
    chars_per_page: float        # Expected: 500-3000 for text PDFs
    words_per_page: float        # Expected: 100-600 for text PDFs
    avg_word_length: float       # Expected: 4-8 for English text
    whitespace_ratio: float      # Expected: 0.1-0.3
    non_ascii_ratio: float       # Expected: < 0.05 for English
    empty_pages_ratio: float     # Expected: < 0.1

class ExtractionQualityChecker:
    def __init__(self):
        self.thresholds = {
            "min_chars_per_page": 100,      # Below = likely scanned/image
            "min_words_per_page": 20,       # Below = extraction failed
            "max_whitespace_ratio": 0.5,    # Above = garbled extraction
            "max_non_ascii_ratio": 0.15,    # Above = encoding issues
            "max_empty_pages": 0.3,         # Above = partial extraction
            "min_avg_word_length": 2,       # Below = OCR garbage
            "max_avg_word_length": 15,      # Above = no word boundaries
        }

    def analyze(self, text: str, page_count: int) -> ExtractionQualityMetrics:
        words = text.split()
        return ExtractionQualityMetrics(
            chars_per_page=len(text) / max(page_count, 1),
            words_per_page=len(words) / max(page_count, 1),
            avg_word_length=sum(len(w) for w in words) / max(len(words), 1),
            whitespace_ratio=text.count(' ') / max(len(text), 1),
            non_ascii_ratio=sum(1 for c in text if ord(c) > 127) / max(len(text), 1),
            empty_pages_ratio=0  # Calculated per-page during extraction
        )

    def needs_ocr_fallback(self, metrics: ExtractionQualityMetrics) -> tuple[bool, str]:
        """Returns (needs_fallback, reason)."""
        if metrics.chars_per_page < self.thresholds["min_chars_per_page"]:
            return True, "Low character density - likely scanned document"
        if metrics.words_per_page < self.thresholds["min_words_per_page"]:
            return True, "Low word count - extraction may have failed"
        if metrics.avg_word_length < self.thresholds["min_avg_word_length"]:
            return True, "Very short words - possible OCR garbage"
        if metrics.avg_word_length > self.thresholds["max_avg_word_length"]:
            return True, "No word boundaries detected"
        if metrics.non_ascii_ratio > self.thresholds["max_non_ascii_ratio"]:
            return True, "High non-ASCII ratio - encoding issues"
        return False, "Quality acceptable"

# Usage in pipeline
def extract_with_fallback(pdf_path: str) -> ExtractedDocument:
    # Try PyMuPDF first
    result = pymupdf_extract(pdf_path)
    metrics = quality_checker.analyze(result.text, result.page_count)
    needs_fallback, reason = quality_checker.needs_ocr_fallback(metrics)

    if needs_fallback:
        logger.info(f"Falling back to Google Document AI: {reason}")
        result = google_docai_extract(pdf_path)
        result.extraction_method = "google_docai"
        result.fallback_reason = reason

    return result
```

### Image Storage Strategy

```
data/
├── images/
│   ├── {document_id}/
│   │   ├── page_001_img_001.png
│   │   ├── page_001_img_002.png
│   │   └── ...
│   └── ...
└── metadata/
    └── image_registry.json  # Maps images to source documents
```

**Image Metadata Schema:**
```json
{
  "image_id": "uuid",
  "document_id": "uuid",
  "page_number": 1,
  "position": {"x": 100, "y": 200, "width": 300, "height": 400},
  "file_path": "data/images/{doc_id}/page_001_img_001.png",
  "caption": "extracted or generated caption",
  "embedding": [...]  // Optional: CLIP embedding for image search
}
```

---

## Phase 2: Transform

### Text Processing Pipeline

```
Raw Text ─▶ Clean ─▶ Chunk ─▶ Embed ─▶ Store
```

#### 2.1 Text Cleaning
- Remove headers/footers
- Normalize whitespace
- Handle special characters
- Preserve semantic structure (headings, lists, tables)

#### 2.2 Chunking Strategy

**Why Not Size-Only Chunking?**

Fixed-size chunking (e.g., "split every 1000 characters") has problems:
- Splits mid-sentence or mid-paragraph, breaking semantic meaning
- Loses document structure context (headings, sections)
- Creates chunks that don't align with natural topic boundaries
- Poor retrieval quality when a concept spans chunk boundaries

**Recommended: Multi-Strategy Approach**

Since PyMuPDF outputs **Markdown**, we use structure-aware chunking first, then fall back to semantic chunking.

```python
from langchain.text_splitter import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings

class HybridChunker:
    """
    Strategy priority:
    1. Markdown headers (if document has clear structure)
    2. Semantic chunking (if embeddings available)
    3. Recursive character splitting (fallback)
    """

    def __init__(self, use_semantic: bool = True):
        self.use_semantic = use_semantic

        # Level 1: Markdown structure-aware splitting
        self.md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ]
        )

        # Level 2: Semantic chunking (groups by meaning)
        if use_semantic:
            self.semantic_splitter = SemanticChunker(
                embeddings=OpenAIEmbeddings(),
                breakpoint_threshold_type="percentile",  # or "standard_deviation"
                breakpoint_threshold_amount=95
            )

        # Level 3: Fallback recursive splitting
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,          # Optimal for RAG: 256-512 tokens
            chunk_overlap=50,        # ~10% overlap
            separators=[
                "\n\n",              # Paragraph breaks
                "\n",                # Line breaks
                ". ",                # Sentences
                "? ",
                "! ",
                "; ",
                ", ",
                " ",
                ""
            ],
            length_function=len,
        )

    def chunk(self, markdown_text: str, document_title: str) -> list[Chunk]:
        chunks = []

        # Step 1: Split by markdown headers (preserves section context)
        md_docs = self.md_splitter.split_text(markdown_text)

        for md_doc in md_docs:
            section_text = md_doc.page_content
            section_headers = md_doc.metadata  # {"h1": "...", "h2": "..."}

            # Step 2: Further split large sections
            if len(section_text) > 1000:
                if self.use_semantic:
                    sub_chunks = self.semantic_splitter.split_text(section_text)
                else:
                    sub_chunks = self.recursive_splitter.split_text(section_text)
            else:
                sub_chunks = [section_text]

            # Step 3: Create chunk objects with rich metadata
            for i, text in enumerate(sub_chunks):
                chunks.append(Chunk(
                    text=text,
                    document_title=document_title,
                    section_headers=section_headers,
                    chunk_index=len(chunks),
                    is_section_start=(i == 0)
                ))

        return chunks
```

**Chunk Metadata (Enhanced):**
```json
{
  "chunk_id": "uuid",
  "document_id": "uuid",
  "document_title": "EP001 Nutrition During Pregnancy",
  "section_h1": "Eating Well During Pregnancy",
  "section_h2": "Vitamins and Minerals",
  "page_numbers": [1, 2],
  "chunk_index": 0,
  "total_chunks": 15,
  "is_section_start": true,
  "text": "chunk content...",
  "token_count": 250
}
```

**Why This Approach?**

| Strategy | Pros | Best For |
|----------|------|----------|
| Markdown Headers | Preserves logical structure, low compute | Well-structured docs |
| Semantic Chunking | Groups by meaning, 70% accuracy boost | Technical/knowledge docs |
| Recursive Fallback | Fast, predictable, no API calls | Any document type |

For medical documents with clear sections (symptoms, treatment, etc.), structure-aware chunking significantly improves retrieval relevance.

#### 2.3 Embedding Generation

**Option A: OpenAI (Production)**
```python
from openai import OpenAI
client = OpenAI()

response = client.embeddings.create(
    model="text-embedding-3-small",  # 1536 dimensions
    input=text_chunks
)
```

**Option B: Local/Free (Development)**
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
embeddings = model.encode(text_chunks)
```

### Image Processing Pipeline

```
Images ─▶ Resize/Optimize ─▶ Generate Caption ─▶ Optional CLIP Embed ─▶ Store
```

#### Image Captioning (for RAG context)
```python
# Option 1: Use multimodal LLM (Claude, GPT-4V)
# Option 2: Use specialized model (BLIP-2)
from transformers import Blip2Processor, Blip2ForConditionalGeneration

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
```

---

## Phase 3: Load

### Local Environment (SQLite + LanceDB)

**Why LanceDB over ChromaDB for Local Testing in 2025?**

| Feature | LanceDB | ChromaDB |
|---------|---------|----------|
| Architecture | Embedded, serverless (Rust) | Embedded + optional server |
| Performance | 100x faster than Parquet, handles 1B vectors | Good, but slower at scale |
| Storage Format | Lance (Parquet evolution) | Parquet via hnswlib |
| Multimodal | Native support (images, audio) | Text-focused |
| Full-text Search | Built-in BM25 hybrid search | Requires external setup |
| Incremental Updates | Efficient (zero-copy versioning) | Full re-index for updates |
| Ecosystem | Default for AnythingLLM, LlamaIndex | LangChain default |

**Key Advantage:** LanceDB has **native hybrid search** (BM25 + vector), eliminating the need for a separate BM25 implementation. This directly addresses the document title matching requirement.

**SQLite: Document & Image Metadata**
```sql
-- Documents table
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    title TEXT,
    file_hash TEXT NOT NULL,          -- SHA-256 for change detection
    page_count INTEGER,
    status TEXT DEFAULT 'pending',     -- pending/processing/completed/failed
    extracted_at TIMESTAMP,
    source_path TEXT,
    extraction_method TEXT,            -- 'pymupdf' or 'google_docai'
    fallback_reason TEXT               -- Why OCR fallback was triggered
);

-- Chunks table (text stored in LanceDB, metadata here for queries)
CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER,
    section_h1 TEXT,
    section_h2 TEXT,
    page_numbers TEXT,  -- JSON array
    token_count INTEGER,
    created_at TIMESTAMP
);

-- Images table
CREATE TABLE images (
    id TEXT PRIMARY KEY,
    document_id TEXT REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER,
    file_path TEXT,
    caption TEXT,
    position TEXT,  -- JSON object
    created_at TIMESTAMP
);

-- Index for incremental processing
CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_hash ON documents(file_hash);
```

**LanceDB: Vector Store with Hybrid Search**
```python
import lancedb
from lancedb.pydantic import LanceModel, Vector
from lancedb.embeddings import get_registry

# Define schema with Pydantic
class ChunkModel(LanceModel):
    id: str
    document_id: str
    document_title: str        # Used for BM25 matching
    section_h1: str | None
    section_h2: str | None
    text: str                  # Full text for BM25 + display
    page_numbers: list[int]
    chunk_index: int
    vector: Vector(1536)       # OpenAI embedding dimension

# Connect to local database
db = lancedb.connect("./data/lancedb")

# Create or open table
table = db.create_table("chunks", schema=ChunkModel, mode="overwrite")

# Add documents
table.add([
    {
        "id": chunk.id,
        "document_id": chunk.document_id,
        "document_title": chunk.document_title,
        "section_h1": chunk.section_h1,
        "section_h2": chunk.section_h2,
        "text": chunk.text,
        "page_numbers": chunk.page_numbers,
        "chunk_index": chunk.chunk_index,
        "vector": chunk.embedding
    }
    for chunk in chunks
])

# Create FTS index for BM25 search on title and text
table.create_fts_index(["document_title", "text"])
```

### Production Environment (Supabase)

**Supabase Setup (pgvector)**
```sql
-- Enable vector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Documents table
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    filename TEXT NOT NULL,
    title TEXT,
    page_count INTEGER,
    extracted_at TIMESTAMPTZ DEFAULT NOW(),
    metadata JSONB
);

-- Chunks with vector embeddings
CREATE TABLE chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER,
    page_numbers INTEGER[],
    content TEXT,
    token_count INTEGER,
    embedding VECTOR(1536),  -- OpenAI embedding dimension
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for vector similarity search
CREATE INDEX ON chunks USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Images metadata (files stored in Supabase Storage)
CREATE TABLE images (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    page_number INTEGER,
    storage_path TEXT,  -- Supabase Storage path
    caption TEXT,
    position JSONB,
    clip_embedding VECTOR(512),  -- Optional CLIP embedding
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Supabase Python Client**
```python
from supabase import create_client
import os

supabase = create_client(
    os.environ["SUPABASE_URL"],
    os.environ["SUPABASE_KEY"]
)

# Insert document
supabase.table("documents").insert({
    "filename": "EP001.pdf",
    "title": "Nutrition During Pregnancy",
    "page_count": 4
}).execute()

# Vector similarity search
result = supabase.rpc("match_chunks", {
    "query_embedding": query_vector,
    "match_threshold": 0.7,
    "match_count": 10
}).execute()
```

---

## Project Structure

```
ETL-pdf-pipeline/
├── pdfs/                       # Source PDF files (dynamic)
├── plans/                      # Planning documents
├── src/
│   ├── __init__.py
│   ├── config.py               # Configuration management
│   ├── models.py               # Pydantic models (DocumentState, Chunk, etc.)
│   ├── extract/
│   │   ├── __init__.py
│   │   ├── base_extractor.py   # Base extractor interface
│   │   ├── pymupdf_extractor.py
│   │   ├── google_docai_extractor.py
│   │   ├── image_extractor.py
│   │   └── quality_checker.py  # Extraction quality heuristics
│   ├── transform/
│   │   ├── __init__.py
│   │   ├── markdown_parser.py  # Parse markdown files with frontmatter
│   │   ├── text_cleaner.py     # Text cleaning utilities
│   │   ├── chunker.py          # Hybrid chunking (markdown + semantic)
│   │   ├── embedder.py         # Embedding generation
│   │   └── image_processor.py  # Image optimization & captioning
│   ├── load/
│   │   ├── __init__.py
│   │   ├── base_loader.py      # Loader interface
│   │   ├── local_loader.py     # SQLite + LanceDB
│   │   └── supabase_loader.py  # Production loader
│   ├── retrieve/
│   │   ├── __init__.py
│   │   └── hybrid_retriever.py # RAG retriever with BM25 + vector
│   └── pipeline.py             # Main orchestration (incremental)
├── data/
│   ├── markdown/               # Extracted markdown files (checkpoint)
│   ├── lancedb/                # LanceDB persistent storage
│   ├── sqlite/                 # SQLite database
│   └── images/                 # Extracted images by document
│       └── {document_id}/      # Images organized per document
├── tests/
│   ├── test_extract.py
│   ├── test_transform.py
│   ├── test_load.py
│   └── test_retriever.py
├── scripts/
│   ├── run_pipeline.py         # CLI entry point
│   ├── run_single.py           # Process single PDF (for future API)
│   ├── reprocess_markdown.py   # Re-transform from existing markdown
│   └── setup_supabase.sql      # Production DB setup
├── requirements.txt
├── .env.example
└── README.md
```

---

## Implementation Phases

### Phase 1: Core Extraction
- [ ] Set up project structure
- [ ] Implement PyMuPDF text extraction
- [ ] Implement image extraction and storage
- [ ] Create extraction tests with sample PDFs

### Phase 2: Transformation
- [ ] Implement text cleaning pipeline
- [ ] Implement chunking with configurable parameters
- [ ] Set up embedding generation (local model first)
- [ ] Implement image captioning (optional)

### Phase 3: Local Loading
- [ ] Design and create SQLite schema
- [ ] Set up LanceDB with persistence
- [ ] Implement local loader
- [ ] Create retrieval/query interface for testing

### Phase 4: Production Integration
- [ ] Set up Supabase project with pgvector
- [ ] Create production database schema
- [ ] Implement Supabase loader
- [ ] Configure image storage (Supabase Storage)
- [ ] Add environment-based configuration switching

### Phase 5: Optimization & Polish
- [ ] Add Google Document AI fallback (optional)
- [ ] Implement batch processing for large collections
- [ ] Add progress tracking and logging
- [ ] Create CLI with options for partial runs
- [ ] Write comprehensive documentation

---

## Dependencies

```
# requirements.txt

# PDF Processing
pymupdf>=1.24.0
pymupdf4llm>=0.0.5

# Text Processing
langchain>=0.2.0
langchain-experimental>=0.0.47   # For SemanticChunker
langchain-openai>=0.1.0
tiktoken>=0.5.0

# Embeddings
sentence-transformers>=2.2.0
openai>=1.0.0

# Vector Database (Local) - LanceDB with native hybrid search
lancedb>=0.4.0

# Database
supabase>=2.0.0

# Image Processing
Pillow>=10.0.0
transformers>=4.35.0  # For BLIP-2 captioning (optional)

# Utilities
python-dotenv>=1.0.0
tqdm>=4.65.0
pydantic>=2.0.0

# Google Cloud (Optional)
google-cloud-documentai>=2.20.0
```

---

## Configuration

```python
# .env.example

# Environment
ENVIRONMENT=local  # local | production

# OpenAI (for embeddings)
OPENAI_API_KEY=sk-...

# Supabase (production)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=eyJ...

# Google Cloud (optional)
GOOGLE_PROJECT_ID=your-project
GOOGLE_LOCATION=us
GOOGLE_PROCESSOR_ID=xxx

# Pipeline Settings
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
EMBEDDING_MODEL=text-embedding-3-small  # or all-MiniLM-L6-v2 for local
```

---

## Input Validation

PDF files must be validated before processing to prevent crashes and ensure data quality.

```python
from enum import Enum
from pathlib import Path
import fitz

class ValidationResult(Enum):
    VALID = "valid"
    FILE_TOO_LARGE = "file_too_large"
    TOO_MANY_PAGES = "too_many_pages"
    CORRUPTED = "corrupted"
    PASSWORD_PROTECTED = "password_protected"
    EMPTY = "empty"

class PDFValidator:
    MAX_FILE_SIZE_MB = 50
    MAX_PAGE_COUNT = 500

    def validate(self, pdf_path: Path) -> ValidationResult:
        # Check file size
        size_mb = pdf_path.stat().st_size / (1024 * 1024)
        if size_mb > self.MAX_FILE_SIZE_MB:
            return ValidationResult.FILE_TOO_LARGE

        # Check if valid PDF
        try:
            doc = fitz.open(pdf_path)
        except Exception:
            return ValidationResult.CORRUPTED

        # Check page count
        if doc.page_count > self.MAX_PAGE_COUNT:
            return ValidationResult.TOO_MANY_PAGES

        # Check if password protected
        if doc.needs_pass:
            return ValidationResult.PASSWORD_PROTECTED

        # Check for content
        if doc.page_count == 0:
            return ValidationResult.EMPTY

        doc.close()
        return ValidationResult.VALID
```

---

## Structured Logging

JSON-formatted logging for debugging and log aggregation.

```python
# src/logging_config.py
import logging
import json
from datetime import datetime
from pathlib import Path

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
        }
        # Add extra fields if present
        for key in ["document_id", "duration_ms", "chunk_count", "error"]:
            if hasattr(record, key):
                log_data[key] = getattr(record, key)
        return json.dumps(log_data)

def setup_logging(log_dir: Path = Path("logs")):
    log_dir.mkdir(exist_ok=True)

    logger = logging.getLogger("etl_pipeline")
    logger.setLevel(logging.INFO)

    # File handler with JSON format
    file_handler = logging.FileHandler(log_dir / "pipeline.json")
    file_handler.setFormatter(JSONFormatter())

    # Console handler for human-readable output
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s"
    ))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
```

---

## Error Handling

**Mode: Stop on Failure** - The pipeline halts on any document failure for strict data integrity.

```python
class PipelineError(Exception):
    """Base exception for pipeline errors."""
    def __init__(self, message: str, document_id: str = None, phase: str = None):
        self.document_id = document_id
        self.phase = phase
        super().__init__(message)

class ExtractionError(PipelineError):
    """Raised when PDF extraction fails."""
    pass

class TransformationError(PipelineError):
    """Raised when chunking or embedding fails."""
    pass

class LoadError(PipelineError):
    """Raised when database loading fails."""
    pass

class Pipeline:
    def process_batch(self, pdf_paths: list[Path]):
        """Process batch with stop-on-failure mode."""
        for i, pdf_path in enumerate(pdf_paths):
            logger.info(f"Processing {i+1}/{len(pdf_paths)}: {pdf_path.name}")
            try:
                self.process_document(pdf_path)
            except PipelineError as e:
                logger.error(f"Pipeline failed at {pdf_path}", extra={
                    "document_id": e.document_id,
                    "phase": e.phase,
                    "error": str(e)
                })
                raise  # Stop entire batch
        logger.info(f"Batch complete: {len(pdf_paths)} documents processed")
```

---

## RAG Query Interface with Hybrid Search

### Hybrid Search Strategy

The retrieval system combines **three search modes**:

```
User Query ──┬──▶ Vector Search (semantic similarity)
             │
             ├──▶ BM25 Search (keyword matching on title + text)
             │
             └──▶ Metadata Filter (document title as tag)
                           │
                           ▼
                  Reciprocal Rank Fusion (RRF)
                           │
                           ▼
                    Re-ranked Results
```

### Document Title as Tag

Document titles serve as **semantic tags** for filtering and boosting:

| Document Title | Extracted Tags |
|----------------|----------------|
| EP001 Nutrition During Pregnancy | `["nutrition", "pregnancy", "diet"]` |
| FF633 COVID-19 and Pregnancy | `["covid-19", "pregnancy", "vaccination"]` |
| EP081 Urinary Incontinence | `["urinary", "incontinence", "bladder"]` |

**Title Matching Flow:**
```
Query: "What vitamins should I take during pregnancy?"
                    │
                    ▼
        ┌───────────────────────┐
        │  Title Keyword Match  │
        │  "pregnancy" found in │
        │  multiple doc titles  │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │   BM25 Boost Factor   │
        │  Title match = 2.0x   │
        │  Text match = 1.0x    │
        └───────────┬───────────┘
                    │
                    ▼
        ┌───────────────────────┐
        │  Pre-filter Option    │
        │  Only search docs     │
        │  with "pregnancy" in  │
        │  title (faster)       │
        └───────────────────────┘
```

### Implementation

```python
from dataclasses import dataclass
from enum import Enum
import lancedb

class SearchMode(Enum):
    VECTOR = "vector"           # Pure semantic search
    HYBRID = "hybrid"           # Vector + BM25 (recommended)
    KEYWORD = "keyword"         # Pure BM25

@dataclass
class SearchResult:
    chunk_id: str
    document_id: str
    document_title: str
    text: str
    page_numbers: list[int]
    score: float
    search_mode: str            # Which mode found this result

class HybridRAGRetriever:
    def __init__(self, lancedb_path: str = "./data/lancedb"):
        self.db = lancedb.connect(lancedb_path)
        self.table = self.db.open_table("chunks")
        self.embedder = OpenAIEmbeddings()

    def query(
        self,
        question: str,
        top_k: int = 10,
        mode: SearchMode = SearchMode.HYBRID,
        title_filter: str | None = None,
        title_boost: float = 2.0
    ) -> list[SearchResult]:
        """
        Retrieve relevant chunks using hybrid search.

        Args:
            question: User's question
            top_k: Number of results to return
            mode: Search mode (vector, hybrid, keyword)
            title_filter: Optional - only search docs with this in title
            title_boost: BM25 weight multiplier for title matches
        """

        # Optional: Pre-filter by document title (with SQL injection protection)
        base_query = self.table
        if title_filter:
            # Escape single quotes to prevent SQL injection
            safe_filter = title_filter.replace("'", "''").replace("%", "\\%")
            base_query = base_query.search().where(
                f"document_title LIKE '%{safe_filter}%'", prefilter=True
            )

        if mode == SearchMode.VECTOR:
            # Pure vector search
            query_embedding = self.embedder.embed_query(question)
            results = (
                base_query
                .search(query_embedding)
                .limit(top_k)
                .to_list()
            )

        elif mode == SearchMode.KEYWORD:
            # Pure BM25 full-text search
            results = (
                base_query
                .search(question, query_type="fts")  # Full-text search
                .limit(top_k)
                .to_list()
            )

        elif mode == SearchMode.HYBRID:
            # Hybrid: Vector + BM25 with Reciprocal Rank Fusion
            query_embedding = self.embedder.embed_query(question)

            results = (
                base_query
                .search(query_embedding, query_type="hybrid")
                .limit(top_k)
                .to_list()
            )

        return [
            SearchResult(
                chunk_id=r["id"],
                document_id=r["document_id"],
                document_title=r["document_title"],
                text=r["text"],
                page_numbers=r["page_numbers"],
                score=r.get("_score", r.get("_distance", 0)),
                search_mode=mode.value
            )
            for r in results
        ]

    def query_with_title_routing(
        self,
        question: str,
        top_k: int = 10
    ) -> list[SearchResult]:
        """
        Smart routing: detect document title mentions in query.

        Example: "What does the nutrition guide say about iron?"
                 → Detects "nutrition" → filters to nutrition-related docs
        """
        # Extract potential title keywords from question
        title_keywords = self._extract_title_keywords(question)

        if title_keywords:
            # User mentioned specific topic - use as filter + boost
            return self.query(
                question=question,
                top_k=top_k,
                mode=SearchMode.HYBRID,
                title_filter=title_keywords[0]  # Primary keyword
            )
        else:
            # General question - full hybrid search
            return self.query(
                question=question,
                top_k=top_k,
                mode=SearchMode.HYBRID
            )

    def _extract_title_keywords(self, question: str) -> list[str]:
        """
        Extract topic keywords that might match document titles.
        Could be enhanced with NER or keyword extraction models.
        """
        # Simple approach: match against known document topics
        known_topics = [
            "pregnancy", "nutrition", "diabetes", "vaccination",
            "contraception", "menopause", "fertility", "labor",
            "cesarean", "breastfeeding", "depression", "exercise"
        ]

        question_lower = question.lower()
        matches = [t for t in known_topics if t in question_lower]
        return matches

    def get_context(
        self,
        question: str,
        max_tokens: int = 4000,
        mode: SearchMode = SearchMode.HYBRID
    ) -> str:
        """Get formatted context for LLM prompt with source attribution."""
        results = self.query(question, mode=mode)

        context_parts = []
        total_tokens = 0
        seen_docs = set()

        for r in results:
            tokens = count_tokens(r.text)
            if total_tokens + tokens > max_tokens:
                break

            # Track unique documents for citation
            seen_docs.add(r.document_title)

            context_parts.append(
                f"[Source: {r.document_title}, Pages: {r.page_numbers}]\n{r.text}"
            )
            total_tokens += tokens

        context = "\n\n---\n\n".join(context_parts)

        # Add document list for transparency
        doc_list = "\n".join(f"- {doc}" for doc in seen_docs)
        return f"Documents referenced:\n{doc_list}\n\n---\n\n{context}"
```

### Query Examples

```python
retriever = HybridRAGRetriever()

# Example 1: General question (hybrid search)
results = retriever.query(
    "What are the symptoms of gestational diabetes?",
    mode=SearchMode.HYBRID
)

# Example 2: User mentions specific document topic
results = retriever.query_with_title_routing(
    "What does the pregnancy nutrition guide recommend for iron intake?"
)
# → Automatically filters to docs with "nutrition" or "pregnancy" in title

# Example 3: Explicit title filter
results = retriever.query(
    "What are the risks?",
    title_filter="COVID-19",  # Only search COVID-related docs
    mode=SearchMode.HYBRID
)

# Example 4: Keyword-only search (when user uses specific medical terms)
results = retriever.query(
    "preeclampsia hypertension",
    mode=SearchMode.KEYWORD  # BM25 better for exact medical terms
)
```

---

## References

- [PyMuPDF4LLM Documentation](https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/)
- [Google Cloud Document AI](https://codelabs.developers.google.com/codelabs/docai-ocr-python)
- [ChromaDB Tutorial](https://www.datacamp.com/tutorial/chromadb-tutorial-step-by-step-guide)
- [Supabase pgvector Guide](https://supabase.com/docs/guides/database/extensions/pgvector)
- [LangChain Text Splitters](https://python.langchain.com/docs/modules/data_connection/document_transformers/)
